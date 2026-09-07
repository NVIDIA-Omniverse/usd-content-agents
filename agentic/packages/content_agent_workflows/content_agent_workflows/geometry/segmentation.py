# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated mesh-segmentation artifact handoff for Geometry workflows."""

from __future__ import annotations

import hashlib
import json
import math
import mmap
import os
import re
import struct
from array import array
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.usd_cli_session import (
    validated_ovrtx_render_metadata,
)
from content_agent_workflows.common.validation_evidence import (
    EvidenceArtifact,
    ValidationCheck,
)
from content_agent_workflows.mesh_segmentation_contract import (
    MESH_SEGMENTATION_REQUIRED_SKILLS,
    required_mesh_segmentation_artifacts,
)

from .segmentation_routing import GeometrySegmentationRoutingDecision

SEGMENTATION_HANDOFF_SCHEMA_VERSION = (
    "content-agent-workflows.geometry-segmentation-handoff.v1"
)
SEGMENTATION_TERMINAL_SCHEMA_VERSION = (
    "content-agents.mesh-segmentation-terminal-validation.v1"
)
SEGMENTATION_EXPORT_SCHEMA_VERSION = "mesh-segmentation-fragment-export.v1"
SEGMENTATION_REQUEST_SCHEMA_VERSION = "content-agents.mesh-segmentation-request.v3"
SEGMENTATION_TOPOLOGY_SCHEMA_VERSION = "mesh-segmentation-topology.v1"
SEGMENTATION_RENDER_SCHEMA_VERSION = "mesh-segmentation-render-evidence.v2"
_MAX_JSON_BYTES = 16 * 1024 * 1024
# Match the deterministic routing authority. Audit structures are additionally
# compact and budgeted below so a producer cannot turn valid label metadata
# into an unbounded Python-object allocation.
_MAX_SOURCE_FACE_COUNT = 1_250_000
_MAX_LABEL_AUDIT_WORKING_BYTES = 64 * 1024 * 1024
_FRAGMENT_SEMANTIC_BYTES = 8
_PARTITION_SEEN_BYTES = 1
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_USD_CLI_CHECKPOINT_SCHEMA_VERSION = (
    "content-agent-workflows.mesh-evidence-usd-cli-checkpoint.v1"
)
_USD_CLI_RECEIPT_SCHEMA_VERSION = (
    "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
)
_MAX_RECEIPT_JOURNAL_BYTES = 64 * 1024 * 1024

SegmentationOutcome = Literal[
    "not_requested",
    "certified",
    "conditional",
    "rejected",
]


@dataclass(frozen=True)
class _UsdCliReceiptRecord:
    arguments: tuple[str, ...]
    response: dict[str, Any]
    artifact_bindings: tuple[dict[str, Any], ...]


class GeometrySegmentationPart(BaseModel):
    """One producer-owned semantic part bound to an exported mesh prim."""

    model_config = ConfigDict(extra="forbid")

    producer_segment_id: int = Field(ge=0)
    name: str = Field(min_length=1)
    output_prim_path: str = Field(min_length=1)
    source_face_count: int = Field(ge=1)
    output_point_count: int | None = Field(default=None, ge=1)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    def manifest_record(
        self,
        *,
        source_asset_sha256: str | None,
        outcome: SegmentationOutcome,
    ) -> dict[str, Any]:
        """Return an additive semantic-part record without freezing an ontology."""

        return {
            "name": self.name,
            "role": self.name,
            "source": "content-workflow-mesh-segmentation",
            "producer_segment_id": self.producer_segment_id,
            "output_prim_path": self.output_prim_path,
            "source_face_count": self.source_face_count,
            "confidence": self.confidence,
            "source_fidelity_tier": "exact_source_face_membership",
            "validation_state": outcome,
            "source_relation": {
                "source_asset_sha256": source_asset_sha256,
                "relation": "source_face_partition",
            },
        }


class GeometrySegmentationHandoff(BaseModel):
    """Digest-bound result consumed from one mesh-segmentation run directory."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SEGMENTATION_HANDOFF_SCHEMA_VERSION
    requested: bool
    required: bool
    outcome: SegmentationOutcome
    run_dir: str | None = None
    producer_run_id: str | None = None
    producer_workflow_mode: Literal["targeted", "recognition"] | None = None
    source_asset_sha256: str | None = None
    source_face_count: int | None = Field(default=None, ge=1)
    target_prim_path: str | None = None
    topology_digest: str | None = None
    fragment_labels_sha256: str | None = None
    face_labels_sha256: str | None = None
    segmented_usd_path: str | None = None
    segmented_usd_sha256: str | None = None
    request_path: str | None = None
    terminal_validation_path: str | None = None
    producer_manifest_path: str | None = None
    export_manifest_path: str | None = None
    source_topology_path: str | None = None
    fragment_labels_path: str | None = None
    face_labels_path: str | None = None
    topology_validation_path: str | None = None
    render_manifest_path: str | None = None
    routing_decision_path: str | None = None
    routing: GeometrySegmentationRoutingDecision | None = None
    render_session_id: str | None = None
    render_workspace_dir: str | None = None
    producer_schema_versions: dict[str, str] = Field(default_factory=dict)
    exporter_limitations: list[str] = Field(default_factory=list)
    parts: list[GeometrySegmentationPart] = Field(default_factory=list)
    failures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    def manifest_record(self) -> dict[str, Any]:
        """Return the provisional additive Geometry handoff extension."""

        return {
            "schema_version": self.schema_version,
            "contract_status": "provisional_pending_0_6_ontology_approval",
            "requested": self.requested,
            "required": self.required,
            "outcome": self.outcome,
            "run_dir": self.run_dir,
            "producer_run_id": self.producer_run_id,
            "producer_workflow_mode": self.producer_workflow_mode,
            "source_asset_sha256": self.source_asset_sha256,
            "source_face_count": self.source_face_count,
            "target_prim_path": self.target_prim_path,
            "topology_digest": self.topology_digest,
            "fragment_labels_sha256": self.fragment_labels_sha256,
            "face_labels_sha256": self.face_labels_sha256,
            "segmented_usd": {
                "path": self.segmented_usd_path,
                "sha256": self.segmented_usd_sha256,
            },
            "producer_schema_versions": dict(self.producer_schema_versions),
            "routing": (
                self.routing.model_dump(mode="json")
                if self.routing is not None
                else None
            ),
            "render_scope": (
                {
                    "session_id": self.render_session_id,
                    "workspace_dir": self.render_workspace_dir,
                }
                if self.render_session_id is not None
                and self.render_workspace_dir is not None
                else None
            ),
            "exporter_limitations": list(self.exporter_limitations),
            "parts": [
                part.manifest_record(
                    source_asset_sha256=self.source_asset_sha256,
                    outcome=self.outcome,
                )
                for part in self.parts
            ],
            "artifacts": {
                "request": self.request_path,
                "terminal_validation": self.terminal_validation_path,
                "producer_manifest": self.producer_manifest_path,
                "export_manifest": self.export_manifest_path,
                "source_topology": self.source_topology_path,
                "fragment_labels": self.fragment_labels_path,
                "face_labels": self.face_labels_path,
                "topology_validation": self.topology_validation_path,
                "render_manifest": self.render_manifest_path,
                "routing_decision": self.routing_decision_path,
            },
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"Missing {label}: {path}")
    if path.stat().st_size > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds {_MAX_JSON_BYTES} bytes: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _confined_file(
    run_dir: Path, relative: str, *, required: bool = True
) -> Path | None:
    resolved_run_dir = run_dir.resolve()
    raw_path = Path(relative)
    candidate = raw_path if raw_path.is_absolute() else resolved_run_dir / raw_path
    lexical = Path(os.path.abspath(candidate))
    try:
        run_relative = lexical.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError(f"Segmentation artifact escapes its run: {relative}") from exc

    current = resolved_run_dir
    for component in run_relative.parts:
        current /= component
        if current.is_symlink():
            raise ValueError(
                f"Segmentation artifact must not traverse a symlink: {relative}"
            )

    resolved = lexical.resolve()
    try:
        resolved.relative_to(resolved_run_dir)
    except ValueError as exc:
        raise ValueError(f"Segmentation artifact escapes its run: {relative}") from exc
    if not lexical.is_file():
        if required:
            raise ValueError(f"Missing segmentation artifact: {relative}")
        return None
    if lexical.stat(follow_symlinks=False).st_nlink != 1:
        raise ValueError(f"Segmentation artifact must not be a hardlink: {relative}")
    return resolved


def _required_confined_file(run_dir: Path, relative: str) -> Path:
    """Return one required run artifact without relying on an optimized assert."""

    path = _confined_file(run_dir, relative)
    if path is None:
        raise ValueError(f"Missing required segmentation artifact: {relative}")
    return path


def _sha256_value(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _declared_artifact_path(
    run_dir: Path,
    *,
    value: Any,
    expected: Path,
    label: str,
) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} has an invalid path")
    declared = _confined_file(run_dir, value)
    if declared != expected:
        raise ValueError(f"{label} does not identify its canonical run artifact")


def _staged_file_record(
    run_dir: Path,
    *,
    record: Any,
    label: str,
) -> tuple[Path, str, str]:
    if not isinstance(record, dict):
        raise ValueError(f"Segmentation request lacks a valid {label} staging record")
    source_name = record.get("source_name")
    if not isinstance(source_name, str) or Path(source_name).name != source_name:
        raise ValueError(f"Segmentation {label} source_name is invalid")
    staged_path = record.get("staged_path")
    if not isinstance(staged_path, str) or not Path(staged_path).is_absolute():
        raise ValueError(f"Segmentation {label} staged_path is invalid")
    path = _confined_file(run_dir, staged_path)
    if path is None:
        raise ValueError(f"Segmentation {label} staged artifact is missing")
    expected_size = _positive_int(record.get("size_bytes"))
    if expected_size is None or path.stat().st_size != expected_size:
        raise ValueError(f"Segmentation {label} staged size is stale")
    expected_digest = _sha256_value(
        record.get("sha256"),
        label=f"Segmentation {label} staged digest",
    )
    if file_sha256(path) != expected_digest:
        raise ValueError(f"Segmentation {label} staged digest is stale")
    return path, staged_path, expected_digest


def _schema_version(payload: dict[str, Any]) -> str | None:
    value = payload.get("schema_version")
    return str(value) if isinstance(value, str) and value else None


def _is_openusd_diagnostic_error(exc: Exception) -> bool:
    try:
        from pxr import Tf
    except ImportError:
        return False
    return isinstance(exc, Tf.ErrorException)


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return int(value)


def _positive_int(value: Any) -> int | None:
    parsed = _nonnegative_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _plain_semantic_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    name = value.strip()
    if Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{label} must be a plain semantic name")
    return name


def _segment_records(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    raw_records = payload.get("segments")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError("segments.json must contain a non-empty segments array")
    records: dict[int, dict[str, Any]] = {}
    names: set[str] = set()
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, dict):
            raise ValueError(f"segments.json record {index} must be an object")
        segment_id = _nonnegative_int(raw.get("segment_id"))
        if segment_id is None:
            raise ValueError(f"segments.json record {index} has an invalid segment_id")
        name = _plain_semantic_name(raw.get("name"), label=f"segment {segment_id} name")
        if segment_id in records:
            raise ValueError(f"Duplicate segmentation ID: {segment_id}")
        if name in names:
            raise ValueError(f"Duplicate segmentation name: {name}")
        records[segment_id] = {**raw, "name": name}
        names.add(name)
    return records


def _exported_parts(
    export_manifest: dict[str, Any],
    segment_records: dict[int, dict[str, Any]],
) -> list[GeometrySegmentationPart]:
    raw_outputs = export_manifest.get("output_segments")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise ValueError("Export manifest must contain output_segments")
    parts: list[GeometrySegmentationPart] = []
    output_ids: set[int] = set()
    output_paths: set[str] = set()
    for index, raw in enumerate(raw_outputs):
        if not isinstance(raw, dict):
            raise ValueError(f"Exported segment {index} must be an object")
        segment_id = _nonnegative_int(raw.get("segment_id"))
        if segment_id is None or segment_id not in segment_records:
            raise ValueError(f"Exported segment {index} has no producer segment record")
        if segment_id in output_ids:
            raise ValueError(f"Duplicate exported segment ID: {segment_id}")
        prim_path = raw.get("output_prim_path")
        if not isinstance(prim_path, str) or not prim_path.startswith("/"):
            raise ValueError(f"Exported segment {segment_id} has an invalid prim path")
        if prim_path in output_paths:
            raise ValueError(f"Duplicate exported segment prim path: {prim_path}")
        source_face_count = _positive_int(raw.get("source_face_count"))
        if source_face_count is None:
            raise ValueError(
                f"Exported segment {segment_id} has an invalid source_face_count"
            )
        output_point_count_raw = raw.get("output_point_count")
        output_point_count = (
            None
            if output_point_count_raw is None
            else _positive_int(output_point_count_raw)
        )
        if output_point_count_raw is not None and output_point_count is None:
            raise ValueError(
                f"Exported segment {segment_id} has an invalid output_point_count"
            )
        source_record = segment_records[segment_id]
        output_name = raw.get("name")
        if output_name is not None and output_name != source_record["name"]:
            raise ValueError(
                f"Exported segment {segment_id} name does not match segments.json"
            )
        confidence = source_record.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, int | float)
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            raise ValueError(
                f"Segment {segment_id} confidence must be a finite number from 0 to 1"
            )
        parts.append(
            GeometrySegmentationPart(
                producer_segment_id=segment_id,
                name=str(source_record["name"]),
                output_prim_path=prim_path,
                source_face_count=source_face_count,
                output_point_count=output_point_count,
                confidence=float(confidence) if confidence is not None else None,
            )
        )
        output_ids.add(segment_id)
        output_paths.add(prim_path)
    if output_ids != set(segment_records):
        missing = sorted(set(segment_records).difference(output_ids))
        raise ValueError(f"Producer segments were not exported: {missing}")
    return parts


def _require_digest(payload: dict[str, Any], key: str, path: Path) -> str:
    expected = _sha256_value(
        payload.get(key),
        label=f"Export manifest {key}",
    )
    observed = file_sha256(path)
    if expected != observed:
        raise ValueError(f"{path.name} digest does not match the export manifest")
    return observed


def _validate_request_and_terminal(
    root: Path,
    *,
    request: dict[str, Any],
    terminal: dict[str, Any],
) -> tuple[
    Literal["targeted", "recognition"],
    str,
    Path,
    str,
    str | None,
    list[str],
]:
    if _schema_version(request) != SEGMENTATION_REQUEST_SCHEMA_VERSION:
        raise ValueError("Unsupported mesh-segmentation request schema")
    if request.get("workflow") != "mesh-segmentation.run":
        raise ValueError("Segmentation request names an unsupported workflow")
    mode = request.get("workflow_mode")
    if mode not in {"targeted", "recognition"}:
        raise ValueError("Segmentation request has an invalid workflow mode")
    run_id = request.get("run_id")
    if not isinstance(run_id, str) or _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("Segmentation request has an invalid run ID")
    if request.get("run_dir") != str(root):
        raise ValueError("Segmentation request run_dir does not match the consumed run")
    if run_id != root.name:
        raise ValueError("Segmentation request run_id does not match its run directory")
    required_skills = request.get("required_skills")
    if required_skills != list(MESH_SEGMENTATION_REQUIRED_SKILLS):
        raise ValueError("Segmentation request does not name the canonical skills")

    isolation = request.get("isolation")
    if not isinstance(isolation, dict) or any(
        isolation.get(key) != expected
        for key, expected in {
            "fresh_child_thread": True,
            "conversation_context_inherited": False,
            "prior_run_access_allowed": False,
            "working_directory": str(root),
            "permitted_evidence_root": str(root),
        }.items()
    ):
        raise ValueError("Segmentation request isolation is inconsistent with run_dir")

    inputs = request.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("Segmentation request lacks inputs")
    raw_vocabulary = inputs.get("target_semantic_parts")
    if not isinstance(raw_vocabulary, list):
        raise ValueError("Segmentation target vocabulary must be an array")
    target_vocabulary = [
        _plain_semantic_name(value, label="target semantic part")
        for value in raw_vocabulary
    ]
    if len(target_vocabulary) != len(set(target_vocabulary)):
        raise ValueError("Segmentation target vocabulary contains duplicates")
    if "other" in target_vocabulary:
        raise ValueError("Segmentation target vocabulary cannot redefine other")
    if (mode == "targeted") != bool(target_vocabulary):
        raise ValueError(
            "Segmentation workflow mode is inconsistent with its target vocabulary"
        )

    staging = request.get("input_staging")
    if not isinstance(staging, dict) or staging.get("mode") not in {
        "single_file_copy",
        "asset_bundle_copy",
    }:
        raise ValueError("Segmentation request lacks valid input staging")
    staged_path, staged_path_text, staged_sha = _staged_file_record(
        root,
        record=staging.get("asset"),
        label="source asset",
    )
    if inputs.get("asset") != staged_path_text:
        raise ValueError(
            "Segmentation request asset path does not match its staged source"
        )

    raw_input_references = inputs.get("reference_images")
    raw_staged_references = staging.get("reference_images")
    if not isinstance(raw_input_references, list) or not isinstance(
        raw_staged_references, list
    ):
        raise ValueError("Segmentation request has invalid staged references")
    if len(raw_input_references) != len(raw_staged_references):
        raise ValueError("Segmentation staged reference lists differ in length")
    for index, (input_path, record) in enumerate(
        zip(raw_input_references, raw_staged_references, strict=True)
    ):
        _, staged_reference_text, _ = _staged_file_record(
            root,
            record=record,
            label=f"reference image {index}",
        )
        if input_path != staged_reference_text:
            raise ValueError(
                f"Segmentation reference image {index} does not match its staging record"
            )

    continuation = staging.get("continuation")
    if inputs.get("continuation_seed") != continuation:
        raise ValueError("Segmentation continuation staging is inconsistent")
    if continuation is not None and not isinstance(continuation, dict):
        raise ValueError("Segmentation continuation staging is invalid")

    request_artifacts = request.get("required_final_artifacts")
    terminal_artifacts = terminal.get("required_artifacts")
    if not isinstance(request_artifacts, list) or not request_artifacts:
        raise ValueError("Segmentation request lacks required_final_artifacts")
    if request_artifacts != terminal_artifacts:
        raise ValueError(
            "Terminal required_artifacts do not match the authoritative request"
        )
    if any(not isinstance(item, str) or not item for item in request_artifacts):
        raise ValueError("Segmentation required artifacts contain an invalid path")
    if len(request_artifacts) != len(set(request_artifacts)):
        raise ValueError("Segmentation required artifacts contain duplicates")
    expected_artifacts = list(
        required_mesh_segmentation_artifacts(
            mode,
            resumable=continuation is not None,
        )
    )
    if request_artifacts != expected_artifacts:
        omitted = sorted(set(expected_artifacts).difference(request_artifacts))
        unexpected = sorted(set(request_artifacts).difference(expected_artifacts))
        raise ValueError(
            f"Segmentation request does not use the canonical {mode} artifact set; "
            f"omitted={omitted}, unexpected={unexpected}"
        )
    for relative in request_artifacts:
        _confined_file(root, relative)

    constraints = request.get("constraints")
    if not isinstance(constraints, dict):
        raise ValueError("Segmentation request lacks constraints")
    expected_constraints = {
        "source_asset_edits_allowed": False,
        "semantic_decision_unit": "immutable_fragment",
        "require_fragment_atomicity": True,
        "require_exact_face_provenance": True,
        "part_recognition_required": mode == "recognition",
    }
    for key, expected in expected_constraints.items():
        if constraints.get(key) != expected:
            raise ValueError(
                f"Segmentation request violates required constraint: {key}"
            )

    target_prim = inputs.get("target_prim")
    if target_prim is not None and (
        not isinstance(target_prim, str) or not target_prim.startswith("/")
    ):
        raise ValueError("Segmentation request has an invalid target prim")
    return (
        mode,
        run_id,
        staged_path,
        staged_sha,
        target_prim,
        target_vocabulary,
    )


def _validate_label_atomicity(
    *,
    fragment_labels: Path,
    face_labels: Path,
    source_face_count: int,
    segment_ids: set[int],
    expected_fragment_count: int,
) -> None:
    if source_face_count > _MAX_SOURCE_FACE_COUNT:
        raise ValueError(
            "Segmentation source face count exceeds the Geometry handoff audit "
            f"limit: {source_face_count} > {_MAX_SOURCE_FACE_COUNT}"
        )
    if expected_fragment_count < 1 or expected_fragment_count > source_face_count:
        raise ValueError(
            "Export fragment_count must be positive and no greater than the "
            "source face count"
        )
    estimated_working_bytes = (
        expected_fragment_count * _FRAGMENT_SEMANTIC_BYTES
        + source_face_count * _PARTITION_SEEN_BYTES
    )
    if estimated_working_bytes > _MAX_LABEL_AUDIT_WORKING_BYTES:
        raise ValueError(
            "Segmentation label audit working set exceeds the Geometry handoff "
            f"limit: {estimated_working_bytes} > "
            f"{_MAX_LABEL_AUDIT_WORKING_BYTES} bytes"
        )
    expected_bytes = source_face_count * 4
    semantic_by_fragment = array("q", [-1]) * expected_fragment_count
    observed_fragment_count = 0
    observed_segments: set[int] = set()
    chunk_bytes = 1024 * 1024
    with (
        fragment_labels.open("rb") as fragment_stream,
        face_labels.open("rb") as semantic_stream,
    ):
        if os.fstat(fragment_stream.fileno()).st_size != expected_bytes:
            raise ValueError("Fragment labels do not match the source face count")
        if os.fstat(semantic_stream.fileno()).st_size != expected_bytes:
            raise ValueError("Semantic labels do not match the source face count")
        while True:
            fragment_chunk = fragment_stream.read(chunk_bytes)
            semantic_chunk = semantic_stream.read(chunk_bytes)
            if not fragment_chunk and not semantic_chunk:
                break
            if len(fragment_chunk) != len(semantic_chunk):
                raise ValueError("Fragment and semantic label streams differ in length")
            for (fragment_id,), (segment_id,) in zip(
                struct.iter_unpack("<I", fragment_chunk),
                struct.iter_unpack("<I", semantic_chunk),
                strict=True,
            ):
                if segment_id not in segment_ids:
                    raise ValueError(
                        f"Face labels contain unknown semantic segment {segment_id}"
                    )
                if fragment_id >= expected_fragment_count:
                    raise ValueError(
                        "Fragment labels contain an ID outside fragment_count"
                    )
                prior = semantic_by_fragment[fragment_id]
                if prior == -1:
                    semantic_by_fragment[fragment_id] = segment_id
                    observed_fragment_count += 1
                elif prior != segment_id:
                    raise ValueError(
                        f"Semantic labels split immutable fragment {fragment_id}"
                    )
                observed_segments.add(segment_id)

    if observed_segments != segment_ids:
        raise ValueError("Semantic labels do not contain every declared segment")
    if observed_fragment_count == 0:
        raise ValueError("Fragment label stream is empty")
    if observed_fragment_count != expected_fragment_count:
        raise ValueError("Fragment IDs are not a dense zero-based range")


def _source_mesh_topology(
    source_usd: Path,
    *,
    requested_target_prim: str | None,
) -> tuple[
    str,
    str,
    list[tuple[float, float, float]],
    list[tuple[int, int, int]],
    str,
    float,
    dict[str, Any],
]:
    try:
        from pxr import Gf, Usd, UsdGeom
    except Exception as exc:
        raise ValueError(
            f"OpenUSD is required to verify segmentation source: {exc}"
        ) from exc

    stage = Usd.Stage.Open(str(source_usd))
    if stage is None:
        raise ValueError(f"Could not open segmentation source USD: {source_usd}")
    traversed_prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    geometry_prims = [
        prim
        for prim in traversed_prims
        if prim.IsA(UsdGeom.Boundable) and prim.IsActive()
    ]
    if requested_target_prim:
        prim = next(
            (
                candidate
                for candidate in traversed_prims
                if str(candidate.GetPath()) == requested_target_prim
            ),
            None,
        )
        if prim is None or not prim.IsA(UsdGeom.Mesh):
            raise ValueError(
                "Segmentation target is not an active, defined source UsdGeomMesh "
                f"in the composed render traversal: {requested_target_prim}"
            )
        unrelated_geometry_paths = sorted(
            str(candidate.GetPath())
            for candidate in geometry_prims
            if candidate.GetPath() != prim.GetPath()
        )
        if unrelated_geometry_paths:
            raise ValueError(
                "Targeted segmentation cannot replace a source containing unrelated "
                "geometry; preserve or compose these prims before Geometry handoff: "
                f"{unrelated_geometry_paths}"
            )
        mesh = UsdGeom.Mesh(prim)
    else:
        meshes = [
            UsdGeom.Mesh(prim) for prim in traversed_prims if prim.IsA(UsdGeom.Mesh)
        ]
        if len(meshes) != 1:
            raise ValueError(
                "Segmentation source must contain exactly one mesh when target_prim is omitted"
            )
        mesh = meshes[0]
        unrelated_geometry_paths = sorted(
            str(candidate.GetPath())
            for candidate in geometry_prims
            if candidate.GetPath() != mesh.GetPath()
        )
        if unrelated_geometry_paths:
            raise ValueError(
                "Segmentation source contains unsupported unrelated geometry: "
                f"{unrelated_geometry_paths}"
            )

    _reject_time_samples(mesh.GetPrim(), label="Segmentation source mesh")
    source_render_semantics = _mesh_render_semantics(mesh, usd_geom=UsdGeom)
    if source_render_semantics["subdivision_scheme"] != str(UsdGeom.Tokens.none):
        raise ValueError(
            "Mesh segmentation source must use subdivisionScheme=none so source "
            "faces identify rendered triangles exactly"
        )
    if source_render_semantics["hole_indices"]:
        raise ValueError(
            "Mesh segmentation source must not use holeIndices because source "
            "face membership must identify rendered triangles exactly"
        )
    counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
    if not counts or any(int(count) != 3 for count in counts):
        raise ValueError("Mesh segmentation source must be a nonempty triangular mesh")
    raw_indices = list(mesh.GetFaceVertexIndicesAttr().Get() or [])
    if len(raw_indices) != len(counts) * 3:
        raise ValueError("Segmentation source has inconsistent triangle indices")
    raw_points = list(mesh.GetPointsAttr().Get() or [])
    if not raw_points:
        raise ValueError("Segmentation source mesh has no points")
    matrix = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(
        mesh.GetPrim()
    )
    points: list[tuple[float, float, float]] = []
    digest = hashlib.sha256()
    for raw in raw_points:
        point = matrix.Transform(Gf.Vec3d(*(float(value) for value in raw)))
        packed = struct.pack("<fff", *(float(value) for value in point))
        digest.update(packed)
        points.append(struct.unpack("<fff", packed))
    indices: list[int] = []
    for raw_index in raw_indices:
        index = int(raw_index)
        if index < 0 or index >= len(points):
            raise ValueError("Segmentation source has an out-of-range point index")
        digest.update(struct.pack("<i", index))
        indices.append(index)
    triangles = [
        (indices[offset], indices[offset + 1], indices[offset + 2])
        for offset in range(0, len(indices), 3)
    ]
    return (
        str(mesh.GetPath()),
        f"sha256:{digest.hexdigest()}",
        points,
        triangles,
        str(UsdGeom.GetStageUpAxis(stage)),
        float(UsdGeom.GetStageMetersPerUnit(stage)),
        source_render_semantics,
    )


def _mesh_render_semantics(mesh: Any, *, usd_geom: Any) -> dict[str, Any]:
    """Return render-affecting mesh state that an exact partition must preserve."""

    imageable = usd_geom.Imageable(mesh.GetPrim())
    return {
        "subdivision_scheme": str(mesh.GetSubdivisionSchemeAttr().Get()),
        "orientation": str(mesh.GetOrientationAttr().Get()),
        "double_sided": bool(mesh.GetDoubleSidedAttr().Get()),
        "hole_indices": tuple(
            int(value) for value in (mesh.GetHoleIndicesAttr().Get() or [])
        ),
        "visibility": str(imageable.ComputeVisibility()),
        "purpose": str(imageable.ComputePurpose()),
    }


def _face_signature(
    points: list[tuple[float, float, float]],
    triangle: tuple[int, int, int],
) -> bytes:
    return b"".join(struct.pack("<fff", *points[index]) for index in triangle)


def _reject_time_samples(prim: Any, *, label: str) -> None:
    """Reject animation on a mesh or any ancestor that affects its transform."""

    current = prim
    while current and not current.IsPseudoRoot():
        for attribute in current.GetAttributes():
            if attribute.GetNumTimeSamples() != 0:
                raise ValueError(
                    f"{label} must be static; time-sampled attribute found at "
                    f"{attribute.GetPath()}"
                )
        current = current.GetParent()


def _contains_asset_path(value: Any, *, sdf: Any) -> bool:
    if isinstance(value, sdf.AssetPath):
        return bool(value.path)
    if isinstance(value, sdf.AssetPathArray):
        return any(bool(item.path) for item in value)
    if isinstance(value, Mapping):
        return any(
            _contains_asset_path(item, sdf=sdf)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, list | tuple | set | frozenset):
        return any(_contains_asset_path(item, sdf=sdf) for item in value)
    return False


def _self_contained_usd_layer(segmented_usd: Path, *, sdf: Any) -> Any:
    layer = sdf.Layer.FindOrOpen(str(segmented_usd))
    if layer is None:
        raise ValueError(f"Could not open segmented USD layer: {segmented_usd}")
    if layer.subLayerPaths:
        raise ValueError(
            "Segmented USD must be self-contained; sublayers are forbidden"
        )
    if layer.GetExternalReferences() or layer.GetExternalAssetDependencies():
        raise ValueError(
            "Segmented USD must be self-contained; external dependencies are forbidden"
        )

    violations: list[str] = []

    def inspect(path: Any) -> None:
        spec = layer.GetObjectAtPath(path)
        if spec is None:
            return
        keys = spec.ListInfoKeys()
        if "references" in keys:
            violations.append(f"reference arc at {path}")
        if "payload" in keys:
            violations.append(f"payload arc at {path}")
        if any(_contains_asset_path(spec.GetInfo(key), sdf=sdf) for key in keys):
            violations.append(f"asset path at {path}")

    layer.Traverse(sdf.Path.absoluteRootPath, inspect)
    if violations:
        raise ValueError(
            "Segmented USD must be self-contained; authored dependency found: "
            f"{violations[0]}"
        )
    return layer


def _validate_usd_partition(
    segmented_usd: Path,
    *,
    parts: list[GeometrySegmentationPart],
    source_face_count: int,
    source_points: list[tuple[float, float, float]],
    source_triangles: list[tuple[int, int, int]],
    face_labels: Path,
    source_up_axis: str,
    source_meters_per_unit: float,
    source_render_semantics: dict[str, Any],
) -> None:
    if source_face_count > _MAX_SOURCE_FACE_COUNT:
        raise ValueError(
            "Segmentation source face count exceeds the Geometry handoff audit limit: "
            f"{source_face_count} > {_MAX_SOURCE_FACE_COUNT}"
        )
    try:
        from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    except Exception as exc:
        raise ValueError(f"OpenUSD is required to verify segmented USD: {exc}") from exc

    root_layer = _self_contained_usd_layer(segmented_usd, sdf=Sdf)
    stage = Usd.Stage.Open(root_layer, load=Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"Could not open segmented USD: {segmented_usd}")
    if str(UsdGeom.GetStageUpAxis(stage)) != source_up_axis:
        raise ValueError("Segmented USD up axis does not match the source USD")
    if float(UsdGeom.GetStageMetersPerUnit(stage)) != source_meters_per_unit:
        raise ValueError("Segmented USD metersPerUnit does not match the source USD")
    declared_mesh_paths = {part.output_prim_path for part in parts}
    traversed_prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    unsupported_boundables = sorted(
        f"{prim.GetPath()} ({prim.GetTypeName()})"
        for prim in traversed_prims
        if prim.IsA(UsdGeom.Boundable) and not prim.IsA(UsdGeom.Mesh)
    )
    if unsupported_boundables:
        raise ValueError(
            "Segmented USD contains undeclared non-mesh boundable geometry: "
            f"{unsupported_boundables}"
        )
    observed_mesh_paths = {
        str(prim.GetPath()) for prim in traversed_prims if prim.IsA(UsdGeom.Mesh)
    }
    if observed_mesh_paths != declared_mesh_paths:
        undeclared = sorted(observed_mesh_paths.difference(declared_mesh_paths))
        missing = sorted(declared_mesh_paths.difference(observed_mesh_paths))
        raise ValueError(
            "Segmented USD mesh prims do not exactly match output_segments; "
            f"undeclared={undeclared}, missing={missing}"
        )
    seen = bytearray(source_face_count)
    observed = 0
    with face_labels.open("rb") as labels_stream:
        expected_label_bytes = source_face_count * 4
        if os.fstat(labels_stream.fileno()).st_size != expected_label_bytes:
            raise ValueError("Semantic labels do not match the source face count")
        try:
            labels_mapping = mmap.mmap(
                labels_stream.fileno(),
                expected_label_bytes,
                access=mmap.ACCESS_READ,
            )
        except (OSError, ValueError) as exc:
            raise ValueError("Could not map the bounded semantic label file") from exc
        with labels_mapping as labels:
            for part in parts:
                prim = stage.GetPrimAtPath(part.output_prim_path)
                if not prim or not prim.IsA(UsdGeom.Mesh):
                    raise ValueError(
                        f"Segment {part.name!r} output prim is not a UsdGeomMesh: "
                        f"{part.output_prim_path}"
                    )
                _reject_time_samples(
                    prim,
                    label=f"Segment {part.name!r} output mesh",
                )
                provenance = prim.GetAttribute("meshSegmentation:sourceFaceIds")
                if (
                    not provenance
                    or provenance.GetTypeName() != Sdf.ValueTypeNames.UIntArray
                ):
                    raise ValueError(
                        f"Segment {part.name!r} provenance must be an exact UIntArray"
                    )
                source_ids = provenance.Get()
                if not isinstance(source_ids, Vt.UIntArray) or (
                    len(source_ids) != part.source_face_count
                ):
                    raise ValueError(
                        f"Segment {part.name!r} source-face provenance does not match "
                        "its manifest"
                    )
                mesh = UsdGeom.Mesh(prim)
                output_render_semantics = _mesh_render_semantics(
                    mesh,
                    usd_geom=UsdGeom,
                )
                if output_render_semantics != source_render_semantics:
                    raise ValueError(
                        f"Segment {part.name!r} render semantics do not match the "
                        "source mesh; "
                        f"source={source_render_semantics}, "
                        f"output={output_render_semantics}"
                    )
                counts = list(mesh.GetFaceVertexCountsAttr().Get() or [])
                indices = [
                    int(value)
                    for value in (mesh.GetFaceVertexIndicesAttr().Get() or [])
                ]
                raw_points = list(mesh.GetPointsAttr().Get() or [])
                if part.output_point_count is not None and (
                    part.output_point_count != len(raw_points)
                ):
                    raise ValueError(
                        f"Segment {part.name!r} output_point_count does not match "
                        "its mesh"
                    )
                if len(counts) != len(source_ids) or any(
                    int(count) != 3 for count in counts
                ):
                    raise ValueError(
                        f"Segment {part.name!r} does not contain one triangle per "
                        "source face"
                    )
                if len(indices) != len(counts) * 3 or not raw_points:
                    raise ValueError(f"Segment {part.name!r} has invalid mesh topology")
                matrix = UsdGeom.XformCache(
                    Usd.TimeCode.Default()
                ).GetLocalToWorldTransform(prim)
                output_points: list[tuple[float, float, float]] = []
                for raw_point in raw_points:
                    point = matrix.Transform(
                        Gf.Vec3d(*(float(value) for value in raw_point))
                    )
                    output_points.append(
                        struct.unpack(
                            "<fff",
                            struct.pack("<fff", *(float(value) for value in point)),
                        )
                    )
                for output_face_index, face_id in enumerate(source_ids):
                    if not isinstance(face_id, int) or not (
                        0 <= face_id < source_face_count
                    ):
                        raise ValueError(
                            f"Segment {part.name!r} contains invalid source-face "
                            "provenance"
                        )
                    label_id = struct.unpack_from("<I", labels, face_id * 4)[0]
                    if label_id != part.producer_segment_id:
                        raise ValueError(
                            f"Segment {part.name!r} source face {face_id} is bound to "
                            f"face-label segment {label_id}, not "
                            f"{part.producer_segment_id}"
                        )
                    if seen[face_id]:
                        raise ValueError(
                            f"Source face {face_id} appears in multiple segments"
                        )
                    offset = output_face_index * 3
                    output_triangle = (
                        indices[offset],
                        indices[offset + 1],
                        indices[offset + 2],
                    )
                    if any(
                        index < 0 or index >= len(output_points)
                        for index in output_triangle
                    ):
                        raise ValueError(
                            f"Segment {part.name!r} has an invalid point index"
                        )
                    if _face_signature(
                        output_points, output_triangle
                    ) != _face_signature(
                        source_points,
                        source_triangles[face_id],
                    ):
                        raise ValueError(
                            f"Segment {part.name!r} changed source face geometry "
                            f"{face_id}"
                        )
                    seen[face_id] = 1
                    observed += 1
    if observed != source_face_count or 0 in seen:
        raise ValueError(
            "Segmented USD does not provide exact, non-overlapping source-face coverage"
        )


def _render_artifact(
    *,
    run_dir: Path,
    manifest_path: Path,
    value: Any,
    label: str,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Render evidence has an invalid {label} path")
    raw_path = Path(value)
    candidate = raw_path if raw_path.is_absolute() else manifest_path.parent / raw_path
    resolved = _confined_file(run_dir, str(candidate))
    if resolved is None:
        raise ValueError(f"Render evidence {label} artifact is missing")
    return resolved


def _render_digest(record: dict[str, Any], key: str, path: Path) -> None:
    expected = _sha256_value(
        record.get(key),
        label=f"Render evidence {key}",
    )
    if file_sha256(path) != expected:
        raise ValueError(f"Render evidence {path.name} does not match {key}")


def _renderer_name(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Render evidence has an invalid {label} renderer")
    return value.strip().lower()


def _normalized_camera_value(value: Any) -> Hashable:
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int | float):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Render evidence camera_state contains a nonfinite number")
        normalized = Decimal(str(value)).normalize()
        if normalized.is_zero():
            normalized = Decimal(0)
        return ("number", str(normalized))
    if isinstance(value, str):
        return ("string", value)
    if isinstance(value, list):
        return ("array", tuple(_normalized_camera_value(item) for item in value))
    if isinstance(value, dict):
        return (
            "object",
            tuple(
                (key, _normalized_camera_value(item))
                for key, item in sorted(value.items())
            ),
        )
    raise ValueError("Render evidence camera_state contains an invalid value")


def _validate_usd_cli_receipts(
    *,
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    segmented_usd: Path,
) -> tuple[str, Path, list[_UsdCliReceiptRecord]]:
    if manifest.get("scene_tool") != "usd-cli":
        raise ValueError("OVRTX render evidence is not attributed to usd-cli")
    session_id = manifest.get("usd_cli_session_id")
    if not isinstance(session_id, str) or _RUN_ID_PATTERN.fullmatch(session_id) is None:
        raise ValueError("OVRTX render evidence has an invalid usd-cli session ID")

    receipts = _render_artifact(
        run_dir=run_dir,
        manifest_path=manifest_path,
        value=manifest.get("usd_cli_command_receipts"),
        label="usd-cli command receipts",
    )
    checkpoint_path = _render_artifact(
        run_dir=run_dir,
        manifest_path=manifest_path,
        value=manifest.get("usd_cli_receipt_checkpoint"),
        label="usd-cli receipt checkpoint",
    )
    _render_digest(manifest, "usd_cli_command_receipts_sha256", receipts)
    _render_digest(manifest, "usd_cli_receipt_checkpoint_sha256", checkpoint_path)
    if receipts.stat().st_size > _MAX_RECEIPT_JOURNAL_BYTES:
        raise ValueError("usd-cli receipt journal exceeds the Geometry handoff limit")

    checkpoint = _read_json(checkpoint_path, label="usd-cli receipt checkpoint")
    receipt_stat = receipts.stat(follow_symlinks=False)
    expected_receipt_digest = _sha256_value(
        checkpoint.get("receipt_sha256"),
        label="usd-cli checkpoint receipt_sha256",
    )
    if (
        checkpoint.get("schema_version") != _USD_CLI_CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("workflow") != "mesh-segmentation"
        or checkpoint.get("session_id") != session_id
        or checkpoint.get("receipt_device") != receipt_stat.st_dev
        or checkpoint.get("receipt_inode") != receipt_stat.st_ino
        or checkpoint.get("receipt_size_bytes") != receipt_stat.st_size
        or expected_receipt_digest != file_sha256(receipts)
        or not isinstance(checkpoint.get("usd_cli_source_revision"), str)
        or not checkpoint["usd_cli_source_revision"]
    ):
        raise ValueError("usd-cli receipt checkpoint does not bind its journal")

    receipt_records: list[_UsdCliReceiptRecord] = []
    opened_expected_scene = False
    try:
        lines = receipts.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read usd-cli receipt journal: {exc}") from exc
    if not lines:
        raise ValueError("usd-cli receipt journal is empty")
    for index, line in enumerate(lines):
        try:
            receipt = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"usd-cli receipt {index} is invalid JSON") from exc
        arguments = receipt.get("arguments") if isinstance(receipt, dict) else None
        tool = receipt.get("tool") if isinstance(receipt, dict) else None
        response = receipt.get("response") if isinstance(receipt, dict) else None
        raw_artifact_bindings = (
            receipt.get("artifact_bindings") if isinstance(receipt, dict) else None
        )
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != _USD_CLI_RECEIPT_SCHEMA_VERSION
            or receipt.get("workflow") != "mesh-segmentation"
            or receipt.get("session_id") != session_id
            or receipt.get("status") != "completed"
            or not isinstance(arguments, list)
            or not arguments
            or any(not isinstance(value, str) or not value for value in arguments)
            or not isinstance(tool, dict)
            or tool.get("name") != "usd-cli"
            or tool.get("source_revision") != checkpoint["usd_cli_source_revision"]
            or not isinstance(response, dict)
            or not isinstance(raw_artifact_bindings, list)
            or any(not isinstance(item, dict) for item in raw_artifact_bindings)
        ):
            raise ValueError(f"usd-cli receipt {index} violates the evidence contract")
        if arguments[0] == "open" and len(arguments) == 2:
            try:
                opened_scene = Path(arguments[1]).expanduser().resolve()
            except OSError:
                opened_scene = None
            if opened_scene != segmented_usd.resolve():
                raise ValueError(
                    "usd-cli receipts open a scene other than the segmented render scene"
                )
            opened_expected_scene = True
        receipt_records.append(
            _UsdCliReceiptRecord(
                arguments=tuple(arguments),
                response=response,
                artifact_bindings=tuple(raw_artifact_bindings),
            )
        )
    if not opened_expected_scene:
        raise ValueError("usd-cli receipts do not open the segmented render scene")
    return session_id, receipts.parent, receipt_records


def _single_cli_option(arguments: tuple[str, ...], option: str, *, label: str) -> str:
    indexes = [index for index, value in enumerate(arguments) if value == option]
    if len(indexes) != 1 or indexes[0] + 1 >= len(arguments):
        raise ValueError(f"{label} must declare exactly one {option} value")
    return arguments[indexes[0] + 1]


def _numeric_sequence(value: Any, *, label: str) -> tuple[float, ...]:
    if isinstance(value, str):
        raw_values: Any = value.split(",")
    else:
        raw_values = value
    if not isinstance(raw_values, list | tuple) or not raw_values:
        raise ValueError(f"{label} must be a nonempty numeric sequence")
    try:
        values = tuple(float(item) for item in raw_values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a numeric sequence") from exc
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"{label} contains a nonfinite number")
    return values


def _matching_numeric_sequences(left: Any, right: Any, *, label: str) -> None:
    observed = _numeric_sequence(left, label=f"{label} receipt value")
    expected = _numeric_sequence(right, label=f"{label} camera value")
    if len(observed) != len(expected) or any(
        not math.isclose(actual, target, rel_tol=1e-9, abs_tol=1e-9)
        for actual, target in zip(observed, expected, strict=True)
    ):
        raise ValueError(f"{label} disagrees with its usd-cli camera receipt")


def _validate_camera_receipt(
    *,
    name: str,
    arguments: tuple[str, ...],
    camera: dict[str, Any],
    focus: str,
) -> None:
    state = camera.get("camera_state")
    if not isinstance(state, dict):
        raise ValueError(f"Segmentation render view {name} lacks camera_state")
    if arguments[:2] == ("camera", "orbit"):
        if len(arguments) < 3 or arguments[2] != focus:
            raise ValueError(
                f"Segmentation render view {name} camera focus disagrees with its "
                "usd-cli receipt"
            )
        for option, field in (
            ("--az", "yaw_degrees"),
            ("--el", "pitch_degrees"),
            ("--dist", "distance"),
        ):
            _matching_numeric_sequences(
                _single_cli_option(arguments, option, label=f"view {name} camera"),
                [state.get(field)],
                label=f"Segmentation render view {name} {field}",
            )
        if state.get("last_framed_prim_path") != focus:
            raise ValueError(
                f"Segmentation render view {name} camera state is not bound to focus"
            )
        return

    if arguments[:2] == ("camera", "create"):
        camera_name = _single_cli_option(
            arguments, "--name", label=f"view {name} camera"
        )
        camera_path = camera.get("camera_path")
        if (
            not isinstance(camera_path, str)
            or camera_path.rsplit("/", 1)[-1] != camera_name
        ):
            raise ValueError(
                f"Segmentation render view {name} camera path disagrees with its "
                "usd-cli receipt"
            )
        transform = camera.get("camera_world_transform")
        if (
            not isinstance(transform, list)
            or len(transform) != 4
            or not isinstance(transform[3], list)
            or len(transform[3]) < 3
        ):
            raise ValueError(
                f"Segmentation render view {name} lacks a valid camera transform"
            )
        _matching_numeric_sequences(
            _single_cli_option(arguments, "--at", label=f"view {name} camera"),
            transform[3][:3],
            label=f"Segmentation render view {name} position",
        )
        _matching_numeric_sequences(
            _single_cli_option(arguments, "--look-at", label=f"view {name} camera"),
            state.get("target"),
            label=f"Segmentation render view {name} target",
        )
        for option, field in (
            ("--focal", "focal_length"),
            ("--aperture", "horizontal_aperture"),
        ):
            _matching_numeric_sequences(
                _single_cli_option(arguments, option, label=f"view {name} camera"),
                [state.get(field)],
                label=f"Segmentation render view {name} {field}",
            )
        return

    raise ValueError(
        f"Segmentation render view {name} is not preceded by a supported usd-cli "
        "camera receipt"
    )


def _response_artifact_paths(
    *,
    run_dir: Path,
    manifest_path: Path,
    name: str,
    response: dict[str, Any],
) -> dict[str, Path]:
    raw_artifacts = response.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError(f"Segmentation render view {name} lacks usd-cli artifacts")
    paths: dict[str, Path] = {}
    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            continue
        label = raw_artifact.get("label")
        semantic_label = (
            "rgb"
            if isinstance(label, str) and (label == "rgb" or label.startswith("rgb:"))
            else label
        )
        if semantic_label not in {"rgb", "normals", "depth", "linear_depth"}:
            continue
        if semantic_label in paths:
            raise ValueError(
                f"Segmentation render view {name} has ambiguous {semantic_label} "
                "usd-cli artifacts"
            )
        paths[str(semantic_label)] = _render_artifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            value=raw_artifact.get("path"),
            label=f"view {name} usd-cli {semantic_label} artifact",
        )
    missing = {"rgb", "normals", "depth", "linear_depth"}.difference(paths)
    if missing:
        raise ValueError(
            f"Segmentation render view {name} lacks required usd-cli render "
            f"artifacts: {sorted(missing)}"
        )
    return paths


def _validate_receipted_artifact_bindings(
    *,
    run_dir: Path,
    manifest_path: Path,
    name: str,
    receipt: _UsdCliReceiptRecord,
    response_paths: dict[str, Path],
) -> None:
    binding_paths: dict[str, Path] = {}
    for raw_binding in receipt.artifact_bindings:
        label = raw_binding.get("label")
        semantic_label = (
            "rgb"
            if isinstance(label, str) and (label == "rgb" or label.startswith("rgb:"))
            else label
        )
        if semantic_label not in response_paths:
            continue
        semantic_label = str(semantic_label)
        if semantic_label in binding_paths:
            raise ValueError(
                f"Segmentation render view {name} has ambiguous receipted "
                f"{semantic_label} bindings"
            )
        path = _render_artifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            value=raw_binding.get("path"),
            label=f"view {name} receipted {semantic_label} artifact",
        )
        digest = _sha256_value(
            raw_binding.get("sha256"),
            label=f"view {name} receipted {semantic_label} sha256",
        )
        size_bytes = raw_binding.get("size_bytes")
        if (
            path != response_paths[semantic_label]
            or not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 1
            or path.stat().st_size != size_bytes
            or file_sha256(path) != digest
        ):
            raise ValueError(
                f"Segmentation render view {name} {semantic_label} bytes disagree "
                "with the usd-cli receipt"
            )
        binding_paths[semantic_label] = path
    missing = set(response_paths).difference(binding_paths)
    if missing:
        raise ValueError(
            f"Segmentation render view {name} lacks receipted artifact bindings: "
            f"{sorted(missing)}"
        )


def _validate_usd_cli_render_response(
    *,
    run_dir: Path,
    manifest_path: Path,
    name: str,
    response: dict[str, Any],
    camera: dict[str, Any],
    channel_records: dict[str, Any],
    image: Path,
    channel_artifacts: dict[str, dict[str, Path]],
    focus: str,
    width: int,
    height: int,
    record_renderer: str,
    receipt_records: list[_UsdCliReceiptRecord],
    consumed_render_receipts: set[int],
) -> None:
    if response.get("ok") is not True:
        raise ValueError(f"Segmentation render view {name} did not report success")
    try:
        metadata = validated_ovrtx_render_metadata(response)
    except RuntimeError as exc:
        raise ValueError(
            f"Segmentation render view {name} lacks complete OVRTX metadata: {exc}"
        ) from exc
    if metadata["backend"] != record_renderer:
        raise ValueError(
            f"Segmentation render view {name} renderer disagrees with its response"
        )
    matching_receipts = [
        (index, receipt.arguments)
        for index, receipt in enumerate(receipt_records)
        if index not in consumed_render_receipts
        and receipt.arguments[0] == "render"
        and receipt.response == response
    ]
    if len(matching_receipts) != 1:
        raise ValueError(
            f"Segmentation render view {name} is not uniquely bound to one usd-cli "
            "render receipt"
        )
    receipt_index, render_arguments = matching_receipts[0]
    consumed_render_receipts.add(receipt_index)
    if receipt_index == 0:
        raise ValueError(
            f"Segmentation render view {name} lacks a preceding camera receipt"
        )
    _validate_camera_receipt(
        name=name,
        arguments=receipt_records[receipt_index - 1].arguments,
        camera=camera,
        focus=focus,
    )
    expected_camera = {
        "image_width": width,
        "image_height": height,
        "renderer": "ovrtx",
    }
    mismatched_camera_fields = [
        field
        for field, expected in expected_camera.items()
        if camera.get(field) != expected
    ]
    if mismatched_camera_fields:
        raise ValueError(
            f"Segmentation render view {name} camera metadata disagrees with OVRTX: "
            f"{mismatched_camera_fields}"
        )
    if not isinstance(camera.get("camera_path"), str) or not camera["camera_path"]:
        raise ValueError(
            f"Segmentation render view {name} lacks an OVRTX camera prim path"
        )

    required_flags = {"--photoreal", "--depth", "--normals"}
    if not required_flags.issubset(render_arguments):
        raise ValueError(
            f"Segmentation render view {name} receipt lacks required render flags"
        )
    resolution = _single_cli_option(
        render_arguments, "--res", label=f"view {name} render receipt"
    )
    if resolution != f"{width}x{height}":
        raise ValueError(
            f"Segmentation render view {name} receipt has the wrong resolution"
        )
    output_value = _single_cli_option(
        render_arguments, "--output", label=f"view {name} render receipt"
    )
    output_path = Path(output_value).expanduser()
    if not output_path.is_absolute():
        raise ValueError(
            f"Segmentation render view {name} receipt output must be absolute"
        )
    output_path = output_path.resolve()
    if not output_path.is_dir() or not output_path.is_relative_to(run_dir.resolve()):
        raise ValueError(
            f"Segmentation render view {name} receipt output is not a confined "
            "directory"
        )
    artifact_paths = _response_artifact_paths(
        run_dir=run_dir,
        manifest_path=manifest_path,
        name=name,
        response=response,
    )
    _validate_receipted_artifact_bindings(
        run_dir=run_dir,
        manifest_path=manifest_path,
        name=name,
        receipt=receipt_records[receipt_index],
        response_paths=artifact_paths,
    )
    if any(
        path != output_path and not path.is_relative_to(output_path)
        for path in artifact_paths.values()
    ):
        raise ValueError(
            f"Segmentation render view {name} artifacts escape its receipted output"
        )
    expected_artifacts = {
        "rgb": image,
        "normals": channel_artifacts["normal"]["preview"],
        "depth": channel_artifacts["linear_depth"]["preview"],
        "linear_depth": channel_artifacts["linear_depth"]["raw"],
    }
    mismatched_artifacts = sorted(
        label
        for label, expected_path in expected_artifacts.items()
        if artifact_paths[label] != expected_path
    )
    if mismatched_artifacts:
        raise ValueError(
            f"Segmentation render view {name} manifest artifacts disagree with its "
            f"usd-cli response: {mismatched_artifacts}"
        )

    for channel_name in ("normal", "linear_depth"):
        channel = channel_records.get(channel_name)
        if not isinstance(channel, dict):
            raise ValueError(
                f"Segmentation render view {name} lacks {channel_name} metadata"
            )
        shape = channel.get("shape")
        if (
            channel.get("aov") != channel_name
            or channel.get("evidence_role") != "auxiliary_cpu_aov"
            or channel.get("final_render_evidence") is not False
            or not isinstance(channel.get("dtype"), str)
            or not channel["dtype"]
            or not isinstance(shape, list)
            or len(shape) < 2
            or shape[:2] != [height, width]
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 1
                for value in shape
            )
        ):
            raise ValueError(
                f"Segmentation render view {name} has invalid {channel_name} metadata"
            )
    depth = channel_records["linear_depth"]
    response_data = response.get("data")
    if (
        depth.get("encoding") != "camera_space_linear_depth"
        or depth.get("unit") != "meter"
        or depth.get("preview_encoding") != "per_frame_normalized_uint8"
        or not isinstance(response_data, dict)
        or response_data.get("linear_depth_unit") != "meter"
    ):
        raise ValueError(
            f"Segmentation render view {name} lacks metric linear-depth provenance"
        )


def _validate_render_evidence(
    *,
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    segmented_usd: Path,
    require_ovrtx: bool,
) -> tuple[bool, str | None, str | None]:
    if _schema_version(manifest) != SEGMENTATION_RENDER_SCHEMA_VERSION:
        raise ValueError("Unsupported mesh-segmentation render evidence schema")

    scene = _render_artifact(
        run_dir=run_dir,
        manifest_path=manifest_path,
        value=manifest.get("scene"),
        label="scene",
    )
    if scene != segmented_usd:
        raise ValueError("Segmentation renders are not bound to segmented.usdc")
    scene_sha256 = _sha256_value(
        manifest.get("scene_sha256"),
        label="Render evidence scene_sha256",
    )
    if file_sha256(scene) != scene_sha256:
        raise ValueError("Segmentation render scene digest is stale")

    for dimension in ("width", "height"):
        if _positive_int(manifest.get(dimension)) is None:
            raise ValueError(f"Render evidence has an invalid {dimension}")
    channels = manifest.get("auxiliary_cpu_aov_channels")
    required_channels = {"normal", "linear_depth"}
    if not isinstance(channels, list) or any(
        not isinstance(channel, str) for channel in channels
    ):
        raise ValueError("Render evidence has invalid render_channels")
    if not required_channels.issubset(channels):
        raise ValueError(
            "Render evidence lacks required normal and linear_depth channels"
        )

    raw_renders = manifest.get("renders")
    if not isinstance(raw_renders, list) or len(raw_renders) < 2:
        raise ValueError("Render evidence must contain at least two viewpoints")
    names: set[str] = set()
    camera_states: set[Hashable] = set()
    all_ovrtx = True
    session_id, receipt_dir, receipt_records = _validate_usd_cli_receipts(
        run_dir=run_dir,
        manifest_path=manifest_path,
        manifest=manifest,
        segmented_usd=segmented_usd,
    )
    focus = manifest.get("focus")
    if not isinstance(focus, str) or not focus.startswith("/"):
        raise ValueError("Render evidence has an invalid focus prim path")
    consumed_render_receipts: set[int] = set()
    for index, raw_record in enumerate(raw_renders):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Render evidence record {index} must be an object")
        name = raw_record.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Render evidence record {index} has an invalid name")
        if name in names:
            raise ValueError(f"Render evidence has duplicate view name: {name}")
        names.add(name)

        image = _render_artifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            value=raw_record.get("image"),
            label=f"view {name} image",
        )
        camera = _render_artifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            value=raw_record.get("camera"),
            label=f"view {name} camera",
        )
        response_path = _render_artifact(
            run_dir=run_dir,
            manifest_path=manifest_path,
            value=raw_record.get("response"),
            label=f"view {name} response",
        )
        _render_digest(raw_record, "image_sha256", image)
        _render_digest(raw_record, "camera_sha256", camera)
        _render_digest(raw_record, "response_sha256", response_path)

        camera_payload = _read_json(camera, label=f"view {name} camera")
        camera_state = camera_payload.get("camera_state")
        if not isinstance(camera_state, dict):
            raise ValueError(f"Render evidence view {name} lacks camera_state")
        camera_states.add(_normalized_camera_value(camera_state))
        response = _read_json(response_path, label=f"view {name} response")
        record_renderer = _renderer_name(
            raw_record.get("renderer"), label=f"view {name} manifest"
        )
        summary = response.get("summary")
        response_renderer = _renderer_name(
            summary.get("backend") if isinstance(summary, dict) else None,
            label=f"view {name} response",
        )
        if record_renderer != response_renderer:
            raise ValueError(
                f"Segmentation render view {name} renderer disagrees with its response"
            )
        render_is_ovrtx = record_renderer in {"ovrtx", "remote"}
        all_ovrtx = all_ovrtx and render_is_ovrtx
        if require_ovrtx and not render_is_ovrtx:
            raise ValueError(
                f"Segmentation render view {name} is not attributed to OVRTX"
            )

        raw_channel_records = raw_record.get("channels")
        if not isinstance(raw_channel_records, dict):
            raise ValueError(f"Render evidence view {name} has invalid channels")
        if not required_channels.issubset(raw_channel_records):
            raise ValueError(
                f"Render evidence view {name} lacks normal or linear_depth evidence"
            )
        channel_artifacts: dict[str, dict[str, Path]] = {}
        for channel_name in required_channels:
            channel_record = raw_channel_records[channel_name]
            if not isinstance(channel_record, dict):
                raise ValueError(
                    f"Render evidence view {name} channel {channel_name} is invalid"
                )
            channel_artifacts[channel_name] = {}
            for artifact_kind in ("raw", "preview"):
                artifact = _render_artifact(
                    run_dir=run_dir,
                    manifest_path=manifest_path,
                    value=channel_record.get(artifact_kind),
                    label=f"view {name} {channel_name} {artifact_kind}",
                )
                _render_digest(
                    channel_record,
                    f"{artifact_kind}_sha256",
                    artifact,
                )
                channel_artifacts[channel_name][artifact_kind] = artifact
        if render_is_ovrtx:
            _validate_usd_cli_render_response(
                run_dir=run_dir,
                manifest_path=manifest_path,
                name=name,
                response=response,
                camera=camera_payload,
                channel_records=raw_channel_records,
                image=image,
                channel_artifacts=channel_artifacts,
                focus=focus,
                width=int(manifest["width"]),
                height=int(manifest["height"]),
                record_renderer=record_renderer,
                receipt_records=receipt_records,
                consumed_render_receipts=consumed_render_receipts,
            )

    if len(camera_states) < 2:
        raise ValueError("Render evidence viewpoints do not use distinct cameras")
    return (
        all_ovrtx,
        session_id,
        str(receipt_dir),
    )


def segmentation_handoff_from_routing(
    *,
    decision: GeometrySegmentationRoutingDecision,
    expected_source_usd: Path | str,
    required: bool,
) -> GeometrySegmentationHandoff:
    """Convert a non-producer routing decision into fail-closed handoff state."""

    if decision.route == "consume_completed_run":
        raise ValueError(
            "Completed segmentation runs must be validated by "
            "consume_segmentation_handoff"
        )

    source = Path(expected_source_usd).expanduser().resolve()
    source_sha256 = file_sha256(source) if source.is_file() else None
    source_face_count = decision.metrics.face_count or None
    failures: list[str] = []
    warnings: list[str] = []

    if decision.route == "not_requested":
        outcome: SegmentationOutcome = "not_requested"
    elif decision.route == "reuse_source_identity":
        if decision.missing_required_semantic_names:
            failures.append(
                "Required semantic parts are missing from source identity: "
                f"{decision.missing_required_semantic_names}"
            )
        else:
            warnings.append(
                "Source-authored part identity bypassed mesh segmentation, but names "
                "alone do not certify semantic geometry."
            )
        outcome = "rejected" if failures else "conditional"
    elif decision.route == "deterministic_shell_split":
        failures.append(
            "The source requires a deterministic shell-split artifact before "
            "Geometry handoff; routing does not mutate the source asset."
        )
        outcome = "rejected"
    elif decision.route == "agentic_semantic_segmentation":
        failures.append(
            "The fused source requires a complete digest-bound semantic-segmentation "
            "run before Geometry handoff."
        )
        outcome = "rejected"
    else:
        failures.extend(decision.reasons)
        outcome = "rejected"

    return GeometrySegmentationHandoff(
        requested=decision.requested,
        required=required,
        outcome=outcome,
        source_asset_sha256=source_sha256,
        source_face_count=source_face_count,
        routing=decision,
        failures=failures,
        warnings=warnings,
    )


def consume_segmentation_handoff(
    *,
    run_dir: Path | str | None,
    expected_source_usd: Path | str,
    required: bool = False,
    required_parts: list[str] | None = None,
    require_ovrtx_evidence: bool = True,
) -> GeometrySegmentationHandoff:
    """Validate and consume producer artifacts without importing the CLI package."""

    requested = run_dir is not None
    if run_dir is None:
        failures: list[str] = (
            ["A valid mesh-segmentation run is required for this handoff."]
            if required
            else []
        )
        return GeometrySegmentationHandoff(
            requested=False,
            required=required,
            outcome="rejected" if failures else "not_requested",
            failures=failures,
        )

    requested_root = Path(run_dir).expanduser()
    root = requested_root.resolve()
    failures = []
    warnings: list[str] = []
    producer_versions: dict[str, str] = {}
    artifact_paths: dict[str, Path | None] = {}
    parts: list[GeometrySegmentationPart] = []
    source_sha256: str | None = None
    source_face_count: int | None = None
    target_prim_path: str | None = None
    topology_digest: str | None = None
    fragment_labels_sha256: str | None = None
    face_labels_sha256: str | None = None
    segmented_usd_sha256: str | None = None
    producer_run_id: str | None = None
    producer_workflow_mode: Literal["targeted", "recognition"] | None = None
    exporter_limitations: list[str] = []
    render_session_id: str | None = None
    render_workspace_dir: str | None = None

    try:
        if requested_root.is_symlink():
            raise ValueError("Segmentation run directory must not be a symlink")
        if not root.is_dir():
            raise ValueError(f"Segmentation run directory not found: {root}")
        terminal_path = _required_confined_file(root, "terminal_validation.json")
        terminal = _read_json(terminal_path, label="terminal validation")
        if _schema_version(terminal) != SEGMENTATION_TERMINAL_SCHEMA_VERSION:
            raise ValueError("Unsupported mesh-segmentation terminal schema")
        producer_versions["terminal_validation"] = SEGMENTATION_TERMINAL_SCHEMA_VERSION
        if terminal.get("valid") is not True:
            errors = terminal.get("semantic_validation_errors") or []
            raise ValueError(
                "Mesh-segmentation terminal validation rejected the run"
                + (f": {errors}" if errors else "")
            )
        request_path = _required_confined_file(root, "request.json")
        segments_path = _required_confined_file(root, "segments.json")
        source_topology_path = _required_confined_file(root, "prepare/topology.json")
        fragment_labels_path = _required_confined_file(
            root, "fragments/fragment_ids.u32le"
        )
        face_labels_path = _required_confined_file(root, "state/final_labels.u32le")
        export_path = _required_confined_file(root, "final/export_manifest.json")
        segmented_usd = _required_confined_file(root, "final/segmented.usdc")
        render_path = _required_confined_file(
            root, "final/renders/render_manifest.json"
        )
        rich_manifest_path = _confined_file(
            root,
            "final/segment_manifest.json",
            required=False,
        )
        topology_path = _confined_file(
            root,
            "final/topology_validation.json",
            required=False,
        )
        artifact_paths = {
            "request": request_path,
            "terminal": terminal_path,
            "producer_manifest": rich_manifest_path or segments_path,
            "export": export_path,
            "source_topology": source_topology_path,
            "fragment_labels": fragment_labels_path,
            "face_labels": face_labels_path,
            "topology": topology_path,
            "render": render_path,
            "segmented_usd": segmented_usd,
        }

        request = _read_json(request_path, label="segmentation request")
        (
            producer_workflow_mode,
            producer_run_id,
            staged_source_path,
            staged_source_sha256,
            requested_target_prim,
            target_vocabulary,
        ) = _validate_request_and_terminal(root, request=request, terminal=terminal)
        producer_versions["request"] = SEGMENTATION_REQUEST_SCHEMA_VERSION

        segments = _read_json(segments_path, label="segments manifest")
        source_topology = _read_json(
            source_topology_path,
            label="source topology",
        )
        export = _read_json(export_path, label="export manifest")
        render_manifest = _read_json(render_path, label="render manifest")
        if _schema_version(source_topology) != SEGMENTATION_TOPOLOGY_SCHEMA_VERSION:
            raise ValueError("Unsupported mesh-segmentation topology schema")
        producer_versions["source_topology"] = SEGMENTATION_TOPOLOGY_SCHEMA_VERSION
        if export.get("schema_version") != SEGMENTATION_EXPORT_SCHEMA_VERSION:
            raise ValueError("Unsupported mesh-segmentation export schema")
        producer_versions["export_manifest"] = SEGMENTATION_EXPORT_SCHEMA_VERSION
        if export.get("status") != "passed":
            raise ValueError("Mesh-segmentation export did not pass")
        if export.get("semantic_decision_unit") != "immutable_fragment":
            raise ValueError("Segmentation decisions were not immutable fragments")
        if export.get("exact_source_face_coverage") is not True:
            raise ValueError("Segmentation export lacks exact source-face coverage")
        if export.get("fragment_atomicity_conflict_ids") != []:
            raise ValueError("Segmentation export reports fragment atomicity conflicts")

        for value, expected_path, label in (
            (
                source_topology.get("source_asset"),
                staged_source_path,
                "Source topology source_asset",
            ),
            (
                export.get("source_asset"),
                staged_source_path,
                "Export manifest source_asset",
            ),
            (export.get("segments"), segments_path, "Export manifest segments"),
            (
                export.get("fragment_labels"),
                fragment_labels_path,
                "Export manifest fragment_labels",
            ),
            (
                export.get("face_labels"),
                face_labels_path,
                "Export manifest face_labels",
            ),
            (
                export.get("output_usd"),
                segmented_usd,
                "Export manifest output_usd",
            ),
        ):
            _declared_artifact_path(
                root,
                value=value,
                expected=expected_path,
                label=label,
            )

        expected_source = Path(expected_source_usd).expanduser().resolve()
        expected_source_sha256 = file_sha256(expected_source)
        source_sha256 = str(export.get("source_sha256") or "") or None
        if (
            source_sha256 != expected_source_sha256
            or staged_source_sha256 != expected_source_sha256
            or source_topology.get("source_sha256") != expected_source_sha256
        ):
            raise ValueError(
                "Segmentation source digest does not match the Geometry optimizer input"
            )
        source_face_count_raw = _positive_int(export.get("source_face_count"))
        if source_face_count_raw is None:
            raise ValueError("Segmentation export has an invalid source_face_count")
        source_face_count = source_face_count_raw
        if source_face_count > _MAX_SOURCE_FACE_COUNT:
            raise ValueError(
                "Segmentation source face count exceeds the Geometry handoff audit "
                f"limit: {source_face_count} > {_MAX_SOURCE_FACE_COUNT}"
            )
        (
            actual_target_prim,
            actual_topology_digest,
            source_points,
            source_triangles,
            source_up_axis,
            source_meters_per_unit,
            source_render_semantics,
        ) = _source_mesh_topology(
            expected_source,
            requested_target_prim=requested_target_prim,
        )
        target_prim_path = str(export.get("target_prim_path") or "") or None
        if (
            target_prim_path != actual_target_prim
            or source_topology.get("target_prim_path") != actual_target_prim
        ):
            raise ValueError("Segmentation target prim does not match the source mesh")
        if len(source_triangles) != source_face_count or (
            source_topology.get("source_face_count") != source_face_count
        ):
            raise ValueError("Segmentation source face count does not match source USD")
        topology_digest = str(export.get("topology_digest") or "") or None
        if (
            topology_digest != actual_topology_digest
            or source_topology.get("topology_digest") != actual_topology_digest
        ):
            raise ValueError("Segmentation topology digest does not match source USD")
        for label, record in (
            ("Source topology", source_topology),
            ("Export manifest", export),
        ):
            if record.get("up_axis") != source_up_axis:
                raise ValueError(f"{label} up_axis does not match source USD")
            declared_meters = record.get("meters_per_unit")
            if (
                isinstance(declared_meters, bool)
                or not isinstance(declared_meters, int | float)
                or not math.isfinite(float(declared_meters))
                or float(declared_meters) != source_meters_per_unit
            ):
                raise ValueError(f"{label} meters_per_unit does not match source USD")

        segment_records = _segment_records(segments)
        raw_segment_source = segments.get("source")
        if raw_segment_source is not None:
            if not isinstance(raw_segment_source, dict):
                raise ValueError("segments.json source provenance must be an object")
            _declared_artifact_path(
                root,
                value=raw_segment_source.get("path"),
                expected=staged_source_path,
                label="segments.json source path",
            )
            if raw_segment_source.get("sha256") != expected_source_sha256:
                raise ValueError(
                    "segments.json source digest does not match staged USD"
                )
            if raw_segment_source.get("face_count") != source_face_count:
                raise ValueError(
                    "segments.json source face count does not match source USD"
                )
        if producer_workflow_mode == "targeted":
            expected_names = {*target_vocabulary, "other"}
            observed_names = {
                str(record["name"]) for record in segment_records.values()
            }
            if observed_names != expected_names:
                raise ValueError(
                    "Targeted segmentation names do not match the exact request "
                    "vocabulary plus other"
                )
        _require_digest(export, "segments_sha256", segments_path)
        fragment_labels_sha256 = _require_digest(
            export,
            "fragment_labels_sha256",
            fragment_labels_path,
        )
        face_labels_sha256 = _require_digest(
            export,
            "face_labels_sha256",
            face_labels_path,
        )
        fragment_count = _positive_int(export.get("fragment_count"))
        if fragment_count is None:
            raise ValueError("Segmentation export has an invalid fragment_count")
        _validate_label_atomicity(
            fragment_labels=fragment_labels_path,
            face_labels=face_labels_path,
            source_face_count=source_face_count,
            segment_ids=set(segment_records),
            expected_fragment_count=fragment_count,
        )

        segmented_usd_sha256 = _require_digest(
            export,
            "output_usd_sha256",
            segmented_usd,
        )
        parts = _exported_parts(export, segment_records)
        if sum(part.source_face_count for part in parts) != source_face_count:
            raise ValueError(
                "Exported part face counts do not cover the source exactly"
            )
        _validate_usd_partition(
            segmented_usd,
            parts=parts,
            source_face_count=source_face_count,
            source_points=source_points,
            source_triangles=source_triangles,
            face_labels=face_labels_path,
            source_up_axis=source_up_axis,
            source_meters_per_unit=source_meters_per_unit,
            source_render_semantics=source_render_semantics,
        )

        raw_limitations = export.get("limitations")
        if not isinstance(raw_limitations, list) or any(
            not isinstance(item, str) or not item for item in raw_limitations
        ):
            raise ValueError("Segmentation export has invalid limitations")
        exporter_limitations = list(raw_limitations)

        if rich_manifest_path is not None:
            rich_manifest = _read_json(
                rich_manifest_path,
                label="semantic segment manifest",
            )
            version = _schema_version(rich_manifest)
            if version:
                producer_versions["segment_manifest"] = version
            raw_rich_parts = rich_manifest.get("segments")
            if not isinstance(raw_rich_parts, list) or not raw_rich_parts:
                raise ValueError("Semantic segment manifest has no segments")
            rich_names = {
                _plain_semantic_name(record.get("name"), label="semantic part name")
                for record in raw_rich_parts
                if isinstance(record, dict)
            }
            exported_names = {part.name for part in parts}
            if not rich_names.issubset(exported_names):
                raise ValueError(
                    "Semantic segment manifest names are not bound to exported parts"
                )

        missing_required = sorted(
            set(required_parts or []).difference(part.name for part in parts)
        )
        if missing_required:
            raise ValueError(f"Required semantic parts are missing: {missing_required}")

        (
            render_is_ovrtx,
            render_session_id,
            render_workspace_dir,
        ) = _validate_render_evidence(
            run_dir=root,
            manifest_path=render_path,
            manifest=render_manifest,
            segmented_usd=segmented_usd,
            require_ovrtx=require_ovrtx_evidence,
        )
        producer_versions["render_manifest"] = SEGMENTATION_RENDER_SCHEMA_VERSION
        if not require_ovrtx_evidence and not render_is_ovrtx:
            warnings.append(
                "Segmentation render evidence lacks explicit OVRTX attribution."
            )
        warnings.append(
            "The source-face partition passed mechanical handoff validation, but the "
            "0.6 semantic ontology and certification policy are still pending DG-03."
        )
    except Exception as exc:
        if not isinstance(
            exc,
            AttributeError
            | KeyError
            | OSError
            | OverflowError
            | RuntimeError
            | TypeError
            | ValueError,
        ) and not _is_openusd_diagnostic_error(exc):
            raise
        failures.append(str(exc))

    outcome: SegmentationOutcome = "rejected" if failures else "conditional"
    return GeometrySegmentationHandoff(
        requested=requested,
        required=required,
        outcome=outcome,
        run_dir=str(root),
        producer_run_id=producer_run_id,
        producer_workflow_mode=producer_workflow_mode,
        source_asset_sha256=source_sha256,
        source_face_count=source_face_count,
        target_prim_path=target_prim_path,
        topology_digest=topology_digest,
        fragment_labels_sha256=fragment_labels_sha256,
        face_labels_sha256=face_labels_sha256,
        segmented_usd_path=(
            str(artifact_paths["segmented_usd"])
            if artifact_paths.get("segmented_usd") and not failures
            else None
        ),
        segmented_usd_sha256=segmented_usd_sha256 if not failures else None,
        request_path=(
            str(artifact_paths["request"]) if artifact_paths.get("request") else None
        ),
        terminal_validation_path=(
            str(artifact_paths["terminal"]) if artifact_paths.get("terminal") else None
        ),
        producer_manifest_path=(
            str(artifact_paths["producer_manifest"])
            if artifact_paths.get("producer_manifest")
            else None
        ),
        export_manifest_path=(
            str(artifact_paths["export"]) if artifact_paths.get("export") else None
        ),
        source_topology_path=(
            str(artifact_paths["source_topology"])
            if artifact_paths.get("source_topology")
            else None
        ),
        fragment_labels_path=(
            str(artifact_paths["fragment_labels"])
            if artifact_paths.get("fragment_labels")
            else None
        ),
        face_labels_path=(
            str(artifact_paths["face_labels"])
            if artifact_paths.get("face_labels")
            else None
        ),
        topology_validation_path=(
            str(artifact_paths["topology"]) if artifact_paths.get("topology") else None
        ),
        render_manifest_path=(
            str(artifact_paths["render"]) if artifact_paths.get("render") else None
        ),
        render_session_id=render_session_id,
        render_workspace_dir=render_workspace_dir,
        producer_schema_versions=producer_versions,
        exporter_limitations=exporter_limitations,
        parts=parts if not failures else [],
        failures=failures,
        warnings=warnings,
    )


def segmentation_validation_check(
    handoff: GeometrySegmentationHandoff,
) -> tuple[ValidationCheck, list[EvidenceArtifact]]:
    """Map one consumed handoff to the shared validation evidence contract."""

    routing_path = handoff.routing_decision_path
    if not isinstance(routing_path, str) or not routing_path:
        raise ValueError(
            "Segmentation routing_decision_path is required at the validation "
            "evidence boundary"
        )
    if not Path(routing_path).is_file():
        raise ValueError(
            "Segmentation routing_decision_path does not identify a readable artifact"
        )

    artifacts: list[EvidenceArtifact] = []
    for kind, path in (
        ("mesh_segmentation_request", handoff.request_path),
        ("mesh_segmentation_terminal_validation", handoff.terminal_validation_path),
        ("mesh_segmentation_manifest", handoff.producer_manifest_path),
        ("mesh_segmentation_export_manifest", handoff.export_manifest_path),
        ("mesh_segmentation_source_topology", handoff.source_topology_path),
        ("mesh_segmentation_fragment_labels", handoff.fragment_labels_path),
        ("mesh_segmentation_face_labels", handoff.face_labels_path),
        ("mesh_segmentation_topology_validation", handoff.topology_validation_path),
        ("mesh_segmentation_render_manifest", handoff.render_manifest_path),
        ("mesh_segmentation_usd", handoff.segmented_usd_path),
        ("mesh_segmentation_routing", handoff.routing_decision_path),
    ):
        if path and Path(path).is_file():
            producer = (
                "content_agent_workflows.geometry"
                if kind == "mesh_segmentation_routing"
                else "content_workflow_cli.mesh_segmentation"
            )
            artifacts.append(
                EvidenceArtifact(
                    kind=kind,
                    path=path,
                    description="Digest-bound mesh-segmentation handoff artifact.",
                    metadata={
                        "producer": producer,
                        "claim_scope": "semantic_part_handoff",
                        "status": handoff.outcome,
                        "sha256": file_sha256(path),
                    },
                )
            )
    status: Literal["pass", "fail", "warning", "not_evaluated"] = (
        "not_evaluated"
        if handoff.outcome == "not_requested"
        else "pass"
        if handoff.outcome == "certified"
        else "warning"
        if handoff.outcome == "conditional"
        else "fail"
    )
    return (
        ValidationCheck(
            name="part_segregation",
            status=status,
            summary=(
                "Part Segregation was not requested."
                if handoff.outcome == "not_requested"
                else (
                    "Source-authored semantic identity bypassed mesh segmentation; "
                    "the identity is available for conditional downstream handoff "
                    "but is not semantic geometry certification."
                )
                if handoff.routing is not None
                and handoff.routing.route == "reuse_source_identity"
                and handoff.outcome == "conditional"
                else (
                    "Part Segregation produced a mechanically validated, digest-bound "
                    "source-face partition; semantic certification remains provisional "
                    f"for {len(handoff.parts)} parts."
                )
                if handoff.outcome in {"certified", "conditional"}
                else "Part Segregation artifacts were rejected before Geometry handoff."
            ),
            evidence_artifacts=artifacts,
            failures=list(handoff.failures),
            warnings=list(handoff.warnings),
            repair_hints=(
                [
                    "Resume or rerun the mesh-segmentation workflow, then provide its complete digest-bound run directory."
                ]
                if handoff.outcome == "rejected"
                else []
            ),
            metadata={
                "outcome": handoff.outcome,
                "required": handoff.required,
                "part_count": len(handoff.parts),
                "source_asset_sha256": handoff.source_asset_sha256,
                "target_prim_path": handoff.target_prim_path,
                "topology_digest": handoff.topology_digest,
                "fragment_labels_sha256": handoff.fragment_labels_sha256,
                "face_labels_sha256": handoff.face_labels_sha256,
                "routing": (
                    handoff.routing.model_dump(mode="json")
                    if handoff.routing is not None
                    else None
                ),
            },
        ),
        artifacts,
    )
