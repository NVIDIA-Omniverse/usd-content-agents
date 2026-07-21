# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-backed adjudication for Stage 2 articulation evidence conflicts."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, NamedTuple

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from world_understanding.utils.llm_parsing import (
    iter_json_dicts_from_llm_response,
    iter_json_dicts_in_text_order,
)

from joint_agent.functions.articulation_candidates import (
    READY_FOR_RIGGER_INPUT_STATUS,
    REVIEW_REQUIRED_STATUS,
    Stage2EvidenceItem,
)

ADJUDICATION_SCHEMA_VERSION: Literal["joint-agent-articulation-adjudication-v0"] = (
    "joint-agent-articulation-adjudication-v0"
)
ADJUDICATION_ARTIFACT_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-adjudication-artifact-v1"
] = "joint-agent-articulation-adjudication-artifact-v1"
TOPOLOGY_RECONCILIATION_SCHEMA_VERSION: Literal[
    "joint-agent-articulation-topology-reconciliation-v0"
] = "joint-agent-articulation-topology-reconciliation-v0"
_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_TOPOLOGY_FIELDS = (
    "role",
    "instance_id",
    "is_articulation_candidate",
    "joint_type_hint",
    "axis_hint",
    "parent_hint",
    "child_hint",
    "confidence",
    "rigger_evidence",
    "consistency",
)
_TOPOLOGY_AXIS_VALUES = frozenset({"x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z"})
_TOPOLOGY_MOVING_JOINT_TYPES = frozenset({"revolute", "prismatic", "spherical"})
_TOPOLOGY_CORRECTIVE_RETRY_TOKEN_CAP = 16_384
_TOPOLOGY_CORRECTIVE_RETRY_SUFFIX = """

CORRECTIVE RETRY: The previous output was rejected. Regenerate the entire answer
from scratch as exactly one complete JSON object matching the required schema.
Do not output analysis, prose, or Markdown. Include every source prediction
exactly once in member_prims, use only supplied paths and enum values, keep each
rationale to one short sentence, and close every JSON delimiter.
"""
_ADJUDICATION_SYSTEM_PROMPT = (
    "You are a rigorous evidence adjudicator for 3D articulation candidate "
    "reports. Return JSON only."
)
_LOGGER = logging.getLogger(__name__)


class ArticulationConflictAdjudicationRequest(NamedTuple):
    """Prompt and optional images for one model adjudication request."""

    prompt: str
    images: list[str]


class ArticulationConflictAdjudication(BaseModel):
    """One model decision for a Stage 2 conflict-bearing candidate."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    decision: Literal["accept_candidate", "keep_review"]
    confidence: Literal["high", "medium", "low"] = "low"
    resolved_reason_codes: list[Literal["compound_edge_conflict"]] = Field(
        default_factory=list
    )
    motion_type: str | None = None
    fixed_parent_prim: str | None = None
    axis_hint: str | None = None
    rationale: str


class ArticulationAdjudicationDocument(BaseModel):
    """Structured LLM response for articulation conflict adjudication."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["joint-agent-articulation-adjudication-v0"]
    adjudications: list[ArticulationConflictAdjudication] = Field(default_factory=list)


class ArticulationTopologyCompoundEdge(BaseModel):
    """One exact raw compound-edge claim accounted for by reconciliation."""

    model_config = ConfigDict(extra="forbid")

    body0: str = Field(min_length=1)
    body1: str = Field(min_length=1)
    joint_type_hint: Literal["revolute", "prismatic", "spherical"]
    axis_hint: Literal["x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z", "unknown"]


class ArticulationTopologyLink(BaseModel):
    """One complete fixed or moving physical link selected asset-wide."""

    model_config = ConfigDict(extra="forbid")

    link_id: str = Field(min_length=1)
    kind: Literal["fixed", "moving"]
    role: Literal[
        "body",
        "drawer",
        "door",
        "lid",
        "wheel",
        "caster_frame",
        "knob",
        "unknown",
    ]
    member_prims: list[str] = Field(min_length=1)
    anchor_prim: str | None = None
    body0: str | None = None
    body1: str | None = None
    joint_type_hint: Literal["revolute", "prismatic", "spherical"] | None = None
    axis_hint: Literal["x", "+x", "-x", "y", "+y", "-y", "z", "+z", "-z"] | None = None
    superseded_compound_edges: list[ArticulationTopologyCompoundEdge]
    confidence: Literal["high", "medium", "low"]
    rationale: str = Field(min_length=1)


class ArticulationTopologyReconciliationDocument(BaseModel):
    """Strict asset-level physical-link partition and joint topology."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["joint-agent-articulation-topology-reconciliation-v0"]
    links: list[ArticulationTopologyLink] = Field(min_length=1)


class ArticulationAdjudicationArtifact(BaseModel):
    """Stable shared artifact envelope for both adjudication paths."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["joint-agent-articulation-adjudication-artifact-v1"]
    topology_reconciliation: ArticulationTopologyReconciliationDocument | None = None
    adjudications: list[ArticulationConflictAdjudication] = Field(default_factory=list)


class ArticulationTopologyReconciliationRequest(NamedTuple):
    """One asset-wide reconciliation prompt and its bounded source images."""

    prompt: str
    images: list[str]


def build_articulation_conflict_adjudication_prompt(
    candidate_document: Mapping[str, Any],
    source_predictions: Iterable[Mapping[str, Any]],
    *,
    dataset_entries: Iterable[Mapping[str, Any]] | None = None,
    image_base_dir: str | Path | None = None,
    max_images: int = 16,
) -> str:
    """Build the text-only adjudication prompt for conflict-bearing candidates."""
    return build_articulation_conflict_adjudication_request(
        candidate_document,
        source_predictions,
        dataset_entries=dataset_entries,
        image_base_dir=image_base_dir,
        max_images=max_images,
    ).prompt


def build_articulation_conflict_adjudication_request(
    candidate_document: Mapping[str, Any],
    source_predictions: Iterable[Mapping[str, Any]],
    *,
    dataset_entries: Iterable[Mapping[str, Any]] | None = None,
    image_base_dir: str | Path | None = None,
    max_images: int = 16,
) -> ArticulationConflictAdjudicationRequest:
    """Build a model adjudication request with optional source media."""
    prediction_index = {
        str(row.get("id", "")): row for row in source_predictions if row.get("id")
    }
    dataset_index = {
        str(row.get("id", "")): row for row in dataset_entries or () if row.get("id")
    }
    payload: list[dict[str, Any]] = []
    image_records: list[dict[str, Any]] = []
    image_paths: list[str] = []
    image_index_by_path: dict[str, int] = {}
    for candidate in candidate_document.get("candidates", []):
        if not _has_compound_edge_conflict(candidate):
            continue
        source_prediction_ids = candidate.get("source_prediction_ids", [])
        candidate_images = _candidate_image_records(
            candidate,
            dataset_index=dataset_index,
            image_base_dir=image_base_dir,
            max_images=max_images - len(image_paths),
            start_index=len(image_paths) + 1,
            image_index_by_path=image_index_by_path,
        )
        image_records.extend(candidate_images.records)
        image_paths.extend(candidate_images.paths)
        candidate_payload = {
            "candidate": _candidate_adjudication_view(candidate),
            "source_predictions": [
                prediction_index[prediction_id]
                for prediction_id in source_prediction_ids
                if prediction_id in prediction_index
            ],
        }
        if candidate_images.records:
            candidate_payload["source_images"] = candidate_images.records
        payload.append(candidate_payload)

    attached_image_records: list[dict[str, Any]] = []
    seen_attached_image_indexes: set[int] = set()
    for image_record in image_records:
        image_index = image_record["image_index"]
        if image_index in seen_attached_image_indexes:
            continue
        seen_attached_image_indexes.add(image_index)
        attached_image_records.append(image_record)

    attached_images_block = (
        "Attached image order:\n"
        f"{json.dumps(attached_image_records, indent=2, ensure_ascii=False)}"
        if attached_image_records
        else "No source images are attached."
    )
    image_rule = (
        "- If source_images are supplied, use those images only as additional "
        "evidence for the listed candidate/source prims.\n"
        if attached_image_records
        else ""
    )
    prompt = (
        "Adjudicate Stage 2 articulation candidates that are blocked only by "
        "conflicting explicit Stage 1 compound-edge evidence.\n\n"
        "Rules:\n"
        "- Use only the supplied candidate fields and source predictions.\n"
        "- Do not infer from asset-specific priors, visual guesswork, path-name "
        "shortcuts, or likely object behavior.\n"
        "- Return accept_candidate only when the existing candidate motion_type, "
        "fixed_parent_prim, and axis_hint are explicitly better supported than "
        "the conflicting compound edge evidence.\n"
        "- If the conflict cannot be resolved from the supplied evidence, return "
        "keep_review.\n"
        f"{image_rule}"
        "- Do not invent new axes, parents, limits, pivots, or body endpoints.\n\n"
        "Return JSON only in this shape:\n"
        "{\n"
        f'  "schema_version": "{ADJUDICATION_SCHEMA_VERSION}",\n'
        '  "adjudications": [\n'
        "    {\n"
        '      "candidate_id": "candidate_0001",\n'
        '      "decision": "accept_candidate",\n'
        '      "confidence": "high",\n'
        '      "resolved_reason_codes": ["compound_edge_conflict"],\n'
        '      "motion_type": "revolute",\n'
        '      "fixed_parent_prim": "/path/to/fixed/body",\n'
        '      "axis_hint": "+y",\n'
        '      "rationale": "Specific supplied evidence that resolves the conflict."\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "Candidates to adjudicate:\n"
        f"{json.dumps(payload, indent=2, ensure_ascii=False)}\n\n"
        f"{attached_images_block}"
    )
    return ArticulationConflictAdjudicationRequest(prompt=prompt, images=image_paths)


def build_articulation_topology_reconciliation_request(
    candidate_document: Mapping[str, Any],
    source_predictions: Iterable[Mapping[str, Any]],
    *,
    source_metadata: Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
    dataset_entries: Iterable[Mapping[str, Any]] | None = None,
    image_base_dir: str | Path | None = None,
    max_images: int = 64,
    require_images: bool = True,
    output_key: str = "classification",
) -> ArticulationTopologyReconciliationRequest:
    """Build one asset-wide, exact-vocabulary topology reconciliation request."""
    if max_images < 0:
        raise ValueError("max_images must be non-negative")

    prediction_rows = [deepcopy(dict(row)) for row in source_predictions]
    source_ids = _source_prediction_ids(prediction_rows)
    dataset_rows = [dict(row) for row in dataset_entries or ()]
    dataset_index = {
        str(row.get("id", "")): row for row in dataset_rows if row.get("id")
    }
    metadata_index = _normalize_topology_source_metadata(
        source_metadata,
        dataset_entries=dataset_rows,
    )
    source_vocabulary = sorted(_topology_source_vocabulary(source_ids, metadata_index))
    image_records = _topology_image_records(
        source_ids,
        dataset_index=dataset_index,
        image_base_dir=image_base_dir,
        max_images=max_images,
        require_complete=require_images,
    )
    attached_image_records = _dedupe_image_records(image_records.records)
    prompt_predictions = [
        _topology_prompt_prediction(row, output_key=output_key)
        for row in prediction_rows
    ]
    prompt_metadata = {
        prim_path: _topology_prompt_metadata(metadata_index.get(prim_path, {}))
        for prim_path in source_ids
    }
    prompt = (
        "Reconcile the complete physical-link membership and simple-joint topology "
        "for this one asset in a single response.\n\n"
        "Rules:\n"
        "- Partition every source prediction prim exactly once. Do not omit, repeat, "
        "or invent a member.\n"
        "- Use visual and supplied semantic evidence, not path-name substrings or "
        "asset-specific priors.\n"
        "- Emit one fixed link or one moving link for each physical rigid link.\n"
        "- Fixed links must use role body. Moving links must use a supported moving "
        "role and may never use body or unknown. If images do not resolve an unknown "
        "role, return low confidence so the reconciliation fails closed.\n"
        "- Every moving link has exactly one anchor_prim chosen from its members.\n"
        "- For a flat multi-member moving link, use the member whose final path "
        "component is shortest as anchor_prim and body1; break equal-length "
        "ties lexicographically by exact path. This is only a deterministic "
        "representative choice after membership is resolved, not semantic "
        "path-name inference.\n"
        "- body1 must equal anchor_prim. body0 and body1 must be exact paths from "
        "the supplied endpoint vocabulary.\n"
        "- Emit at most one fixed link. A multi-member fixed link must contain "
        "direct siblings beneath one non-root supplied Xform, and no fixed-member "
        "namespace may overlap another link member by ancestry in either "
        "direction.\n"
        "- Use an anchor from another reconciled moving link as body0 for a nested "
        "joint. A root moving link must use a fixed member or the fixed members' "
        "shared direct-parent Xform as body0.\n"
        "- Preserve every supplied compound edge that matches the selected topology.\n"
        "- A raw compound edge whose axis is unknown is a partial constraint: keep "
        "its body0, body1, and joint type exactly and resolve a concrete axis. Do "
        "not list partial edges in superseded_compound_edges.\n"
        "- A fully resolved raw edge may target a non-anchor member of the same "
        "validated flat aggregate. Keep the deterministic aggregate anchor and "
        "list that exact raw member edge in superseded_compound_edges; this only "
        "re-anchors identity and does not reject the physical topology.\n"
        "- A selected topology may otherwise differ from one unambiguous raw edge "
        "only when confidence is high and superseded_compound_edges lists every "
        "rejected raw body0/body1/type/axis tuple exactly. Never list a matching or "
        "invented raw edge. Explain every supersession in rationale.\n"
        "- Mutually conflicting raw edges remain unsafe and must not be resolved here.\n"
        "- Moving-link members must share role, joint type, and stage-space axis.\n"
        "- Fixed links have null anchor/body/joint/axis fields.\n"
        "- Do not invent limits, pivots, geometry, endpoints, or source prims.\n\n"
        "Return JSON only in this exact shape:\n"
        "{\n"
        f'  "schema_version": "{TOPOLOGY_RECONCILIATION_SCHEMA_VERSION}",\n'
        '  "links": [\n'
        "    {\n"
        '      "link_id": "stable unique link identifier",\n'
        '      "kind": "fixed|moving",\n'
        '      "role": "body|drawer|door|lid|wheel|caster_frame|knob|unknown",\n'
        '      "member_prims": ["/exact/source/prim"],\n'
        '      "anchor_prim": null,\n'
        '      "body0": null,\n'
        '      "body1": null,\n'
        '      "joint_type_hint": null,\n'
        '      "axis_hint": null,\n'
        '      "superseded_compound_edges": [],\n'
        '      "confidence": "high|medium|low",\n'
        '      "rationale": "specific supplied evidence"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Exact endpoint vocabulary:\n"
        f"{json.dumps(source_vocabulary, indent=2, ensure_ascii=False)}\n\n"
        "Authoritative source structure by prediction prim:\n"
        f"{json.dumps(prompt_metadata, indent=2, ensure_ascii=False)}\n\n"
        "Current Stage 2 candidate document:\n"
        f"{json.dumps(dict(candidate_document), indent=2, ensure_ascii=False)}\n\n"
        "Source predictions:\n"
        f"{json.dumps(prompt_predictions, indent=2, ensure_ascii=False)}\n\n"
        "Attached source-image order:\n"
        f"{json.dumps(attached_image_records, indent=2, ensure_ascii=False)}"
    )
    return ArticulationTopologyReconciliationRequest(
        prompt=prompt,
        images=image_records.paths,
    )


def parse_articulation_topology_reconciliation_response(
    response_text: str,
) -> ArticulationTopologyReconciliationDocument:
    """Parse the last topology-shaped JSON object, failing closed if it is invalid."""
    last_candidate: dict[str, Any] | None = None
    for candidate in iter_json_dicts_in_text_order(response_text):
        if "links" in candidate:
            last_candidate = candidate
    if last_candidate is None:
        raise ValueError("Model response did not contain topology reconciliation JSON")
    try:
        return ArticulationTopologyReconciliationDocument.model_validate(last_candidate)
    except ValidationError as exc:
        raise ValueError(f"Invalid topology reconciliation response: {exc}") from exc


def reconcile_articulation_topology_with_model(
    *,
    model: Any,
    candidate_document: Mapping[str, Any],
    source_predictions: Iterable[Mapping[str, Any]],
    source_metadata: Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
    dataset_entries: Iterable[Mapping[str, Any]] | None = None,
    image_base_dir: str | Path | None = None,
    max_images: int = 64,
    require_images: bool = True,
    use_images: bool = True,
    min_confidence: Literal["high", "medium", "low"] = "high",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    output_key: str = "classification",
    diagnostics: dict[str, str] | None = None,
) -> ArticulationTopologyReconciliationDocument | None:
    """Return a fully validated topology from at most two asset-wide attempts.

    A valid first response uses exactly one model call. A response that cannot be
    parsed or fails topology validation gets one corrective compact-JSON retry
    with the same evidence and a bounded larger output budget. Request-construction
    and model-invocation failures do not retry.

    ``diagnostics`` is an optional, response-free status sink for callers that need
    to distinguish request construction, model invocation, response parsing, and
    topology validation failures. It never contains prompt or model response text.
    """

    def record_diagnostics(
        outcome: str,
        *,
        failure_stage: str | None = None,
        error_type: str | None = None,
    ) -> None:
        if diagnostics is None:
            return
        diagnostics.clear()
        diagnostics["outcome"] = outcome
        if failure_stage is not None:
            diagnostics["failure_stage"] = failure_stage
        if error_type is not None:
            diagnostics["error_type"] = error_type

    if min_confidence != "high":
        _LOGGER.warning(
            "Topology reconciliation requires a high-confidence gate; keeping "
            "candidates review-required"
        )
        record_diagnostics(
            "failed",
            failure_stage="request",
            error_type="ValueError",
        )
        return None
    prediction_rows = [deepcopy(dict(row)) for row in source_predictions]
    dataset_rows = [dict(row) for row in dataset_entries or ()]
    metadata_index = _normalize_topology_source_metadata(
        source_metadata,
        dataset_entries=dataset_rows,
    )
    try:
        request = build_articulation_topology_reconciliation_request(
            candidate_document,
            prediction_rows,
            source_metadata=metadata_index,
            dataset_entries=dataset_rows,
            image_base_dir=image_base_dir,
            max_images=max_images if use_images else 0,
            require_images=require_images and use_images,
            output_key=output_key,
        )
    except (TypeError, ValueError) as exc:
        _LOGGER.warning(
            "Topology reconciliation request is invalid; keeping candidates "
            "review-required (%s)",
            type(exc).__name__,
        )
        record_diagnostics(
            "failed",
            failure_stage="request",
            error_type=type(exc).__name__,
        )
        return None
    if require_images and not request.images:
        _LOGGER.warning(
            "Topology reconciliation requires source images; keeping candidates "
            "review-required"
        )
        record_diagnostics(
            "failed",
            failure_stage="request",
            error_type="ValueError",
        )
        return None
    max_attempts = 2
    for attempt in range(1, max_attempts + 1):
        attempt_prompt = request.prompt
        attempt_max_tokens = max_tokens
        if attempt > 1:
            attempt_prompt += _TOPOLOGY_CORRECTIVE_RETRY_SUFFIX
            attempt_max_tokens = max(
                max_tokens,
                min(max_tokens * 2, _TOPOLOGY_CORRECTIVE_RETRY_TOKEN_CAP),
            )
        try:
            response_text = _invoke_model(
                model,
                prompt=attempt_prompt,
                images=request.images,
                use_images=use_images,
                temperature=temperature,
                max_tokens=attempt_max_tokens,
            )
        except Exception as exc:
            _LOGGER.warning(
                "Topology reconciliation failed closed during model invocation; "
                "keeping candidates review-required (%s)",
                type(exc).__name__,
            )
            record_diagnostics(
                "failed",
                failure_stage="invocation",
                error_type=type(exc).__name__,
            )
            return None
        try:
            reconciliation = parse_articulation_topology_reconciliation_response(
                response_text
            )
        except Exception as exc:
            if attempt < max_attempts:
                _LOGGER.warning(
                    "Topology reconciliation response could not be parsed; "
                    "retrying once with a corrective compact-JSON request (%s)",
                    type(exc).__name__,
                )
                continue
            _LOGGER.warning(
                "Topology reconciliation response could not be parsed after the "
                "bounded retry; keeping candidates review-required (%s)",
                type(exc).__name__,
            )
            record_diagnostics(
                "failed",
                failure_stage="parse",
                error_type=type(exc).__name__,
            )
            return None
        try:
            _validate_topology_reconciliation(
                reconciliation,
                source_predictions=prediction_rows,
                source_metadata=metadata_index,
                min_confidence=min_confidence,
                output_key=output_key,
            )
            reconciliation = _canonicalize_flat_topology_anchors(
                reconciliation,
                source_predictions=prediction_rows,
                source_metadata=metadata_index,
                output_key=output_key,
            )
            _validate_topology_reconciliation(
                reconciliation,
                source_predictions=prediction_rows,
                source_metadata=metadata_index,
                min_confidence=min_confidence,
                output_key=output_key,
            )
            if not any(link.kind == "moving" for link in reconciliation.links):
                raise ValueError(
                    "topology reconciliation must retain at least one moving link"
                )
        except Exception as exc:
            if attempt < max_attempts:
                _LOGGER.warning(
                    "Topology reconciliation response failed validation; retrying "
                    "once with a corrective compact-JSON request (%s)",
                    type(exc).__name__,
                )
                continue
            _LOGGER.warning(
                "Topology reconciliation response failed validation after the "
                "bounded retry; keeping candidates review-required (%s)",
                type(exc).__name__,
            )
            record_diagnostics(
                "failed",
                failure_stage="validation",
                error_type=type(exc).__name__,
            )
            return None
        record_diagnostics("validated")
        return reconciliation
    raise AssertionError("bounded topology reconciliation loop did not terminate")


def apply_articulation_topology_reconciliation(
    source_predictions: Iterable[Mapping[str, Any]],
    reconciliation: ArticulationTopologyReconciliationDocument | Mapping[str, Any],
    *,
    output_key: str = "classification",
) -> list[dict[str, Any]]:
    """Overlay a validated topology while retaining original values in history."""
    document = (
        reconciliation
        if isinstance(reconciliation, ArticulationTopologyReconciliationDocument)
        else ArticulationTopologyReconciliationDocument.model_validate(reconciliation)
    )
    rows = [deepcopy(dict(row)) for row in source_predictions]
    link_by_member = {
        member: link for link in document.links for member in link.member_prims
    }
    if len(link_by_member) != sum(len(link.member_prims) for link in document.links):
        raise ValueError("topology reconciliation repeats source membership")
    if len({link.link_id for link in document.links}) != len(document.links):
        raise ValueError("topology reconciliation repeats a link identifier")
    if set(link_by_member) != set(_source_prediction_ids(rows)):
        raise ValueError(
            "topology reconciliation must cover each source prediction exactly once"
        )
    limits_by_link_id = _reconciled_link_limits(
        rows,
        document,
        output_key=output_key,
    )
    matching_compound_edges_by_link_id = _preserved_matching_compound_edges(
        rows,
        document,
        output_key=output_key,
    )
    topology_document_sha256 = _topology_document_sha256(document)

    result: list[dict[str, Any]] = []
    for row in rows:
        prim_path = str(row.get("id", ""))
        raw_payload = row.get(output_key)
        if not isinstance(raw_payload, Mapping):
            raise ValueError(f"prediction {prim_path!r} has no {output_key!r} object")
        payload = deepcopy(dict(raw_payload))
        original_payload = deepcopy(payload)
        link = link_by_member[prim_path]
        original_values = {
            field: deepcopy(payload.get(field))
            for field in _TOPOLOGY_FIELDS
            if field in payload
        }
        previous_provenance = payload.get("provenance")
        previous_field_sources = (
            previous_provenance.get("field_sources")
            if isinstance(previous_provenance, Mapping)
            else None
        )
        original_field_sources = (
            deepcopy(dict(previous_field_sources))
            if isinstance(previous_field_sources, Mapping)
            else {}
        )

        payload["role"] = link.role
        payload["instance_id"] = link.link_id
        payload["confidence"] = link.confidence
        if link.kind == "fixed":
            payload["is_articulation_candidate"] = False
            payload["joint_type_hint"] = "fixed"
            payload["axis_hint"] = "unknown"
            payload["parent_hint"] = "unknown"
            payload["child_hint"] = "unknown"
            payload.pop("rigger_evidence", None)
        else:
            is_anchor = prim_path == link.anchor_prim
            payload["is_articulation_candidate"] = is_anchor
            payload["joint_type_hint"] = link.joint_type_hint
            payload["axis_hint"] = link.axis_hint
            payload["parent_hint"] = link.body0
            payload["child_hint"] = link.body1
            if is_anchor:
                limits = deepcopy(limits_by_link_id.get(link.link_id))
                rigger_evidence: dict[str, Any] = {
                    "body0": {
                        "value": link.body0,
                        "prim_paths": [link.body0],
                        "confidence": link.confidence,
                        "rationale": link.rationale,
                        "source": "llm_adjudicated",
                    },
                    "body1": {
                        "value": link.body1,
                        "prim_paths": [link.body1],
                        "confidence": link.confidence,
                        "rationale": link.rationale,
                        "source": "llm_adjudicated",
                    },
                    "motion_axis": {
                        "value": link.axis_hint,
                        "prim_paths": [link.body1],
                        "confidence": link.confidence,
                        "rationale": link.rationale,
                        "source": "llm_adjudicated",
                    },
                    "compound_edges": deepcopy(
                        matching_compound_edges_by_link_id.get(link.link_id, [])
                    ),
                }
                if limits is not None:
                    rigger_evidence["limits"] = limits
                payload["rigger_evidence"] = rigger_evidence
            else:
                payload.pop("rigger_evidence", None)

        _clear_reconciled_consistency_conflicts(payload)
        provenance = payload.get("provenance")
        if not isinstance(provenance, dict):
            provenance = {}
            payload["provenance"] = provenance
        field_sources = provenance.get("field_sources")
        if not isinstance(field_sources, dict):
            field_sources = {}
            provenance["field_sources"] = field_sources
        for field in (
            "role",
            "instance_id",
            "is_articulation_candidate",
            "joint_type_hint",
            "axis_hint",
            "parent_hint",
            "child_hint",
            "confidence",
        ):
            field_sources[field] = "llm_adjudicated"
        history = provenance.get("topology_reconciliation_history")
        if not isinstance(history, list):
            history = []
            provenance["topology_reconciliation_history"] = history
        history.append(
            {
                "schema_version": TOPOLOGY_RECONCILIATION_SCHEMA_VERSION,
                "source": "llm_adjudicated",
                "topology_document_sha256": topology_document_sha256,
                "link_id": link.link_id,
                "confidence": link.confidence,
                "rationale": link.rationale,
                "superseded_compound_edges": [
                    edge.model_dump(mode="json")
                    for edge in link.superseded_compound_edges
                ],
                "reconciled_link": link.model_dump(mode="json"),
                "original_payload": original_payload,
                "original_provenance_present": isinstance(previous_provenance, Mapping),
                "original_field_sources": original_field_sources,
                "original_values": original_values,
            }
        )
        row[output_key] = payload
        result.append(row)
    return result


def recover_articulation_topology_reconciliation_from_history(
    source_predictions: Iterable[Mapping[str, Any]],
    *,
    output_key: str = "classification",
    source_metadata: Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
) -> ArticulationTopologyReconciliationDocument | None:
    """Recover one exact current topology receipt or fail closed on stale rows."""
    try:
        rows = [dict(row) for row in source_predictions]
        source_ids = _source_prediction_ids(rows)
        links_by_id: dict[str, ArticulationTopologyLink] = {}
        document_digests: set[str] = set()
        for row in rows:
            prim_path = str(row.get("id", ""))
            payload = row.get(output_key)
            provenance = (
                payload.get("provenance") if isinstance(payload, Mapping) else None
            )
            history = (
                provenance.get("topology_reconciliation_history")
                if isinstance(provenance, Mapping)
                else None
            )
            if (
                not isinstance(payload, Mapping)
                or not isinstance(history, list)
                or not history
            ):
                return None
            latest = history[-1]
            if (
                not isinstance(latest, Mapping)
                or latest.get("schema_version")
                != TOPOLOGY_RECONCILIATION_SCHEMA_VERSION
                or latest.get("source") != "llm_adjudicated"
                or not isinstance(latest.get("reconciled_link"), Mapping)
                or not isinstance(latest.get("topology_document_sha256"), str)
            ):
                return None
            document_digests.add(str(latest["topology_document_sha256"]))
            link = ArticulationTopologyLink.model_validate(latest["reconciled_link"])
            if (
                latest.get("link_id") != link.link_id
                or prim_path not in link.member_prims
                or not _payload_matches_reconciled_link(prim_path, payload, link)
            ):
                return None
            previous = links_by_id.get(link.link_id)
            if previous is not None and previous != link:
                return None
            links_by_id.setdefault(link.link_id, link)

        links = list(links_by_id.values())
        if sum(len(link.member_prims) for link in links) != len(source_ids):
            return None
        if {member for link in links for member in link.member_prims} != set(
            source_ids
        ):
            return None
        document = ArticulationTopologyReconciliationDocument(
            schema_version=TOPOLOGY_RECONCILIATION_SCHEMA_VERSION,
            links=links,
        )
        _validate_recovered_topology_graph(document)
        if document_digests != {_topology_document_sha256(document)}:
            return None
        original_rows = _restore_topology_reconciliation_once(
            rows,
            output_key=output_key,
        )
        reproduced_rows = apply_articulation_topology_reconciliation(
            original_rows,
            document,
            output_key=output_key,
        )
        if any(
            current.get(output_key) != reproduced.get(output_key)
            for current, reproduced in zip(rows, reproduced_rows, strict=True)
        ):
            return None
        if source_metadata is not None:
            metadata_index = _normalize_topology_source_metadata(
                source_metadata,
                dataset_entries=(),
            )
            _validate_topology_reconciliation(
                document,
                source_predictions=original_rows,
                source_metadata=metadata_index,
                min_confidence="high",
                output_key=output_key,
            )
        return document
    except (KeyError, TypeError, ValueError, ValidationError):
        return None


def restore_articulation_topology_reconciliation_originals(
    source_predictions: Iterable[Mapping[str, Any]],
    *,
    output_key: str = "classification",
    source_metadata: Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
) -> list[dict[str, Any]]:
    """Restore every accepted topology overlay to its exact pre-overlay rows."""
    rows = [deepcopy(dict(row)) for row in source_predictions]
    while True:
        history_presence = [_row_has_topology_history(row, output_key) for row in rows]
        if not any(history_presence):
            return rows
        if not all(history_presence):
            raise ValueError("topology reconciliation history is incomplete")
        document = recover_articulation_topology_reconciliation_from_history(
            rows,
            output_key=output_key,
            source_metadata=source_metadata,
        )
        if document is None:
            raise ValueError("topology reconciliation history is invalid")
        rows = _restore_topology_reconciliation_once(rows, output_key=output_key)


def _restore_topology_reconciliation_once(
    source_predictions: Sequence[Mapping[str, Any]],
    *,
    output_key: str,
) -> list[dict[str, Any]]:
    restored: list[dict[str, Any]] = []
    for source_row in source_predictions:
        row = deepcopy(dict(source_row))
        payload = row.get(output_key)
        if not isinstance(payload, dict):
            raise ValueError("topology prediction payload is missing")
        provenance = payload.get("provenance")
        history = (
            provenance.get("topology_reconciliation_history")
            if isinstance(provenance, dict)
            else None
        )
        if not isinstance(history, list) or not history:
            raise ValueError("topology reconciliation history is incomplete")
        latest = history[-1]
        if not isinstance(latest, Mapping):
            raise ValueError("topology reconciliation history is malformed")
        original_payload = latest.get("original_payload")
        if not isinstance(original_payload, Mapping):
            raise ValueError("topology reconciliation originals are malformed")
        row[output_key] = deepcopy(dict(original_payload))
        restored.append(row)
    return restored


def _row_has_topology_history(row: Mapping[str, Any], output_key: str) -> bool:
    payload = row.get(output_key)
    provenance = payload.get("provenance") if isinstance(payload, Mapping) else None
    history = (
        provenance.get("topology_reconciliation_history")
        if isinstance(provenance, Mapping)
        else None
    )
    return isinstance(history, list) and bool(history)


def _topology_document_sha256(
    document: ArticulationTopologyReconciliationDocument,
) -> str:
    payload = {
        "schema_version": document.schema_version,
        "links": [
            link.model_dump(mode="json")
            for link in sorted(document.links, key=lambda item: item.link_id)
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_recovered_topology_graph(
    document: ArticulationTopologyReconciliationDocument,
) -> None:
    member_to_link: dict[str, ArticulationTopologyLink] = {}
    moving_body_to_link: dict[str, ArticulationTopologyLink] = {}
    link_ids: set[str] = set()
    for link in document.links:
        if (
            not link.link_id.strip()
            or link.link_id != link.link_id.strip()
            or link.link_id in link_ids
        ):
            raise ValueError("recovered topology link identifiers are invalid")
        link_ids.add(link.link_id)
        if link.confidence != "high":
            raise ValueError("recovered topology is below the confidence gate")
        _validate_topology_link_shape(link, structure_mode="hierarchy")
        for member in link.member_prims:
            if member in member_to_link:
                raise ValueError("recovered topology repeats source membership")
            member_to_link[member] = link
        if link.kind == "moving":
            assert link.body1 is not None
            if link.body1 in moving_body_to_link:
                raise ValueError("recovered topology repeats a moving body")
            moving_body_to_link[link.body1] = link

    _validate_owned_core_fixed_projection_compatibility(document)

    endpoint_to_link = {**member_to_link, **moving_body_to_link}
    moving_parent_by_link_id: dict[str, str] = {}
    for link in document.links:
        if link.kind != "moving":
            continue
        assert link.body0 is not None
        parent_link = endpoint_to_link.get(link.body0)
        if parent_link is link:
            raise ValueError("recovered topology is self-parented")
        if parent_link is not None and parent_link.kind == "moving":
            if parent_link.anchor_prim != link.body0:
                raise ValueError("recovered nested parent is not its link anchor")
            moving_parent_by_link_id[link.link_id] = parent_link.link_id
    _validate_acyclic_moving_parent_graph(moving_parent_by_link_id)


def _payload_matches_reconciled_link(
    prim_path: str,
    payload: Mapping[str, Any],
    link: ArticulationTopologyLink,
) -> bool:
    if link.kind == "fixed":
        expected: dict[str, Any] = {
            "role": "body",
            "instance_id": link.link_id,
            "is_articulation_candidate": False,
            "joint_type_hint": "fixed",
            "axis_hint": "unknown",
            "parent_hint": "unknown",
            "child_hint": "unknown",
            "confidence": link.confidence,
        }
        expect_rigger_evidence = False
    else:
        is_anchor = prim_path == link.anchor_prim
        expected = {
            "role": link.role,
            "instance_id": link.link_id,
            "is_articulation_candidate": is_anchor,
            "joint_type_hint": link.joint_type_hint,
            "axis_hint": link.axis_hint,
            "parent_hint": link.body0,
            "child_hint": link.body1,
            "confidence": link.confidence,
        }
        expect_rigger_evidence = is_anchor
    if any(
        field not in payload or payload[field] != value
        for field, value in expected.items()
    ):
        return False

    provenance = payload.get("provenance")
    field_sources = (
        provenance.get("field_sources") if isinstance(provenance, Mapping) else None
    )
    if not isinstance(field_sources, Mapping) or any(
        field_sources.get(field) != "llm_adjudicated" for field in expected
    ):
        return False

    evidence = payload.get("rigger_evidence")
    if not expect_rigger_evidence:
        return evidence is None
    if not isinstance(evidence, Mapping):
        return False
    expected_claims = {
        "body0": link.body0,
        "body1": link.body1,
        "motion_axis": link.axis_hint,
    }
    for field, value in expected_claims.items():
        claim = evidence.get(field)
        if (
            not isinstance(claim, Mapping)
            or claim.get("value") != value
            or claim.get("source") != "llm_adjudicated"
        ):
            return False
    raw_edges = evidence.get("compound_edges")
    if not isinstance(raw_edges, list):
        return False
    try:
        return all(
            isinstance(raw_edge, Mapping)
            and _compound_edge_matches_link(
                _normalize_raw_compound_edge(raw_edge),
                link,
            )
            for raw_edge in raw_edges
        )
    except (TypeError, ValueError):
        return False


def _reconciled_link_limits(
    rows: Sequence[Mapping[str, Any]],
    document: ArticulationTopologyReconciliationDocument,
    *,
    output_key: str,
) -> dict[str, dict[str, Any]]:
    """Carry one exact member limit claim and reject link-local disagreement."""
    row_by_id = {str(row.get("id", "")): row for row in rows}
    result: dict[str, dict[str, Any]] = {}
    for link in document.links:
        if link.kind != "moving":
            continue
        unique_claim: dict[str, Any] | None = None
        for member in link.member_prims:
            payload = row_by_id[member].get(output_key)
            evidence = (
                payload.get("rigger_evidence") if isinstance(payload, Mapping) else None
            )
            if not isinstance(evidence, Mapping) or "limits" not in evidence:
                continue
            raw_limits = evidence.get("limits")
            if not isinstance(raw_limits, Mapping):
                raise ValueError(
                    f"topology link {link.link_id!r} has malformed limit evidence"
                )
            claim = deepcopy(dict(raw_limits))
            if unique_claim is None:
                unique_claim = claim
            elif claim != unique_claim:
                raise ValueError(
                    f"topology link {link.link_id!r} has conflicting limit evidence"
                )
        if unique_claim is not None:
            result[link.link_id] = unique_claim
    return result


def adjudicate_articulation_conflicts_with_model(
    *,
    model: Any,
    candidate_document: Mapping[str, Any],
    source_predictions: Iterable[Mapping[str, Any]],
    dataset_entries: Iterable[Mapping[str, Any]] | None = None,
    image_base_dir: str | Path | None = None,
    max_images: int = 16,
    use_images: bool = False,
    require_images: bool = False,
    max_adjudications: int | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> ArticulationAdjudicationDocument:
    """Ask a chat model to adjudicate Stage 2 conflict evidence."""
    if max_adjudications is not None and max_adjudications < 0:
        raise ValueError("max_adjudications must be non-negative")
    if require_images and not use_images:
        raise ValueError("require_images requires use_images")

    source_prediction_rows = list(source_predictions)
    dataset_entry_rows = list(dataset_entries) if dataset_entries is not None else None
    adjudications: list[ArticulationConflictAdjudication] = []
    invocation_count = 0
    for candidate in candidate_document.get("candidates", []):
        if not _has_compound_edge_conflict(candidate):
            continue
        candidate_id = str(candidate.get("candidate_id", ""))
        request = build_articulation_conflict_adjudication_request(
            {
                **dict(candidate_document),
                "candidates": [candidate],
            },
            source_prediction_rows,
            dataset_entries=dataset_entry_rows if use_images else None,
            image_base_dir=image_base_dir if use_images else None,
            max_images=max_images if use_images else 0,
        )
        if require_images and not request.images:
            _LOGGER.warning(
                "Keeping articulation candidate %s review-required because no "
                "source images were available",
                candidate_id or "<unknown>",
            )
            continue
        if max_adjudications is not None and invocation_count >= max_adjudications:
            _LOGGER.warning(
                "Reached articulation adjudication budget of %d candidates; "
                "remaining conflicts stay review-required",
                max_adjudications,
            )
            break
        invocation_count += 1
        try:
            response_text = _invoke_model(
                model,
                prompt=request.prompt,
                images=request.images,
                use_images=use_images,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            _LOGGER.warning(
                "Keeping articulation candidate %s review-required after model "
                "adjudication failed (%s)",
                candidate_id or "<unknown>",
                type(exc).__name__,
            )
            continue
        try:
            response_document = parse_articulation_conflict_adjudication_response(
                response_text
            )
        except ValueError as exc:
            _LOGGER.warning(
                "Skipping invalid articulation adjudication response for %s: %s",
                candidate_id or "<unknown>",
                exc,
            )
            continue
        adjudications.extend(
            adjudication
            for adjudication in response_document.adjudications
            if adjudication.candidate_id == candidate_id
        )
    return ArticulationAdjudicationDocument(
        schema_version=ADJUDICATION_SCHEMA_VERSION,
        adjudications=adjudications,
    )


def parse_articulation_conflict_adjudication_response(
    response_text: str,
) -> ArticulationAdjudicationDocument:
    """Parse a model adjudication response."""
    last_validation_error: ValidationError | None = None
    for candidate in iter_json_dicts_from_llm_response(response_text):
        if "adjudications" not in candidate:
            continue
        try:
            return ArticulationAdjudicationDocument.model_validate(candidate)
        except ValidationError as exc:
            last_validation_error = exc
    if last_validation_error is not None:
        raise ValueError(f"Invalid adjudication response: {last_validation_error}")
    raise ValueError("Model adjudication response did not contain adjudications JSON")


def apply_articulation_conflict_adjudications(
    candidate_document: Mapping[str, Any],
    adjudications: Sequence[ArticulationConflictAdjudication | Mapping[str, Any]],
    *,
    min_confidence: Literal["high", "medium", "low"] = "high",
) -> dict[str, Any]:
    """Apply structured model adjudications to a Stage 2 candidate document.

    The function only clears ``compound_edge_conflict`` when the model explicitly
    accepts the existing candidate fields. It does not infer or modify Stage 2
    axis, parent, limit, or body fields.
    """
    updated_document = deepcopy(dict(candidate_document))
    updated_candidates = [
        deepcopy(dict(candidate)) for candidate in candidate_document["candidates"]
    ]
    adjudication_by_id: dict[str, ArticulationConflictAdjudication] = {}
    duplicate_candidate_ids: set[str] = set()
    for raw_adjudication in adjudications:
        adjudication = _coerce_adjudication(raw_adjudication)
        if adjudication.candidate_id in adjudication_by_id:
            duplicate_candidate_ids.add(adjudication.candidate_id)
            del adjudication_by_id[adjudication.candidate_id]
            continue
        if adjudication.candidate_id not in duplicate_candidate_ids:
            adjudication_by_id[adjudication.candidate_id] = adjudication

    for candidate in updated_candidates:
        candidate_adjudication = adjudication_by_id.get(
            str(candidate.get("candidate_id", ""))
        )
        if candidate_adjudication is None or not _accepts_existing_candidate(
            candidate,
            candidate_adjudication,
            min_confidence=min_confidence,
        ):
            continue
        _clear_compound_edge_conflict(candidate, candidate_adjudication)

    updated_document["candidates"] = updated_candidates
    updated_summary = _candidate_summary(updated_candidates)
    original_summary = candidate_document.get("summary", {})
    if isinstance(original_summary, Mapping):
        updated_summary = {**dict(original_summary), **updated_summary}
        updated_summary["total_predictions"] = original_summary.get(
            "total_predictions",
            updated_summary["total_predictions"],
        )
    updated_document["summary"] = updated_summary
    return updated_document


def _invoke_model(
    model: Any,
    *,
    prompt: str,
    images: Sequence[str],
    use_images: bool,
    temperature: float,
    max_tokens: int,
) -> str:
    if use_images:
        if not hasattr(model, "generate"):
            raise ValueError("VLM adjudication requires a model with generate()")
        try:
            response_text = model.generate(
                prompt=prompt,
                images=list(images) if images else None,
                system_prompt=_ADJUDICATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            error_text = str(exc)
            if (
                "max_tokens" not in error_text
                and "max_completion_tokens" not in error_text
            ):
                raise
            response_text = model.generate(
                prompt=prompt,
                images=list(images) if images else None,
                system_prompt=_ADJUDICATION_SYSTEM_PROMPT,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        return response_text if isinstance(response_text, str) else str(response_text)

    messages = [
        SystemMessage(content=_ADJUDICATION_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]
    try:
        response = model.invoke(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except TypeError:
        try:
            response = model.invoke(
                messages,
                temperature=temperature,
                max_completion_tokens=max_tokens,
            )
        except TypeError as completion_exc:
            try:
                response = model.invoke(messages)
            except TypeError as fallback_exc:
                raise fallback_exc from completion_exc
    except Exception as exc:
        error_text = str(exc)
        if "max_tokens" not in error_text and "max_completion_tokens" not in error_text:
            raise
        response = model.invoke(
            messages,
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    content = getattr(response, "content", response)
    return content if isinstance(content, str) else str(content)


def _candidate_adjudication_view(candidate: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "candidate_id",
        "motion_type",
        "moving_part_prims",
        "fixed_parent_prim",
        "parent_resolution_source",
        "joint_type_hint",
        "axis_hint",
        "motion_axis_world",
        "parent_hint",
        "component_name",
        "component_type",
        "role",
        "source_prediction_ids",
        "axis_evidence",
        "connectivity_evidence",
        "unresolved_reason_codes",
        "unresolved_questions",
        "evidence",
    )
    return {key: candidate.get(key) for key in keys if key in candidate}


class _CandidateImageRecords(NamedTuple):
    records: list[dict[str, Any]]
    paths: list[str]


class _ResolvedMediaPath(NamedTuple):
    resolved_path: str
    prompt_path: str


def _candidate_image_records(
    candidate: Mapping[str, Any],
    *,
    dataset_index: Mapping[str, Mapping[str, Any]],
    image_base_dir: str | Path | None,
    max_images: int,
    start_index: int,
    image_index_by_path: dict[str, int],
) -> _CandidateImageRecords:
    base_dir = Path(image_base_dir) if image_base_dir else None
    records: list[dict[str, Any]] = []
    paths: list[str] = []
    for source_prediction_id in candidate.get("source_prediction_ids", []):
        dataset_entry = dataset_index.get(str(source_prediction_id))
        if dataset_entry is None:
            continue
        for image_entry in _iter_dataset_image_entries(dataset_entry):
            media_path = _resolve_media_path(image_entry.get("path"), base_dir)
            if media_path is None:
                continue
            image_index = image_index_by_path.get(media_path.resolved_path)
            if image_index is None:
                if len(paths) >= max_images:
                    continue
                image_index = start_index + len(paths)
                image_index_by_path[media_path.resolved_path] = image_index
                paths.append(media_path.resolved_path)
            records.append(
                {
                    "image_index": image_index,
                    "source_prediction_id": str(source_prediction_id),
                    "path": media_path.prompt_path,
                    "type": image_entry.get("type"),
                    "metadata": image_entry.get("metadata", {}),
                }
            )
    return _CandidateImageRecords(records=records, paths=paths)


def _topology_image_records(
    source_prediction_ids: Sequence[str],
    *,
    dataset_index: Mapping[str, Mapping[str, Any]],
    image_base_dir: str | Path | None,
    max_images: int,
    require_complete: bool,
) -> _CandidateImageRecords:
    """Select one resolved render per prim before adding any extra views."""
    if require_complete and max_images < len(source_prediction_ids):
        raise ValueError(
            "max_images must allow at least one source render per prediction prim"
        )
    base_dir = Path(image_base_dir) if image_base_dir else None
    entries_by_prim: dict[str, list[tuple[Mapping[str, Any], _ResolvedMediaPath]]] = {}
    for prim_path in source_prediction_ids:
        prim_render_entries: list[tuple[Mapping[str, Any], _ResolvedMediaPath]] = []
        reference_entries: list[tuple[Mapping[str, Any], _ResolvedMediaPath]] = []
        dataset_entry = dataset_index.get(prim_path)
        if dataset_entry is not None:
            for image_entry in _iter_dataset_image_entries(dataset_entry):
                media_path = _resolve_media_path(image_entry.get("path"), base_dir)
                if media_path is not None and Path(media_path.resolved_path).is_file():
                    target = (
                        reference_entries
                        if str(image_entry.get("type", "")).strip().lower()
                        == "reference"
                        else prim_render_entries
                    )
                    target.append((image_entry, media_path))
        if require_complete and not prim_render_entries:
            raise ValueError(f"source prediction {prim_path!r} has no resolved render")
        entries_by_prim[prim_path] = [*prim_render_entries, *reference_entries]

    records: list[dict[str, Any]] = []
    paths: list[str] = []
    image_index_by_path: dict[str, int] = {}

    def add(
        prim_path: str,
        entry: Mapping[str, Any],
        media_path: _ResolvedMediaPath,
    ) -> None:
        image_index = image_index_by_path.get(media_path.resolved_path)
        if image_index is None:
            if len(paths) >= max_images:
                return
            paths.append(media_path.resolved_path)
            image_index = len(paths)
            image_index_by_path[media_path.resolved_path] = image_index
        records.append(
            {
                "image_index": image_index,
                "source_prediction_id": prim_path,
                "path": media_path.prompt_path,
                "type": entry.get("type"),
                "metadata": entry.get("metadata", {}),
            }
        )

    for prim_path in source_prediction_ids:
        resolved_entries = entries_by_prim[prim_path]
        if resolved_entries:
            add(prim_path, *resolved_entries[0])
    if require_complete:
        primary_paths = [
            entries_by_prim[prim_path][0][1].resolved_path
            for prim_path in source_prediction_ids
        ]
        if len(set(primary_paths)) != len(primary_paths):
            raise ValueError(
                "each source prediction requires a distinct prim-specific render"
            )
    if require_complete and {
        record["source_prediction_id"] for record in records
    } != set(source_prediction_ids):
        raise ValueError("not every source prediction received an attached render")

    max_extra_views = max(
        (len(entries) - 1 for entries in entries_by_prim.values()),
        default=0,
    )
    for view_index in range(1, max_extra_views + 1):
        for prim_path in source_prediction_ids:
            resolved_entries = entries_by_prim[prim_path]
            if view_index < len(resolved_entries):
                add(prim_path, *resolved_entries[view_index])
    return _CandidateImageRecords(records=records, paths=paths)


def _iter_dataset_image_entries(
    dataset_entry: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    media = dataset_entry.get("media", {})
    if isinstance(media, Mapping):
        raw_images = media.get("images", [])
    else:
        raw_images = []
    if not isinstance(raw_images, list):
        return
    for image_entry in raw_images:
        if isinstance(image_entry, Mapping):
            yield image_entry


def _resolve_media_path(
    raw_path: Any, base_dir: Path | None
) -> _ResolvedMediaPath | None:
    if not isinstance(raw_path, str) or not raw_path:
        return None
    path = Path(raw_path)
    if base_dir is None:
        return None

    resolved_base_dir = base_dir.resolve()
    resolved_path = path if path.is_absolute() else resolved_base_dir / path
    resolved_path = resolved_path.resolve()
    try:
        prompt_path = resolved_path.relative_to(resolved_base_dir).as_posix()
    except ValueError:
        return None
    return _ResolvedMediaPath(
        resolved_path=str(resolved_path),
        prompt_path=prompt_path,
    )


def _source_prediction_ids(
    source_predictions: Iterable[Mapping[str, Any]],
) -> list[str]:
    source_ids: list[str] = []
    seen: set[str] = set()
    for row in source_predictions:
        prim_path = str(row.get("id", "")).strip()
        if not prim_path.startswith("/"):
            raise ValueError("every source prediction must have an absolute id")
        if prim_path in seen:
            raise ValueError(f"duplicate source prediction id: {prim_path}")
        seen.add(prim_path)
        source_ids.append(prim_path)
    if not source_ids:
        raise ValueError("topology reconciliation requires source predictions")
    return source_ids


def _normalize_topology_source_metadata(
    source_metadata: Iterable[Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None,
    *,
    dataset_entries: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    if isinstance(source_metadata, Mapping):
        for prim_path, raw_metadata in source_metadata.items():
            if isinstance(raw_metadata, Mapping):
                row = dict(raw_metadata)
                row.setdefault("id", str(prim_path))
                index[str(prim_path)] = row
    elif source_metadata is not None:
        for raw_metadata in source_metadata:
            if not isinstance(raw_metadata, Mapping):
                continue
            prim_path = str(
                raw_metadata.get("id")
                or raw_metadata.get("prim_path")
                or raw_metadata.get("path")
                or ""
            )
            if prim_path:
                index[prim_path] = dict(raw_metadata)
    for raw_entry in dataset_entries:
        if not isinstance(raw_entry, Mapping):
            continue
        prim_path = str(raw_entry.get("id", ""))
        if prim_path and prim_path not in index:
            index[prim_path] = dict(raw_entry)
    return index


def _source_structure_payload(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    usd_metadata = metadata.get("usd_metadata")
    return usd_metadata if isinstance(usd_metadata, Mapping) else metadata


def _topology_structure_mode(metadata: Mapping[str, Any]) -> str:
    structure = _source_structure_payload(metadata)
    provenance = str(structure.get("structure_provenance", "")).strip().lower()
    if provenance == "source_hierarchy":
        return "hierarchy"
    if provenance == "source_metadata":
        return "rigid_body"
    return "unknown"


def _absolute_string_list(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    return [
        item.strip()
        for item in value
        if isinstance(item, str) and item.strip().startswith("/")
    ]


def _topology_source_vocabulary(
    source_ids: Sequence[str],
    metadata_index: Mapping[str, Mapping[str, Any]],
) -> set[str]:
    vocabulary: set[str] = set(source_ids)
    modes: set[str] = set()
    for prim_path in source_ids:
        metadata = metadata_index.get(prim_path)
        if not isinstance(metadata, Mapping):
            raise ValueError(f"missing authoritative source metadata for {prim_path}")
        structure = _source_structure_payload(metadata)
        mode = _topology_structure_mode(metadata)
        if mode == "unknown":
            raise ValueError(f"source structure provenance is missing for {prim_path}")
        modes.add(mode)
        if mode == "hierarchy":
            vocabulary.update(
                _absolute_string_list(structure.get("hierarchy_xform_paths"))
            )
            vocabulary.update(
                _absolute_string_list(structure.get("hierarchy_ancestor_xform_paths"))
            )
        else:
            vocabulary.update(
                _absolute_string_list(structure.get("rigid_body_endpoint_paths"))
            )
    if len(modes) != 1:
        raise ValueError("source structure modes conflict within one asset")
    if modes != {"hierarchy"}:
        raise ValueError(
            "v0 topology reconciliation requires authoritative source hierarchy"
        )
    return vocabulary


def _topology_prompt_prediction(
    row: Mapping[str, Any],
    *,
    output_key: str,
) -> dict[str, Any]:
    result = deepcopy(dict(row))
    payload = result.get(output_key)
    if isinstance(payload, dict):
        payload.pop("original_response", None)
    return result


def _topology_prompt_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    structure = _source_structure_payload(metadata)
    allowed_fields = (
        "structure_provenance",
        "rigid_body_endpoint_paths",
        "rigid_body_owner_path",
        "rigid_body_owner_resolution",
        "rigid_body_hierarchy_gap_paths",
        "hierarchy_xform_paths",
        "hierarchy_ancestor_xform_paths",
    )
    return {
        field: deepcopy(structure[field])
        for field in allowed_fields
        if field in structure
    }


def _dedupe_image_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[int] = set()
    for record in records:
        image_index = record.get("image_index")
        if not isinstance(image_index, int) or image_index in seen:
            continue
        seen.add(image_index)
        result.append(dict(record))
    return result


def _validate_topology_reconciliation(
    reconciliation: ArticulationTopologyReconciliationDocument,
    *,
    source_predictions: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    min_confidence: Literal["high", "medium", "low"],
    output_key: str,
) -> None:
    source_ids = _source_prediction_ids(source_predictions)
    source_id_set = set(source_ids)
    vocabulary = _topology_source_vocabulary(source_ids, source_metadata)
    modes = {
        _topology_structure_mode(source_metadata[prim_path]) for prim_path in source_ids
    }
    [structure_mode] = modes

    link_ids: set[str] = set()
    member_to_link: dict[str, ArticulationTopologyLink] = {}
    moving_body_to_link: dict[str, ArticulationTopologyLink] = {}
    for link in reconciliation.links:
        if (
            not link.link_id.strip()
            or link.link_id != link.link_id.strip()
            or link.link_id in link_ids
        ):
            raise ValueError("topology link identifiers must be unique and nonblank")
        if not link.rationale.strip():
            raise ValueError(f"topology link {link.link_id!r} has no rationale")
        if _CONFIDENCE_RANK[link.confidence] < _CONFIDENCE_RANK[min_confidence]:
            raise ValueError(f"topology link {link.link_id!r} is below confidence gate")
        link_ids.add(link.link_id)
        if len(set(link.member_prims)) != len(link.member_prims):
            raise ValueError(f"topology link {link.link_id!r} repeats a member")
        for member in link.member_prims:
            if member not in source_id_set:
                raise ValueError(f"topology link {link.link_id!r} invents {member!r}")
            if member in member_to_link:
                raise ValueError(f"source prim {member!r} belongs to multiple links")
            member_to_link[member] = link
        _validate_topology_link_shape(link, structure_mode=structure_mode)
        if link.kind == "moving":
            assert link.body1 is not None
            if link.body1 in moving_body_to_link:
                raise ValueError(f"multiple moving links use body1 {link.body1!r}")
            moving_body_to_link[link.body1] = link

    if set(member_to_link) != source_id_set:
        missing = sorted(source_id_set - set(member_to_link))
        raise ValueError(f"topology reconciliation omits source prims: {missing}")

    _validate_owned_core_fixed_projection_compatibility(reconciliation)

    endpoint_to_link = dict(member_to_link)
    endpoint_to_link.update(moving_body_to_link)
    if structure_mode == "rigid_body":
        for link in reconciliation.links:
            owner_paths = {
                _rigid_body_owner(source_metadata[member])
                for member in link.member_prims
            }
            owner_paths.discard(None)
            if len(owner_paths) == 1:
                for owner_path in owner_paths:
                    if owner_path is not None:
                        endpoint_to_link[owner_path] = link

    moving_parent_by_link_id: dict[str, str] = {}
    for link in reconciliation.links:
        if link.kind != "moving":
            continue
        assert link.body0 is not None and link.body1 is not None
        if link.body0 not in vocabulary or link.body1 not in vocabulary:
            raise ValueError(f"topology link {link.link_id!r} uses unknown endpoints")
        parent_link = endpoint_to_link.get(link.body0)
        if parent_link is link:
            raise ValueError(f"topology link {link.link_id!r} is self-parented")
        if parent_link is not None and parent_link.kind == "moving":
            if parent_link.anchor_prim != link.body0:
                raise ValueError(f"nested parent {link.body0!r} is not its link anchor")
            moving_parent_by_link_id[link.link_id] = parent_link.link_id
        _validate_source_assembly_containment(
            link,
            parent_link=parent_link,
            source_metadata=source_metadata,
            structure_mode=structure_mode,
        )
    _validate_acyclic_moving_parent_graph(moving_parent_by_link_id)

    incoming_edges = _validated_incoming_compound_edges(
        source_predictions,
        vocabulary=vocabulary,
        output_key=output_key,
    )
    moving_members = {
        member
        for link in reconciliation.links
        if link.kind == "moving"
        for member in link.member_prims
    }
    unassigned_edge_body1s = set(incoming_edges) - moving_members
    if unassigned_edge_body1s:
        raise ValueError(
            "compound edges target fixed or unassigned links: "
            f"{sorted(unassigned_edge_body1s)}"
        )
    for link in reconciliation.links:
        if link.kind != "moving":
            continue
        assert link.body0 is not None and link.body1 is not None
        relevant_edges = [
            edge
            for member in link.member_prims
            for edge in incoming_edges.get(member, ())
        ]
        _validate_semantically_compatible_compound_edges(
            relevant_edges,
            error=f"topology link {link.link_id!r} has conflicting incoming edges",
        )
        partial_edges = [edge for edge in relevant_edges if edge.axis_hint == "unknown"]
        if any(
            edge.body0 != link.body0
            or edge.body1 != link.body1
            or edge.joint_type_hint != link.joint_type_hint
            for edge in partial_edges
        ):
            raise ValueError(
                f"topology link {link.link_id!r} changes the endpoints or joint "
                "type of an unresolved-axis compound edge"
            )
        expected_superseded = {
            _exact_compound_edge_key(edge)
            for edge in relevant_edges
            if edge.axis_hint != "unknown"
            and not _compound_edge_matches_link(edge, link)
        }
        provided_superseded = [
            _exact_compound_edge_key(edge) for edge in link.superseded_compound_edges
        ]
        if len(set(provided_superseded)) != len(provided_superseded):
            raise ValueError(
                f"topology link {link.link_id!r} repeats a superseded edge"
            )
        if set(provided_superseded) != expected_superseded:
            raise ValueError(
                f"topology link {link.link_id!r} does not exactly account for "
                "superseded incoming edges"
            )
        if provided_superseded and (
            link.confidence != "high" or not link.rationale.strip()
        ):
            raise ValueError(
                f"topology link {link.link_id!r} lacks high-confidence "
                "supersession rationale"
            )


def _canonicalize_flat_topology_anchors(
    reconciliation: ArticulationTopologyReconciliationDocument,
    *,
    source_predictions: Sequence[Mapping[str, Any]],
    source_metadata: Mapping[str, Mapping[str, Any]],
    output_key: str,
) -> ArticulationTopologyReconciliationDocument:
    """Stabilize identity-only anchors for flat aggregate moving links.

    A resolved compound edge may identify any member of one rigid aggregate. It
    therefore does not override the aggregate's deterministic representative;
    after normalization the exact raw edge remains accounted for as superseded
    evidence. Unknown-axis edges remain strict endpoint constraints and block
    normalization of any anchor they reference.
    """
    source_ids = _source_prediction_ids(source_predictions)
    vocabulary = _topology_source_vocabulary(source_ids, source_metadata)
    incoming_edges = _validated_incoming_compound_edges(
        source_predictions,
        vocabulary=vocabulary,
        output_key=output_key,
    )
    partial_edge_endpoints = {
        endpoint
        for edges in incoming_edges.values()
        for edge in edges
        if edge.axis_hint == "unknown"
        for endpoint in (edge.body0, edge.body1)
    }
    replacements: dict[str, str] = {}
    for link in reconciliation.links:
        if (
            link.kind != "moving"
            or len(link.member_prims) < 2
            or len({_direct_parent_prim_path(member) for member in link.member_prims})
            != 1
            or link.anchor_prim in partial_edge_endpoints
        ):
            continue
        assert link.anchor_prim is not None
        canonical_anchor = min(
            link.member_prims,
            key=lambda member: (
                len(member.rsplit("/", 1)[-1]),
                member.rsplit("/", 1)[-1],
                member,
            ),
        )
        if canonical_anchor != link.anchor_prim:
            replacements[link.anchor_prim] = canonical_anchor

    if not replacements:
        return reconciliation

    normalized_links: list[ArticulationTopologyLink] = []
    for link in reconciliation.links:
        updates: dict[str, str] = {}
        if link.kind == "moving":
            assert link.anchor_prim is not None
            canonical_anchor = replacements.get(link.anchor_prim)
            if canonical_anchor is not None:
                updates["anchor_prim"] = canonical_anchor
                updates["body1"] = canonical_anchor
            if link.body0 in replacements:
                assert link.body0 is not None
                updates["body0"] = replacements[link.body0]
        normalized_links.append(link.model_copy(update=updates) if updates else link)

    links: list[ArticulationTopologyLink] = []
    for link in normalized_links:
        if link.kind != "moving":
            links.append(link)
            continue
        relevant_edges = [
            edge
            for member in link.member_prims
            for edge in incoming_edges.get(member, ())
        ]
        superseded_by_key = {
            _exact_compound_edge_key(edge): edge
            for edge in relevant_edges
            if edge.axis_hint != "unknown"
            and not _compound_edge_matches_link(edge, link)
        }
        superseded = [superseded_by_key[key] for key in sorted(superseded_by_key)]
        links.append(link.model_copy(update={"superseded_compound_edges": superseded}))
    return reconciliation.model_copy(update={"links": links})


def _validate_topology_link_shape(
    link: ArticulationTopologyLink,
    *,
    structure_mode: str,
) -> None:
    topology_values = (
        link.anchor_prim,
        link.body0,
        link.body1,
        link.joint_type_hint,
        link.axis_hint,
    )
    if link.kind == "fixed":
        if any(value is not None for value in topology_values):
            raise ValueError(f"fixed link {link.link_id!r} carries joint topology")
        if link.role != "body":
            raise ValueError(f"fixed link {link.link_id!r} must use body role")
        if link.superseded_compound_edges:
            raise ValueError(f"fixed link {link.link_id!r} supersedes moving edges")
        return
    if any(value is None for value in topology_values):
        raise ValueError(f"moving link {link.link_id!r} has incomplete topology")
    assert link.anchor_prim is not None
    assert link.body0 is not None
    assert link.body1 is not None
    assert link.joint_type_hint is not None
    assert link.axis_hint is not None
    if link.role in {"body", "unknown"}:
        raise ValueError(f"moving link {link.link_id!r} has no supported moving role")
    if link.anchor_prim not in link.member_prims:
        raise ValueError(f"moving link {link.link_id!r} anchor is not a member")
    if link.body1 != link.anchor_prim:
        raise ValueError(
            f"moving link {link.link_id!r} body1 must equal its sole anchor"
        )
    if link.body0 == link.body1 or link.body0 in link.member_prims:
        raise ValueError(f"moving link {link.link_id!r} is self-parented")
    if link.joint_type_hint not in _TOPOLOGY_MOVING_JOINT_TYPES:
        raise ValueError(f"moving link {link.link_id!r} has unsupported joint type")
    if link.axis_hint not in _TOPOLOGY_AXIS_VALUES:
        raise ValueError(f"moving link {link.link_id!r} has unresolved axis")
    if structure_mode != "hierarchy":
        raise ValueError(
            "v0 topology reconciliation requires row-local hierarchy body1 anchors"
        )


def _validate_owned_core_fixed_projection_compatibility(
    document: ArticulationTopologyReconciliationDocument,
) -> None:
    """Require topology that the owned-core fixed projection can represent."""
    fixed_links = [link for link in document.links if link.kind == "fixed"]
    if len(fixed_links) > 1:
        raise ValueError("topology reconciliation has multiple fixed links")
    if not fixed_links:
        return

    fixed_link = fixed_links[0]
    direct_parents = {
        _direct_parent_prim_path(member) for member in fixed_link.member_prims
    }
    if len(direct_parents) != 1:
        raise ValueError(
            f"fixed link {fixed_link.link_id!r} spans direct-parent assemblies"
        )
    fixed_parent = next(iter(direct_parents))
    if len(fixed_link.member_prims) > 1 and fixed_parent == "/":
        raise ValueError(
            f"fixed link {fixed_link.link_id!r} has no non-root aggregate parent"
        )
    if len(fixed_link.member_prims) > 1:
        other_members = {
            member
            for link in document.links
            if link is not fixed_link
            for member in link.member_prims
        }
        for fixed_member in fixed_link.member_prims:
            for other_member in other_members:
                if _prim_paths_overlap(fixed_member, other_member):
                    raise ValueError(
                        f"fixed aggregate member {fixed_member!r} overlaps another "
                        f"link member {other_member!r}"
                    )

    fixed_aliases = {*fixed_link.member_prims, fixed_parent}
    moving_anchors = {
        link.anchor_prim
        for link in document.links
        if link.kind == "moving" and link.anchor_prim is not None
    }
    for link in document.links:
        if link.kind != "moving":
            continue
        assert link.body0 is not None
        if link.body0 in moving_anchors:
            continue
        if link.body0 not in fixed_aliases:
            raise ValueError(
                f"root moving link {link.link_id!r} body0 is outside fixed aliases"
            )


def _direct_parent_prim_path(path: str) -> str:
    return path.rsplit("/", 1)[0] or "/"


def _prim_paths_overlap(first: str, second: str) -> bool:
    return (
        first == second
        or first.startswith(f"{second}/")
        or second.startswith(f"{first}/")
    )


def _validate_acyclic_moving_parent_graph(
    moving_parent_by_link_id: Mapping[str, str],
) -> None:
    for link_id in moving_parent_by_link_id:
        seen: set[str] = set()
        current: str | None = link_id
        while current is not None:
            if current in seen:
                raise ValueError("topology moving-parent graph contains a cycle")
            seen.add(current)
            current = moving_parent_by_link_id.get(current)


def _hierarchy_ancestors(metadata: Mapping[str, Any]) -> set[str]:
    structure = _source_structure_payload(metadata)
    return set(_absolute_string_list(structure.get("hierarchy_ancestor_xform_paths")))


def _rigid_body_owner(metadata: Mapping[str, Any]) -> str | None:
    structure = _source_structure_payload(metadata)
    owner = structure.get("rigid_body_owner_path")
    if isinstance(owner, str) and owner.strip().startswith("/"):
        return owner.strip()
    return None


def _validate_source_assembly_containment(
    link: ArticulationTopologyLink,
    *,
    parent_link: ArticulationTopologyLink | None,
    source_metadata: Mapping[str, Mapping[str, Any]],
    structure_mode: str,
) -> None:
    assert link.body0 is not None and link.body1 is not None
    if structure_mode == "hierarchy":
        child_common_ancestors: set[str] | None = None
        for member in link.member_prims:
            ancestors = _hierarchy_ancestors(source_metadata[member])
            if not ancestors:
                raise ValueError(f"source hierarchy is incomplete for {member!r}")
            child_common_ancestors = (
                ancestors
                if child_common_ancestors is None
                else child_common_ancestors.intersection(ancestors)
            )
        if not child_common_ancestors:
            raise ValueError(f"topology link {link.link_id!r} spans source assemblies")
        if parent_link is not None and parent_link.kind == "fixed":
            fixed_direct_parents = {
                _direct_parent_prim_path(member) for member in parent_link.member_prims
            }
            if (
                len(fixed_direct_parents) == 1
                and next(iter(fixed_direct_parents)) in child_common_ancestors
            ):
                return
            raise ValueError(
                f"fixed member body0 {link.body0!r} is not in the moving "
                "link's authoritative source assembly"
            )
        if parent_link is None:
            if link.body0 not in child_common_ancestors:
                raise ValueError(
                    f"fixed body0 {link.body0!r} is not an authoritative shared "
                    "hierarchy ancestor"
                )
            return
        parent_common_ancestors: set[str] | None = None
        for member in parent_link.member_prims:
            ancestors = _hierarchy_ancestors(source_metadata[member])
            parent_common_ancestors = (
                ancestors
                if parent_common_ancestors is None
                else parent_common_ancestors.intersection(ancestors)
            )
        if not parent_common_ancestors or not child_common_ancestors.intersection(
            parent_common_ancestors
        ):
            raise ValueError(
                f"body0 for {link.link_id!r} belongs to a different source assembly"
            )
        return

    for member in link.member_prims:
        structure = _source_structure_payload(source_metadata[member])
        endpoints = set(
            _absolute_string_list(structure.get("rigid_body_endpoint_paths"))
        )
        if link.body0 not in endpoints or link.body1 not in endpoints:
            raise ValueError(
                f"topology link {link.link_id!r} crosses rigid-body vocabularies"
            )


def _canonical_axis(value: Any) -> str:
    axis = str(value or "").strip().lower()
    return axis[1:] if axis.startswith("+") else axis


def _validated_incoming_compound_edges(
    source_predictions: Sequence[Mapping[str, Any]],
    *,
    vocabulary: set[str],
    output_key: str,
) -> dict[str, list[ArticulationTopologyCompoundEdge]]:
    incoming: dict[str, list[ArticulationTopologyCompoundEdge]] = {}
    all_edges: list[ArticulationTopologyCompoundEdge] = []
    for row in source_predictions:
        payload = row.get(output_key)
        raw_edges = _raw_compound_edges(payload)
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping):
                raise ValueError("compound edge must be an object")
            edge = _normalize_raw_compound_edge(raw_edge)
            body0 = edge.body0
            body1 = edge.body1
            if body0 not in vocabulary or body1 not in vocabulary or body0 == body1:
                raise ValueError("compound edge is incomplete or outside vocabulary")
            all_edges.append(edge)
            incoming.setdefault(body1, []).append(edge)
    _validate_raw_compound_edge_collection(all_edges)
    return incoming


def _validate_raw_compound_edge_collection(
    edges: Sequence[ArticulationTopologyCompoundEdge],
) -> None:
    """Reject direction or explicit-topology conflicts without source metadata."""
    directed_by_pair: dict[frozenset[str], tuple[str, str]] = {}
    topology_by_endpoints: dict[tuple[str, str], tuple[str, str]] = {}
    for edge in edges:
        body0 = edge.body0
        body1 = edge.body1
        unordered = frozenset((body0, body1))
        previous_direction = directed_by_pair.get(unordered)
        if previous_direction is not None and previous_direction != (body0, body1):
            raise ValueError("compound edge direction is conflicting")
        directed_by_pair[unordered] = (body0, body1)

        endpoint_key = (body0, body1)
        canonical_axis = _canonical_axis(edge.axis_hint)
        topology = (edge.joint_type_hint, canonical_axis)
        previous_topology = topology_by_endpoints.get(endpoint_key)
        if previous_topology is not None:
            previous_joint_type, previous_axis = previous_topology
            if previous_joint_type != edge.joint_type_hint or (
                previous_axis != "unknown"
                and canonical_axis != "unknown"
                and previous_axis != canonical_axis
            ):
                raise ValueError("compound edge topology is conflicting")
            if previous_axis != "unknown":
                topology = previous_topology
        topology_by_endpoints[endpoint_key] = topology


def _preserved_matching_compound_edges(
    source_predictions: Sequence[Mapping[str, Any]],
    document: ArticulationTopologyReconciliationDocument,
    *,
    output_key: str,
) -> dict[str, list[dict[str, Any]]]:
    """Move matching raw edges to their reconciled anchor without rewriting them."""
    link_by_member = {
        member: link for link in document.links for member in link.member_prims
    }
    edge_records: list[
        tuple[
            ArticulationTopologyCompoundEdge,
            Mapping[str, Any],
            ArticulationTopologyLink,
        ]
    ] = []
    edges_by_link_id: dict[str, list[ArticulationTopologyCompoundEdge]] = {}
    for row in source_predictions:
        payload = row.get(output_key)
        raw_edges = _raw_compound_edges(payload)
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, Mapping):
                raise ValueError("compound edge must be an object")
            edge = _normalize_raw_compound_edge(raw_edge)
            target_link = link_by_member.get(edge.body1)
            if target_link is None or target_link.kind != "moving":
                raise ValueError("compound edge targets a fixed or unassigned link")
            edge_records.append((edge, raw_edge, target_link))
            edges_by_link_id.setdefault(target_link.link_id, []).append(edge)

    _validate_raw_compound_edge_collection([edge for edge, _, _ in edge_records])
    for link_id, edges in edges_by_link_id.items():
        _validate_semantically_compatible_compound_edges(
            edges,
            error=f"topology link {link_id!r} has conflicting incoming edges",
        )

    preserved: dict[str, list[dict[str, Any]]] = {}
    superseded_by_link_id: dict[str, set[tuple[str, str, str, str]]] = {}
    for link in document.links:
        provided_superseded = [
            _exact_compound_edge_key(edge) for edge in link.superseded_compound_edges
        ]
        if len(set(provided_superseded)) != len(provided_superseded):
            raise ValueError(
                f"topology link {link.link_id!r} repeats a superseded edge"
            )
        superseded_by_link_id[link.link_id] = set(provided_superseded)
    required_superseded_by_link_id: dict[str, set[tuple[str, str, str, str]]] = {
        link.link_id: set() for link in document.links
    }
    for edge, raw_edge, target_link in edge_records:
        if edge.axis_hint == "unknown":
            if (
                edge.body0 != target_link.body0
                or edge.body1 != target_link.body1
                or edge.joint_type_hint != target_link.joint_type_hint
            ):
                raise ValueError(
                    "reconciled topology changes an unresolved-axis compound "
                    "edge's endpoints or joint type"
                )
            continue
        if _compound_edge_matches_link(edge, target_link):
            preserved.setdefault(target_link.link_id, []).append(
                deepcopy(dict(raw_edge))
            )
            continue
        if (
            _exact_compound_edge_key(edge)
            not in superseded_by_link_id[target_link.link_id]
        ):
            raise ValueError(
                "nonmatching compound edge lacks exact topology supersession"
            )
        required_superseded_by_link_id[target_link.link_id].add(
            _exact_compound_edge_key(edge)
        )
    for link_id, provided_superseded in superseded_by_link_id.items():
        if provided_superseded != required_superseded_by_link_id[link_id]:
            raise ValueError(
                f"topology link {link_id!r} contains invented or inapplicable "
                "superseded edges"
            )
    return preserved


def _raw_compound_edges(payload: Any) -> list[Any]:
    """Return a present compound-edge list while rejecting malformed containers."""
    if not isinstance(payload, Mapping):
        return []
    if "rigger_evidence" not in payload or payload["rigger_evidence"] is None:
        return []
    evidence = payload["rigger_evidence"]
    if not isinstance(evidence, Mapping):
        raise ValueError("rigger_evidence must be an object or null")
    if "compound_edges" not in evidence:
        return []
    raw_edges = evidence["compound_edges"]
    if not isinstance(raw_edges, list):
        raise ValueError("compound_edges must be a list when present")
    return raw_edges


def _normalize_raw_compound_edge(
    raw_edge: Mapping[str, Any],
) -> ArticulationTopologyCompoundEdge:
    body0 = _compound_endpoint(raw_edge.get("body0"))
    body1 = _compound_endpoint(raw_edge.get("body1"))
    joint_type = str(raw_edge.get("joint_type_hint", "")).strip().lower()
    axis = str(raw_edge.get("axis_hint", "")).strip().lower()
    if (
        not body0
        or not body1
        or body0 == body1
        or joint_type not in _TOPOLOGY_MOVING_JOINT_TYPES
        or axis not in {*_TOPOLOGY_AXIS_VALUES, "unknown"}
    ):
        raise ValueError("compound edge is incomplete")
    return ArticulationTopologyCompoundEdge(
        body0=body0,
        body1=body1,
        joint_type_hint=joint_type,  # type: ignore[arg-type]
        axis_hint=axis,  # type: ignore[arg-type]
    )


def _exact_compound_edge_key(
    edge: ArticulationTopologyCompoundEdge,
) -> tuple[str, str, str, str]:
    return (edge.body0, edge.body1, edge.joint_type_hint, edge.axis_hint)


def _compound_edges_semantically_compatible(
    first: ArticulationTopologyCompoundEdge,
    second: ArticulationTopologyCompoundEdge,
) -> bool:
    """Allow an unknown axis to refine, without weakening edge identity."""
    if (
        first.body0 != second.body0
        or first.body1 != second.body1
        or first.joint_type_hint != second.joint_type_hint
    ):
        return False
    first_axis = _canonical_axis(first.axis_hint)
    second_axis = _canonical_axis(second.axis_hint)
    return (
        first_axis == "unknown" or second_axis == "unknown" or first_axis == second_axis
    )


def _validate_semantically_compatible_compound_edges(
    edges: Sequence[ArticulationTopologyCompoundEdge],
    *,
    error: str,
) -> None:
    """Reject link-local edge disagreement while allowing unknown-axis refinement."""
    if any(
        not _compound_edges_semantically_compatible(first, second)
        for index, first in enumerate(edges)
        for second in edges[index + 1 :]
    ):
        raise ValueError(error)


def _compound_edge_matches_link(
    edge: ArticulationTopologyCompoundEdge,
    link: ArticulationTopologyLink,
) -> bool:
    return (
        edge.body0 == link.body0
        and edge.body1 == link.body1
        and edge.joint_type_hint == link.joint_type_hint
        and _canonical_axis(edge.axis_hint) == _canonical_axis(link.axis_hint)
    )


def _compound_endpoint(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("value")
    return str(value or "").strip()


def _clear_reconciled_consistency_conflicts(payload: dict[str, Any]) -> None:
    consistency = payload.get("consistency")
    if not isinstance(consistency, dict):
        return
    flagged_fields = consistency.get("flagged_fields")
    if isinstance(flagged_fields, dict):
        for field in (
            "role",
            "instance_id",
            "is_articulation_candidate",
            "joint_type_hint",
            "axis_hint",
            "parent_hint",
            "child_hint",
        ):
            flagged_fields.pop(field, None)
        if not flagged_fields:
            consistency.pop("flagged_fields", None)
    consistency["topology_reconciliation"] = {
        "schema_version": TOPOLOGY_RECONCILIATION_SCHEMA_VERSION,
        "source": "llm_adjudicated",
    }


def _coerce_adjudication(
    adjudication: ArticulationConflictAdjudication | Mapping[str, Any],
) -> ArticulationConflictAdjudication:
    if isinstance(adjudication, ArticulationConflictAdjudication):
        return adjudication
    return ArticulationConflictAdjudication.model_validate(adjudication)


def _has_compound_edge_conflict(candidate: Mapping[str, Any]) -> bool:
    reason_codes = candidate.get("unresolved_reason_codes", [])
    return isinstance(reason_codes, list) and "compound_edge_conflict" in reason_codes


def _accepts_existing_candidate(
    candidate: Mapping[str, Any],
    adjudication: ArticulationConflictAdjudication,
    *,
    min_confidence: Literal["high", "medium", "low"],
) -> bool:
    if not _has_compound_edge_conflict(candidate):
        return False
    if adjudication.decision != "accept_candidate":
        return False
    if "compound_edge_conflict" not in adjudication.resolved_reason_codes:
        return False
    if _CONFIDENCE_RANK[adjudication.confidence] < _CONFIDENCE_RANK[min_confidence]:
        return False
    if not adjudication.rationale.strip():
        return False
    return (
        _field_matches(candidate, adjudication.motion_type, "motion_type")
        and _field_matches(
            candidate, adjudication.fixed_parent_prim, "fixed_parent_prim"
        )
        and _field_matches(candidate, adjudication.axis_hint, "axis_hint")
    )


def _field_matches(
    candidate: Mapping[str, Any],
    adjudicated_value: str | None,
    field_name: str,
) -> bool:
    candidate_value = candidate.get(field_name)
    if candidate_value is None or adjudicated_value is None:
        return candidate_value is None and adjudicated_value is None
    return str(candidate_value) == str(adjudicated_value)


def _clear_compound_edge_conflict(
    candidate: dict[str, Any],
    adjudication: ArticulationConflictAdjudication,
) -> None:
    candidate["unresolved_reason_codes"] = [
        code
        for code in candidate.get("unresolved_reason_codes", [])
        if code != "compound_edge_conflict"
    ]
    candidate["unresolved_questions"] = [
        question
        for question in candidate.get("unresolved_questions", [])
        if "compound-edge evidence conflict" not in question
    ]
    evidence_item = Stage2EvidenceItem(
        source="llm_adjudicated",
        description=(
            "LLM adjudication accepted existing Stage 2 fields and resolved "
            "the compound-edge evidence conflict."
        ),
        value=f"{adjudication.confidence}: {adjudication.rationale}",
        prim_paths=list(candidate.get("moving_part_prims", [])),
    )
    candidate.setdefault("adjudication_evidence", []).append(
        evidence_item.model_dump(mode="json")
    )
    if not candidate["unresolved_reason_codes"]:
        candidate["review_status"] = READY_FOR_RIGGER_INPUT_STATUS
    else:
        candidate["review_status"] = REVIEW_REQUIRED_STATUS


def _candidate_summary(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    joint_counts = Counter(
        str(candidate.get("joint_type_hint", "unknown")) for candidate in candidates
    )
    review_status_counts = Counter(
        str(candidate.get("review_status", REVIEW_REQUIRED_STATUS))
        for candidate in candidates
    )
    limit_readiness_counts = Counter(
        str(candidate.get("limit_readiness", "not_provided"))
        for candidate in candidates
    )
    reason_code_counts = Counter(
        str(code)
        for candidate in candidates
        for code in candidate.get("unresolved_reason_codes", [])
    )
    unresolved_axis_count = sum(
        1 for candidate in candidates if candidate.get("motion_axis_world") is None
    )
    unresolved_parent_count = sum(
        1 for candidate in candidates if candidate.get("fixed_parent_prim") is None
    )
    return {
        "total_predictions": 0,
        "candidate_count": len(candidates),
        "ready_candidate_count": review_status_counts.get(
            READY_FOR_RIGGER_INPUT_STATUS,
            0,
        ),
        "review_required_candidate_count": review_status_counts.get(
            REVIEW_REQUIRED_STATUS,
            0,
        ),
        "joint_type_counts": dict(sorted(joint_counts.items())),
        "unresolved_axis_count": unresolved_axis_count,
        "unresolved_parent_count": unresolved_parent_count,
        "review_status_counts": dict(sorted(review_status_counts.items())),
        "limit_readiness_counts": dict(sorted(limit_readiness_counts.items())),
        "reason_code_counts": dict(sorted(reason_code_counts.items())),
    }
