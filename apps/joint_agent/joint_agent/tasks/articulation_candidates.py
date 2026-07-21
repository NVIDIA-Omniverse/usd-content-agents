# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 articulation candidate inference task."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

from world_understanding.agentic.domain_tasks import ModelProvisioningTask
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore

from joint_agent.functions.articulation_adjudication import (
    ADJUDICATION_ARTIFACT_SCHEMA_VERSION,
    ArticulationAdjudicationArtifact,
    ArticulationConflictAdjudication,
    ArticulationTopologyReconciliationDocument,
    adjudicate_articulation_conflicts_with_model,
    apply_articulation_conflict_adjudications,
    apply_articulation_topology_reconciliation,
    reconcile_articulation_topology_with_model,
    recover_articulation_topology_reconciliation_from_history,
    restore_articulation_topology_reconciliation_originals,
)
from joint_agent.functions.articulation_candidates import (
    infer_articulation_candidates,
    load_predictions_jsonl,
    write_articulation_candidate_report_html,
    write_json,
)
from joint_agent.functions.consistency import (
    effective_prediction_payload,
    has_topology_reconciliation_trace,
    write_predictions_jsonl,
)

logger = logging.getLogger(__name__)

_SOURCE_STRUCTURE_FIELDS = frozenset(
    {
        "usd_metadata",
        "structure_provenance",
        "rigid_body_endpoint_paths",
        "rigid_body_owner_path",
        "rigid_body_owner_resolution",
        "rigid_body_hierarchy_gap_paths",
        "hierarchy_xform_paths",
        "hierarchy_ancestor_xform_paths",
    }
)
_TOPOLOGY_RECONCILIATION_REASON_CODES = frozenset(
    {
        "axis_evidence_conflict",
        "axis_missing",
        "axis_non_axis_aligned",
        "body1_unresolved",
        "candidate_flag_conflict",
        "joint_type_conflict",
        "link_membership_conflict",
        "parent_self_reference",
        "parent_unresolved",
    }
)
_UNKNOWN_ROLE_VALUES = frozenset({"", "unknown", "none", "null"})
_RECONCILABLE_ROLE_SOURCES = frozenset(
    {
        "",
        "predicted",
        "llm",
        "vlm",
        "llm_vlm",
        "model",
        "stage1",
        "stage1_model",
        "consistency_corrected",
    }
)


def _has_authoritative_source_structure(row: dict[str, Any]) -> bool:
    usd_metadata = row.get("usd_metadata")
    provenance_values = [row.get("structure_provenance")]
    if isinstance(usd_metadata, dict):
        provenance_values.append(usd_metadata.get("structure_provenance"))
    return any(
        str(value or "").strip().lower() in {"source_metadata", "source_hierarchy"}
        for value in provenance_values
    )


def _merge_source_metadata_row(
    verified_row: dict[str, Any],
    overlay_row: dict[str, Any],
    *,
    allow_overlay_structure: bool,
) -> tuple[dict[str, Any], bool]:
    """Merge auxiliary metadata without replacing dataset-verified structure."""
    overlay = dict(overlay_row)
    suppressed_structure = False
    if not allow_overlay_structure:
        suppressed_structure = bool(_SOURCE_STRUCTURE_FIELDS.intersection(overlay))
        for field in _SOURCE_STRUCTURE_FIELDS:
            overlay.pop(field, None)

    verified_has_structure = "usd_metadata" in verified_row
    verified_structure = verified_row.get("usd_metadata")
    merged_row = dict(verified_row)
    merged_row.update(overlay)
    if verified_has_structure:
        merged_row["usd_metadata"] = verified_structure
        suppressed_structure = suppressed_structure or (
            "usd_metadata" in overlay_row
            and overlay_row.get("usd_metadata") != verified_structure
        )
    return merged_row, suppressed_structure


class AdjudicationModelProvisioningTask(ModelProvisioningTask):
    """Provision the optional adjudication model without blocking Stage 2."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "AdjudicationModelProvisioning"
        self.description = "Provision an optional fail-closed adjudication model"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        config = context.get("config", {})
        adjudication_config = (
            config.get("adjudication", {}) if isinstance(config, dict) else {}
        )
        if not isinstance(adjudication_config, dict) or not adjudication_config.get(
            "enabled", False
        ):
            return context

        listener = get_listener(context, logger_name=__name__)
        try:
            return super().run(context, object_store)
        except Exception as exc:
            listener.warning(
                "Optional articulation adjudication model provisioning failed; "
                "candidates will stay review-required "
                f"({type(exc).__name__})"
            )
            context["vlm"] = None
            context["llm"] = None
            context["vlm_judge"] = None
            context["llm_judge"] = None
            context["adjudication_model_error"] = type(exc).__name__
            return context


class ArticulationCandidatesTask(Task):
    """Infer report-only articulation candidates from Stage 1 predictions."""

    def __init__(self) -> None:
        self.name = "ArticulationCandidates"
        self.description = "Infer Stage 2 articulation candidates"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        listener = get_listener(context, logger_name=__name__)
        context["articulation_topology_reconciled"] = False

        predictions_path = Path(context["predictions_path"])
        output_candidates_path = Path(context["output_candidates_path"])
        output_report_path = Path(context["output_report_path"])

        if not predictions_path.exists():
            raise FileNotFoundError(f"Predictions file not found: {predictions_path}")

        predictions = load_predictions_jsonl(predictions_path)
        listener.info(
            f"Loaded {len(predictions)} predictions for articulation candidates"
        )

        dataset_entries: list[dict[str, Any]] | None = None
        dataset_path = context.get("dataset_path")
        if dataset_path:
            dataset_path_obj = Path(dataset_path)
            if dataset_path_obj.exists():
                dataset_entries = load_predictions_jsonl(dataset_path_obj)

        source_metadata_by_id: dict[str, dict[str, Any]] = {}
        for row in dataset_entries or []:
            row_id = str(row.get("id", ""))
            if row_id:
                source_metadata_by_id[row_id] = dict(row)
        dataset_has_authoritative_structure = any(
            _has_authoritative_source_structure(row)
            for row in source_metadata_by_id.values()
        )
        prim_metadata_path = context.get("prim_metadata_path")
        if prim_metadata_path:
            prim_metadata_path_obj = Path(prim_metadata_path)
            if prim_metadata_path_obj.exists():
                suppressed_structure_rows = 0
                for row in load_predictions_jsonl(prim_metadata_path_obj):
                    row_id = str(
                        row.get("id") or row.get("prim_path") or row.get("path") or ""
                    )
                    if not row_id:
                        continue
                    merged_row, suppressed_structure = _merge_source_metadata_row(
                        source_metadata_by_id.get(row_id, {}),
                        row,
                        allow_overlay_structure=(
                            not dataset_has_authoritative_structure
                        ),
                    )
                    source_metadata_by_id[row_id] = merged_row
                    suppressed_structure_rows += int(suppressed_structure)
                if suppressed_structure_rows:
                    listener.warning(
                        "Ignored prim_metadata_path source-structure fields in "
                        f"{suppressed_structure_rows} row(s); dataset source "
                        "structure remains authoritative"
                    )
            else:
                listener.warning(
                    "Prim metadata path does not exist; continuing without it: "
                    f"{prim_metadata_path_obj}"
                )
        if source_metadata_by_id:
            listener.info(
                f"Loaded source metadata for {len(source_metadata_by_id)} "
                "prediction rows"
            )

        output_key = context.get("output_key", "classification")
        adjudication_config = context.get("adjudication_config", {})
        adjudication_enabled = bool(adjudication_config.get("enabled", False))
        topology_reconciliation_requested = bool(
            adjudication_enabled
            and adjudication_config.get("reconcile_topology", False)
        )
        context["articulation_topology_reconciliation_status"] = {
            "requested": topology_reconciliation_requested,
            "attempted": False,
            "accepted": False,
            "outcome": (
                "not_attempted"
                if topology_reconciliation_requested
                else "not_requested"
            ),
        }
        if self._has_topology_reconciliation_trace(
            predictions,
            output_key=output_key,
        ):
            recovered = recover_articulation_topology_reconciliation_from_history(
                predictions,
                output_key=output_key,
                source_metadata=source_metadata_by_id or None,
            )
            if recovered is None:
                self._record_topology_reconciliation_status(
                    context,
                    attempted=False,
                    accepted=False,
                    outcome="failed",
                    failure_stage="receipt_recovery",
                    error_type="ValueError",
                )
                self._remove_adjudication_artifact(context)
                raise ValueError(
                    "Canonical predictions contain an invalid or incomplete "
                    "topology reconciliation receipt"
                )
            retain_reconciliation = bool(
                adjudication_enabled
                and adjudication_config.get("reconcile_topology", False)
                and adjudication_config.get("min_confidence", "high") == "high"
            )
            if not retain_reconciliation:
                predictions = restore_articulation_topology_reconciliation_originals(
                    predictions,
                    output_key=output_key,
                    source_metadata=source_metadata_by_id or None,
                )
                # Invalidate the old acceptance claim before committing restored
                # rows or running inference. Any later write/inference failure is
                # therefore fail-closed rather than leaving a stale receipt.
                self._remove_adjudication_artifact(context)
                write_predictions_jsonl(predictions_path, predictions)
                listener.info(
                    "Restored canonical predictions to their pre-reconciliation "
                    "values for the current adjudication policy"
                )

        candidate_document = infer_articulation_candidates(
            predictions,
            output_key=output_key,
            candidate_joint_types=context.get("candidate_joint_types"),
            prim_metadata=source_metadata_by_id,
        )
        if adjudication_enabled:
            model_key = adjudication_config.get("model_key", "llm")
            use_source_images = model_key in {"vlm", "vlm_judge"}
            image_base_dir: Path | None = None
            configured_dataset_missing = False
            require_source_images = bool(
                adjudication_config.get("require_source_images", False)
            )
            if use_source_images and dataset_path:
                dataset_path_obj = Path(dataset_path)
                if not dataset_path_obj.exists():
                    configured_dataset_missing = True
                    requirement = "required" if require_source_images else "configured"
                    listener.warning(
                        f"Dataset file not found for {requirement} source-image "
                        "adjudication; candidates will stay review-required: "
                        f"{dataset_path_obj}"
                    )
                else:
                    image_base_dir = dataset_path_obj.parent
            (
                candidate_document,
                reconciled_predictions,
                pending_topology_reconciliation,
            ) = self._run_adjudication(
                context=context,
                predictions=predictions,
                candidate_document=candidate_document,
                dataset_entries=dataset_entries,
                image_base_dir=image_base_dir,
                source_metadata_by_id=source_metadata_by_id,
                configured_dataset_missing=configured_dataset_missing,
            )
            if pending_topology_reconciliation is not None:
                if reconciled_predictions == predictions:
                    raise RuntimeError(
                        "Accepted topology reconciliation did not produce a "
                        "canonical prediction receipt"
                    )
                # Service/default runs intentionally use this stable path as the
                # canonical post-processing artifact, just as consistency does.
                # Each overwritten field retains its pre-adjudication value in
                # the row's provenance history.
                write_predictions_jsonl(
                    predictions_path,
                    reconciled_predictions,
                )
                predictions = reconciled_predictions
                # Publish the acceptance artifact only after the canonical rows
                # containing its reversible receipt have been committed. This
                # prevents a failed prediction write from leaving a false
                # durable claim that reconciliation was accepted.
                self._write_adjudication_artifact(
                    context,
                    topology_reconciliation=pending_topology_reconciliation,
                )
                context["articulation_topology_reconciled"] = True
                self._record_topology_reconciliation_status(
                    context,
                    attempted=True,
                    accepted=True,
                    outcome="accepted",
                )
                moving_link_count = sum(
                    link.kind == "moving"
                    for link in pending_topology_reconciliation.links
                )
                listener.info(
                    "Asset-level topology reconciliation accepted: "
                    f"{moving_link_count} moving links are strict-Stage-2 ready"
                )
            elif reconciled_predictions != predictions:
                write_predictions_jsonl(
                    predictions_path,
                    reconciled_predictions,
                )
                predictions = reconciled_predictions
        else:
            # The service reuses a stable output directory across reruns. An
            # opt-out run must not leave a previous run's model decisions in
            # place as if they were current evidence.
            self._remove_adjudication_artifact(context)

        write_json(output_candidates_path, candidate_document)
        write_articulation_candidate_report_html(
            output_report_path,
            candidate_document,
        )

        summary = candidate_document["summary"]
        listener.info(
            "Articulation candidate inference complete: "
            f"{summary['candidate_count']} candidates, "
            f"{summary['ready_candidate_count']} ready, "
            f"{summary['review_required_candidate_count']} need review, "
            f"{summary['unresolved_axis_count']} missing axis hints, "
            f"{summary['unresolved_parent_count']} missing parent links"
        )

        context["articulation_candidates_path"] = str(output_candidates_path)
        context["articulation_report_path"] = str(output_report_path)
        context["articulation_summary"] = summary
        context["articulation_candidate_count"] = summary["candidate_count"]
        return context

    @staticmethod
    def _record_topology_reconciliation_status(
        context: dict[str, Any],
        *,
        attempted: bool,
        accepted: bool,
        outcome: str,
        failure_stage: str | None = None,
        error_type: str | None = None,
    ) -> None:
        current = context.get("articulation_topology_reconciliation_status", {})
        requested = bool(isinstance(current, dict) and current.get("requested", False))
        status: dict[str, Any] = {
            "requested": requested,
            "attempted": attempted,
            "accepted": accepted,
            "outcome": outcome,
        }
        if failure_stage is not None:
            status["failure_stage"] = failure_stage
        if error_type is not None:
            status["error_type"] = error_type
        context["articulation_topology_reconciliation_status"] = status

    def _run_adjudication(
        self,
        *,
        context: dict[str, Any],
        predictions: list[dict[str, Any]],
        candidate_document: dict[str, Any],
        dataset_entries: list[dict[str, Any]] | None,
        image_base_dir: Path | None,
        source_metadata_by_id: dict[str, dict[str, Any]],
        configured_dataset_missing: bool,
    ) -> tuple[
        dict[str, Any],
        list[dict[str, Any]],
        ArticulationTopologyReconciliationDocument | None,
    ]:
        listener = get_listener(context, logger_name=__name__)
        adjudication_config = context.get("adjudication_config", {})
        model_key = adjudication_config.get("model_key", "llm")

        if adjudication_config.get("reconcile_topology", False):
            has_topology_trace = self._has_topology_reconciliation_trace(
                predictions,
                output_key=context.get("output_key", "classification"),
            )
            recovered = recover_articulation_topology_reconciliation_from_history(
                predictions,
                output_key=context.get("output_key", "classification"),
                source_metadata=source_metadata_by_id or None,
            )
            if recovered is not None:
                moving_link_count = sum(
                    link.kind == "moving" for link in recovered.links
                )
                if self._is_complete_topology_reconciliation(
                    candidate_document,
                    moving_link_count=moving_link_count,
                ):
                    self._write_adjudication_artifact(
                        context,
                        topology_reconciliation=recovered,
                    )
                    context["articulation_topology_reconciled"] = True
                    self._record_topology_reconciliation_status(
                        context,
                        attempted=False,
                        accepted=True,
                        outcome="recovered",
                    )
                    listener.info(
                        "Recovered the accepted asset-level topology receipt; "
                        "skipping another model request"
                    )
                    return candidate_document, predictions, None
                self._remove_adjudication_artifact(context)
                raise ValueError(
                    "Stored topology reconciliation no longer survives strict "
                    "Stage 2 inference"
                )
            if has_topology_trace:
                self._remove_adjudication_artifact(context)
                raise ValueError(
                    "Stored topology reconciliation history is incomplete or no "
                    "longer matches the current rows"
                )

        topology_reconciliation_required = bool(
            adjudication_config.get("reconcile_topology", False)
            and self._requires_topology_reconciliation(
                predictions,
                candidate_document,
                output_key=context.get("output_key", "classification"),
            )
        )
        if (
            adjudication_config.get("reconcile_topology", False)
            and not topology_reconciliation_required
        ):
            self._record_topology_reconciliation_status(
                context,
                attempted=False,
                accepted=False,
                outcome="not_required",
            )

        if configured_dataset_missing:
            if topology_reconciliation_required:
                self._record_topology_reconciliation_status(
                    context,
                    attempted=False,
                    accepted=False,
                    outcome="failed",
                    failure_stage="request",
                    error_type="FileNotFoundError",
                )
            self._write_empty_adjudication_artifact(context)
            return candidate_document, predictions, None

        model = context.get(model_key)
        if model is None:
            listener.warning(
                "No adjudication model is available at context key "
                f"'{model_key}'; candidates will stay review-required"
            )
            if topology_reconciliation_required:
                model_error = context.get("adjudication_model_error")
                self._record_topology_reconciliation_status(
                    context,
                    attempted=False,
                    accepted=False,
                    outcome="failed",
                    failure_stage="model_provisioning",
                    error_type=(
                        str(model_error) if model_error else "ModelUnavailableError"
                    ),
                )
            self._write_empty_adjudication_artifact(context)
            return candidate_document, predictions, None

        if topology_reconciliation_required:
            listener.info(
                "Running bounded asset-level VLM topology reconciliation before "
                "strict Stage 2 reinference"
            )
            self._record_topology_reconciliation_status(
                context,
                attempted=True,
                accepted=False,
                outcome="attempting",
            )
            reconciliation_diagnostics: dict[str, str] = {}
            reconciliation = reconcile_articulation_topology_with_model(
                model=model,
                candidate_document=candidate_document,
                source_predictions=predictions,
                source_metadata=source_metadata_by_id,
                dataset_entries=dataset_entries,
                image_base_dir=image_base_dir,
                max_images=int(adjudication_config.get("max_images", 64)),
                require_images=True,
                use_images=True,
                min_confidence=cast(
                    Literal["high", "medium", "low"],
                    adjudication_config.get("min_confidence", "high"),
                ),
                temperature=float(adjudication_config.get("temperature", 0.0)),
                max_tokens=int(adjudication_config.get("max_tokens", 8192)),
                output_key=context.get("output_key", "classification"),
                diagnostics=reconciliation_diagnostics,
            )
            if reconciliation is None:
                self._record_topology_reconciliation_status(
                    context,
                    attempted=True,
                    accepted=False,
                    outcome="failed",
                    failure_stage=reconciliation_diagnostics.get(
                        "failure_stage", "unknown"
                    ),
                    error_type=reconciliation_diagnostics.get(
                        "error_type", "RuntimeError"
                    ),
                )
                self._write_empty_adjudication_artifact(context)
                return candidate_document, predictions, None

            self._record_topology_reconciliation_status(
                context,
                attempted=True,
                accepted=False,
                outcome="validated",
            )

            try:
                reconciled_predictions = apply_articulation_topology_reconciliation(
                    predictions,
                    reconciliation,
                    output_key=context.get("output_key", "classification"),
                )
                reconciled_document = infer_articulation_candidates(
                    reconciled_predictions,
                    output_key=context.get("output_key", "classification"),
                    candidate_joint_types=context.get("candidate_joint_types"),
                    prim_metadata=source_metadata_by_id,
                )
            except (TypeError, ValueError) as exc:
                listener.warning(
                    "Validated topology reconciliation could not be strictly "
                    f"replayed; predictions remain unchanged ({type(exc).__name__})"
                )
                self._record_topology_reconciliation_status(
                    context,
                    attempted=True,
                    accepted=False,
                    outcome="failed",
                    failure_stage="replay",
                    error_type=type(exc).__name__,
                )
                self._write_empty_adjudication_artifact(context)
                return candidate_document, predictions, None

            moving_link_count = sum(
                link.kind == "moving" for link in reconciliation.links
            )
            if not self._is_complete_topology_reconciliation(
                reconciled_document,
                moving_link_count=moving_link_count,
            ):
                listener.warning(
                    "Topology reconciliation did not survive strict Stage 2 "
                    "reinference; predictions remain unchanged"
                )
                self._record_topology_reconciliation_status(
                    context,
                    attempted=True,
                    accepted=False,
                    outcome="failed",
                    failure_stage="reinference",
                    error_type="ValueError",
                )
                self._write_empty_adjudication_artifact(context)
                return candidate_document, predictions, None

            return reconciled_document, reconciled_predictions, reconciliation

        conflict_count = sum(
            1
            for candidate in candidate_document.get("candidates", [])
            if "compound_edge_conflict" in candidate.get("unresolved_reason_codes", [])
        )
        if conflict_count == 0:
            listener.info("No compound-edge conflicts found for LLM adjudication")
            self._write_empty_adjudication_artifact(context)
            return candidate_document, predictions, None

        listener.info(
            "Running LLM adjudication for "
            f"{conflict_count} compound-edge conflict candidates"
        )
        use_source_images = model_key in {"vlm", "vlm_judge"}
        adjudication_document = adjudicate_articulation_conflicts_with_model(
            model=model,
            candidate_document=candidate_document,
            source_predictions=predictions,
            dataset_entries=dataset_entries if use_source_images else None,
            image_base_dir=image_base_dir if use_source_images else None,
            max_images=(
                int(adjudication_config.get("max_images", 16))
                if use_source_images
                else 0
            ),
            use_images=use_source_images,
            require_images=bool(
                adjudication_config.get("require_source_images", False)
            ),
            max_adjudications=int(adjudication_config.get("max_adjudications", 8)),
            temperature=float(adjudication_config.get("temperature", 0.0)),
            max_tokens=int(adjudication_config.get("max_tokens", 4096)),
        )
        self._write_adjudication_artifact(
            context,
            adjudications=adjudication_document.adjudications,
        )

        raw_min_confidence = adjudication_config.get("min_confidence", "high")
        min_confidence = (
            raw_min_confidence
            if raw_min_confidence in {"high", "medium", "low"}
            else "high"
        )
        updated_document: dict[str, Any] = apply_articulation_conflict_adjudications(
            candidate_document,
            adjudication_document.adjudications,
            min_confidence=cast(
                Literal["high", "medium", "low"],
                min_confidence,
            ),
        )
        return updated_document, predictions, None

    @staticmethod
    def _requires_topology_reconciliation(
        predictions: list[dict[str, Any]],
        candidate_document: dict[str, Any],
        *,
        output_key: str,
    ) -> bool:
        for row in predictions:
            payload = effective_prediction_payload(row, output_key=output_key)
            if payload is None:
                continue
            role = str(payload.get("role", "")).strip().lower()
            if role in _UNKNOWN_ROLE_VALUES:
                return True
            consistency = payload.get("consistency")
            flagged_fields = (
                consistency.get("flagged_fields")
                if isinstance(consistency, dict)
                else None
            )
            role_flag = (
                flagged_fields.get("role") if isinstance(flagged_fields, dict) else None
            )
            if isinstance(role_flag, dict):
                source = str(role_flag.get("source", "")).strip().lower()
                if source in _RECONCILABLE_ROLE_SOURCES:
                    return True
        return bool(
            ArticulationCandidatesTask._topology_reconciliation_reasons(
                candidate_document
            )
        )

    @staticmethod
    def _topology_reconciliation_reasons(
        candidate_document: dict[str, Any],
    ) -> set[str]:
        return {
            str(reason)
            for candidate in candidate_document.get("candidates", [])
            for reason in candidate.get("unresolved_reason_codes", [])
            if reason in _TOPOLOGY_RECONCILIATION_REASON_CODES
        }

    @staticmethod
    def _has_topology_reconciliation_trace(
        predictions: list[dict[str, Any]],
        *,
        output_key: str,
    ) -> bool:
        return any(
            has_topology_reconciliation_trace(row, output_key=output_key)
            for row in predictions
        )

    @staticmethod
    def _is_complete_topology_reconciliation(
        candidate_document: dict[str, Any],
        *,
        moving_link_count: int,
    ) -> bool:
        summary = candidate_document.get("summary", {})
        return bool(
            moving_link_count > 0
            and summary.get("candidate_count") == moving_link_count
            and summary.get("ready_candidate_count") == moving_link_count
            and summary.get("review_required_candidate_count") == 0
            and not ArticulationCandidatesTask._topology_reconciliation_reasons(
                candidate_document
            )
        )

    @staticmethod
    def _write_empty_adjudication_artifact(context: dict[str, Any]) -> None:
        ArticulationCandidatesTask._write_adjudication_artifact(context)

    @staticmethod
    def _remove_adjudication_artifact(context: dict[str, Any]) -> None:
        output_adjudications_path = context.get("output_adjudications_path")
        if output_adjudications_path:
            Path(output_adjudications_path).unlink(missing_ok=True)
        context.pop("articulation_adjudications_path", None)

    @staticmethod
    def _write_adjudication_artifact(
        context: dict[str, Any],
        *,
        topology_reconciliation: ArticulationTopologyReconciliationDocument
        | None = None,
        adjudications: list[ArticulationConflictAdjudication] | None = None,
    ) -> None:
        output_adjudications_path = context.get("output_adjudications_path")
        if output_adjudications_path:
            if topology_reconciliation is not None:
                topology_reconciliation = topology_reconciliation.model_copy(
                    update={
                        "links": sorted(
                            topology_reconciliation.links,
                            key=lambda link: (
                                0 if link.kind == "fixed" else 1,
                                link.link_id,
                            ),
                        )
                    }
                )
            artifact = ArticulationAdjudicationArtifact(
                schema_version=ADJUDICATION_ARTIFACT_SCHEMA_VERSION,
                topology_reconciliation=topology_reconciliation,
                adjudications=adjudications or [],
            )
            write_json(
                output_adjudications_path,
                artifact.model_dump(mode="json"),
            )
            context["articulation_adjudications_path"] = str(output_adjudications_path)
