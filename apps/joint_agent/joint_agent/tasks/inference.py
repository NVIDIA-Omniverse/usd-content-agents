# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM inference task for asset classification."""

import json
import logging
from pathlib import Path
from threading import Lock
from typing import Any, NoReturn, cast

from langchain_core.language_models.chat_models import BaseChatModel
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.utils.object_store import ObjectStore
from world_understanding.utils.token_tracking import TokenTracker, format_token_stats

from joint_agent.functions.consistency import write_predictions_jsonl
from joint_agent.functions.inference import batch_classify_assets
from joint_agent.functions.provider_response_conformance import (
    ProviderAttemptDiagnostic,
    ProviderAttemptJournal,
    ProviderAttemptOutcome,
    ProviderAttemptRecorderError,
    ProviderAttemptRecording,
    ProviderRequestKind,
    ProviderResponseConformanceTerminalError,
    build_provider_attempt_diagnostic,
    evaluate_exhausted_transport_terminal,
    evaluate_stage1_response,
    invoke_provider_attempt_recorder,
    load_provider_attempt_journal,
    persist_provider_attempt_journal,
    project_provider_attempt_persistence,
    require_provider_attempt_journal_persistence,
    run_provider_call_with_journal,
)
from joint_agent.functions.stage1_schema import (
    STAGE1_SCHEMA_VERSION,
    has_parseable_stage1_source_response,
)

logger = logging.getLogger(__name__)

_DEFAULT_COMPLETION_RETRIES = 3
_OMIT_PREDICTION_MEDIA = object()


def _stage1_completion_error(response: Any, *, output_key: str) -> str | None:
    """Return why a nominal Stage 1 classification needs bounded recovery."""
    if (
        output_key != "classification"
        or not isinstance(response, dict)
        or response.get("schema_version") != STAGE1_SCHEMA_VERSION
    ):
        return None

    if not has_parseable_stage1_source_response(response, output_key=output_key):
        return "stage 1 classification has no parseable source response"
    evaluation = evaluate_stage1_response(response, output_key=output_key)
    if not evaluation.accepted:
        return "stage 1 classification violates contract"
    return None


def _completion_checked_result(
    result: dict[str, Any], *, output_key: str
) -> dict[str, Any]:
    """Turn semantically incomplete nominal successes into retryable errors."""
    if result.get("status") != "success":
        return result
    response = result.get("vlm_response")
    if response is None:
        return {
            **result,
            "status": "error",
            "error": "success result has no non-null vlm_response",
        }
    completion_error = _stage1_completion_error(response, output_key=output_key)
    if completion_error is None:
        return result
    evaluation = evaluate_stage1_response(response, output_key=output_key)
    return {
        **result,
        "status": "error",
        "error": completion_error,
        "contract_reason": evaluation.reason,
        "contract_diagnostics": dict(evaluation.diagnostics),
    }


def _index_dataset_entries(
    dataset: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Return stable dataset order and reject ambiguous prediction identities."""
    ordered_ids: list[str] = []
    entries_by_id: dict[str, dict[str, Any]] = {}
    for index, entry in enumerate(dataset):
        if not isinstance(entry, dict):
            raise ValueError(f"dataset entry {index} must be a dictionary")
        entry_id = entry.get("id")
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise ValueError(f"dataset entry {index} must have a non-empty string id")
        if entry_id in entries_by_id:
            raise ValueError(f"dataset contains duplicate id: {entry_id}")
        ordered_ids.append(entry_id)
        entries_by_id[entry_id] = entry
    return ordered_ids, entries_by_id


def _index_batch_results(
    results: list[dict[str, Any]],
    *,
    expected_ids: set[str],
    phase: str,
    output_key: str,
) -> dict[str, dict[str, Any]]:
    """Validate batch result identity without hiding omitted entries."""
    indexed: dict[str, dict[str, Any]] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise RuntimeError(f"{phase} result {index} must be a dictionary")
        entry_id = result.get("id")
        if not isinstance(entry_id, str) or not entry_id:
            raise RuntimeError(f"{phase} result {index} has no valid id")
        if entry_id not in expected_ids:
            raise RuntimeError(f"{phase} returned unexpected id: {entry_id}")
        if entry_id in indexed:
            raise RuntimeError(f"{phase} returned duplicate id: {entry_id}")
        status = result.get("status")
        if status not in {"success", "error"}:
            raise RuntimeError(
                f"{phase} result {entry_id} has invalid status: {status!r}"
            )
        result = _completion_checked_result(result, output_key=output_key)
        indexed[entry_id] = result
    return indexed


def _json_safe_prediction_media(value: Any) -> Any:
    """Keep persistable media references without stringifying image objects."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        normalized = []
        for item in value:
            normalized_item = _json_safe_prediction_media(item)
            if normalized_item is not _OMIT_PREDICTION_MEDIA:
                normalized.append(normalized_item)
        if value and not normalized:
            return _OMIT_PREDICTION_MEDIA
        return normalized
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        filename = getattr(value, "filename", None)
        if isinstance(filename, str | Path) and str(filename):
            return str(filename)
        return _OMIT_PREDICTION_MEDIA
    return value


def _prediction_file_row(
    result: dict[str, Any],
    entry: dict[str, Any],
    *,
    output_key: str,
) -> dict[str, Any]:
    """Build the canonical streamed row for one successful prediction."""
    output_entry: dict[str, Any] = {
        "id": result["id"],
        output_key: result.get("vlm_response"),
    }
    if "images" in entry:
        images = _json_safe_prediction_media(entry["images"])
        if images is not _OMIT_PREDICTION_MEDIA:
            output_entry["images"] = images
    elif "image_path" in entry:
        image_path = _json_safe_prediction_media(entry.get("image_path", ""))
        if image_path is not _OMIT_PREDICTION_MEDIA:
            output_entry["image_path"] = image_path
    return output_entry


class VLMInferenceTask(Task):
    """Run VLM inference on dataset for asset classification.

    This task supports configurable output_key for flexible classification
    tasks (e.g., component identification, material prediction, property estimation).

    Input context keys:
        - dataset or dataset_path: Dataset to process
        - vlm: VLM instance
        - llm: LLM instance (optional, uses VLM if not provided)
        - vlm_config: VLM configuration
        - system_prompt: Base system prompt
        - output_key: Key for classification output (default: "classification")

    Output context keys:
        - predictions: List of prediction results
        - predictions_path: Path to saved predictions file
        - provider_response_diagnostics_path: Durable provider-attempt journal
        - provider_response_diagnostics_sha256: SHA-256 of that journal
        - provider_response_conformance_terminal_status: Typed fail-closed status
    """

    def __init__(
        self,
        vlm: Any = None,
        llm: Any | None = None,
        system_prompt: str | None = None,
        output_key: str | None = None,
    ):
        """Initialize the VLM inference task.

        Args:
            vlm: VLM instance (None to use from context)
            llm: Optional LLM for parsing (None to use from context or VLM)
            system_prompt: Optional custom system prompt (None to use from context)
            output_key: Key for classification output (None to use from context)
        """
        self.vlm = vlm
        self.llm = llm
        self.system_prompt = system_prompt
        self.output_key = output_key
        self.name = "VLMInference"
        self.description = "Run VLM inference for asset classification"

    def run(
        self, context: dict[str, Any], object_store: ObjectStore | None = None
    ) -> dict[str, Any]:
        """Run batch inference on dataset.

        Args:
            context: Workflow context with dataset metadata
            object_store: Storage for dataset and predictions

        Returns:
            Updated context with inference results
        """
        # Get event listener (or logger fallback)
        listener = get_listener(context, logger_name=__name__)

        # Resolve parameters from constructor or context
        vlm = self.vlm if self.vlm is not None else context.get("vlm")
        llm = self.llm if self.llm is not None else context.get("llm")
        if llm is None:
            llm = vlm  # Use VLM as LLM if not provided

        # Get config values from context if not provided in constructor
        vlm_config = context.get("vlm_config", {})
        max_retries = vlm_config.get("max_retries", 3)
        completion_retries = context.get(
            "completion_retries", _DEFAULT_COMPLETION_RETRIES
        )
        if (
            isinstance(completion_retries, bool)
            or not isinstance(completion_retries, int)
            or completion_retries < 0
        ):
            raise ValueError("completion_retries must be a non-negative integer")
        system_prompt = (
            self.system_prompt
            if self.system_prompt is not None
            else context.get("system_prompt")
            or context.get("config", {}).get("system_prompt")
        )

        # Get output_key (configurable)
        output_key = (
            self.output_key
            if self.output_key is not None
            else context.get("output_key", "classification")
        )

        # Build per-invoke kwargs from provisioning
        vlm_invoke_kwargs: dict[str, Any] = dict(context.get("vlm_invoke_kwargs", {}))

        # Validate required values
        if vlm is None:
            raise ValueError("VLM not provided in constructor or context")

        # Get dataset from object store or context
        if object_store and object_store.exists("dataset"):
            dataset = object_store.get("dataset")
        else:
            # Fallback: load from file if not in store
            dataset_path_str = context.get("dataset_path")
            if not dataset_path_str:
                raise ValueError(
                    "dataset_path not found in context and dataset not in object_store"
                )
            dataset_path = Path(dataset_path_str)
            with open(dataset_path, encoding="utf-8") as f:
                dataset = [json.loads(line) for line in f]

        ordered_dataset_ids, dataset_by_id = _index_dataset_entries(dataset)
        expected_dataset_ids = set(ordered_dataset_ids)

        # Emit task started event
        listener.event(
            "task.started",
            {
                "task_name": "VLMInference",
                "total_entries": len(dataset),
            },
        )
        listener.info(f"Starting VLM inference for {len(dataset)} entries")
        listener.info(f"Output key: {output_key}")

        # Progress callback
        processed_count = [0]

        def on_progress(entry_id: str, response: str) -> None:
            """Log progress after processing each entry."""
            if _stage1_completion_error(response, output_key=output_key) is not None:
                return
            processed_count[0] += 1
            listener.event(
                "task.progress",
                {
                    "task_name": "VLMInference",
                    "current": processed_count[0],
                    "total": len(dataset),
                    "percentage": (processed_count[0] / len(dataset)) * 100
                    if dataset
                    else 0,
                    "entry_id": entry_id,
                },
            )
            if processed_count[0] % 10 == 0:
                listener.info(f"Processed {processed_count[0]}/{len(dataset)} entries")

        # Error callback
        def on_error(entry_id: str, error: str) -> None:
            """Handle errors during processing."""
            logger.error(f"Error processing {entry_id}: {error}")
            listener.error(f"Error processing {entry_id}: {error}")

        # Prediction callback
        def on_prediction(entry_id: str, result_dict: dict[str, Any]) -> None:
            """Emit event for each prediction."""
            try:
                if (
                    _stage1_completion_error(result_dict, output_key=output_key)
                    is not None
                ):
                    return
                classification = result_dict.get(output_key)
                if classification is None:
                    classification = result_dict
                confidence = result_dict.get("confidence")

                listener.event(
                    "prediction.completed",
                    {
                        "entry_id": entry_id,
                        output_key: classification,
                        "confidence": confidence,
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to emit prediction event for {entry_id}: {e}")

        # Streaming and resume options
        stream_predictions = context.get("stream_predictions", True)
        resume_enabled = context.get("resume", False)

        # Resolve predictions path
        predictions_path = context.get("predictions_path")
        if predictions_path:
            predictions_path = Path(predictions_path)
            output_dir = predictions_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir_value = context.get("output_dir")
            if output_dir_value is None:
                dataset_path_str = context.get("dataset_path")
                if not dataset_path_str:
                    raise ValueError("dataset_path not found in context")
                output_dir_value = Path(dataset_path_str).parent / "output"
            output_dir = Path(output_dir_value)
            output_dir.mkdir(parents=True, exist_ok=True)
            predictions_path = output_dir / "predictions.jsonl"

        provider_response_diagnostics_path = (
            output_dir / "provider_response_attempts.json"
        )
        expected_diagnostics_sha256 = context.get(
            "provider_response_diagnostics_expected_sha256"
        )

        def bind_persisted_journal(path: Path, digest: str) -> None:
            project_provider_attempt_persistence(
                context,
                path=path,
                digest=digest,
                path_key="provider_response_diagnostics_path",
                digest_key="provider_response_diagnostics_sha256",
            )

        def bind_terminal(
            error: ProviderResponseConformanceTerminalError,
        ) -> None:
            context["provider_response_conformance_terminal_status"] = dict(
                error.status
            )

        if resume_enabled and (
            provider_response_diagnostics_path.is_file()
            or expected_diagnostics_sha256 is not None
        ):
            attempt_journal = load_provider_attempt_journal(
                provider_response_diagnostics_path,
                on_persisted=bind_persisted_journal,
                expected_sha256=expected_diagnostics_sha256,
                failure_stage="stage1_provider_evidence",
                on_terminal=bind_terminal,
            )
        else:
            attempt_journal = ProviderAttemptJournal(
                path=provider_response_diagnostics_path,
                on_persisted=bind_persisted_journal,
            )
        current_run_first_attempt_sequence = attempt_journal.attempt_count() + 1
        request_kind_by_entry: dict[str, ProviderRequestKind] = {}
        require_provider_attempt_journal_persistence(
            attempt_journal,
            failure_stage="stage1_provider_evidence",
            diagnostics_artifact_path=str(provider_response_diagnostics_path),
            on_terminal=bind_terminal,
        )

        def raise_stage1_terminal(
            contract_failures: list[dict[str, Any]],
            *,
            message: str,
            prior_persistence_error: BaseException | None = None,
        ) -> NoReturn:
            persistence = persist_provider_attempt_journal(
                attempt_journal,
                prior_error=prior_persistence_error,
            )
            terminal_error = ProviderResponseConformanceTerminalError.for_stage1(
                reason=str(contract_failures[0]["reason"]),
                attempt_diagnostics=contract_failures,
                diagnostics_artifact_path=(
                    str(provider_response_diagnostics_path)
                    if persistence.artifact_sha256 is not None
                    else None
                ),
                diagnostics_artifact_sha256=persistence.artifact_sha256,
                diagnostics_persistence_error=persistence.error,
                message=message,
            )
            bind_terminal(terminal_error)
            raise terminal_error

        def on_provider_attempt(
            entry_id: str,
            attempt: dict[str, Any],
        ) -> BaseException | None:
            def record_attempt() -> ProviderAttemptRecording:
                attempt_number = int(attempt.get("attempt_number", 1))
                return attempt_journal.record_transport_attempt(
                    attempt,
                    entry_id=entry_id,
                    initial_request_kind=(
                        "transport_retry"
                        if attempt_number > 1
                        else request_kind_by_entry.get(entry_id, "initial")
                    ),
                )

            recording = invoke_provider_attempt_recorder(record_attempt)
            if not isinstance(recording, ProviderAttemptRecording):
                raise ProviderAttemptRecorderError(
                    "provider attempt recorder returned an invalid receipt"
                )
            return cast(BaseException | None, recording.failure_carrier)

        def record_contract_outcomes(
            indexed_results: dict[str, dict[str, Any]],
            *,
            attempt_number: int = 1,
            minimum_sequence_number: int = current_run_first_attempt_sequence,
        ) -> None:
            diagnostics_to_record: list[
                tuple[str | None, ProviderAttemptDiagnostic]
            ] = []
            contract_failures: list[dict[str, Any]] = []
            for entry_id, result in indexed_results.items():
                contract_reason = result.get("contract_reason")
                if result.get("status") == "success":
                    outcome: ProviderAttemptOutcome = "accepted"
                    diagnostics: dict[str, Any] = {}
                elif contract_reason:
                    # A fallback parser can exhaust its own provider transport
                    # and return a legacy sentinel row. Preserve that exact
                    # transport outcome instead of appending a synthetic
                    # contract rejection that would overwrite terminal meaning.
                    if (
                        attempt_journal.latest_transport_attempt(
                            (entry_id,),
                            minimum_sequence_number=minimum_sequence_number,
                        )
                        is not None
                    ):
                        continue
                    outcome = "contract_rejected"
                    diagnostics = {
                        "reason": contract_reason,
                        **dict(result.get("contract_diagnostics") or {}),
                    }
                else:
                    continue
                diagnostics_to_record.append(
                    (
                        entry_id,
                        build_provider_attempt_diagnostic(
                            request_kind=request_kind_by_entry.get(entry_id, "initial"),
                            outcome=outcome,
                            attempt_number=attempt_number,
                            normalized_diagnostics=diagnostics,
                        ),
                    )
                )
                if outcome == "contract_rejected":
                    contract_failures.append(
                        {
                            "entry_id": entry_id,
                            "request_kind": request_kind_by_entry.get(
                                entry_id, "initial"
                            ),
                            "outcome": outcome,
                            "reason": contract_reason,
                            "normalized_diagnostics": diagnostics,
                        }
                    )
            try:
                attempt_journal.record_many(diagnostics_to_record)
            except Exception as persistence_error:
                if contract_failures:
                    raise_stage1_terminal(
                        contract_failures,
                        message=(
                            "Provider-backed Stage 1 contract rejection could not "
                            "be durably checkpointed"
                        ),
                        prior_persistence_error=persistence_error,
                    )
                require_provider_attempt_journal_persistence(
                    attempt_journal,
                    failure_stage="stage1_provider_evidence",
                    diagnostics_artifact_path=str(provider_response_diagnostics_path),
                    on_terminal=bind_terminal,
                    attempt_diagnostics=(
                        {
                            **({"entry_id": entry_id} if entry_id is not None else {}),
                            **diagnostic.to_dict(),
                        }
                        for entry_id, diagnostic in diagnostics_to_record
                    ),
                    prior_error=persistence_error,
                )

        # Clear existing predictions if not resuming
        if stream_predictions and not resume_enabled and predictions_path.exists():
            predictions_path.unlink()
            logger.info("Cleared existing predictions file")

        # Load processed ids when resuming
        processed_ids: set[str] = set()
        resumed_rows_by_id: dict[str, dict[str, Any]] = {}
        if stream_predictions and resume_enabled and predictions_path.exists():
            try:
                seen_resume_ids: set[str] = set()
                with open(predictions_path, encoding="utf-8") as f:
                    nonempty_lines = [
                        (line_number, line)
                        for line_number, line in enumerate(f, start=1)
                        if line.strip()
                    ]
                rewrite_resume_checkpoint = bool(
                    nonempty_lines and not nonempty_lines[-1][1].endswith(("\n", "\r"))
                )
                for row_index, (line_number, line) in enumerate(nonempty_lines):
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        if row_index != len(nonempty_lines) - 1 or line.endswith(
                            ("\n", "\r")
                        ):
                            raise
                        logger.warning(
                            "Ignoring torn final prediction row %s while resuming",
                            line_number,
                        )
                        continue
                    if not isinstance(rec, dict):
                        raise ValueError(
                            f"prediction row {line_number} must be a dictionary"
                        )
                    entry_id = rec.get("id")
                    if not isinstance(entry_id, str) or not entry_id:
                        raise ValueError(
                            f"prediction row {line_number} has no valid id"
                        )
                    if entry_id not in expected_dataset_ids:
                        raise ValueError(
                            f"prediction row {line_number} has unexpected id: "
                            f"{entry_id}"
                        )
                    if entry_id in seen_resume_ids:
                        raise ValueError(
                            f"prediction file contains duplicate id: {entry_id}"
                        )
                    seen_resume_ids.add(entry_id)
                    if output_key not in rec or rec[output_key] is None:
                        raise ValueError(
                            f"prediction row {line_number} is missing a non-null "
                            f"{output_key}"
                        )
                    completion_error = _stage1_completion_error(
                        rec[output_key],
                        output_key=output_key,
                    )
                    if completion_error is not None:
                        logger.warning(
                            "Ignoring semantically incomplete prediction row %s "
                            "while resuming: %s",
                            line_number,
                            completion_error,
                        )
                        rewrite_resume_checkpoint = True
                        continue
                    resumed_rows_by_id[entry_id] = rec
                    processed_ids.add(entry_id)
                if rewrite_resume_checkpoint:
                    write_predictions_jsonl(
                        predictions_path,
                        (
                            resumed_rows_by_id[entry_id]
                            for entry_id in ordered_dataset_ids
                            if entry_id in resumed_rows_by_id
                        ),
                    )
                logger.info(f"Resuming: {len(processed_ids)} entries already processed")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot resume from invalid predictions file: {e}"
                ) from e
        processed_count[0] = len(processed_ids)

        # Define result callback for streaming
        stream_write_lock = Lock()

        def on_result(result: dict[str, Any], entry: dict[str, Any]) -> None:
            if not stream_predictions:
                return
            try:
                result = _completion_checked_result(result, output_key=output_key)
                if (
                    result.get("status") != "success"
                    or result.get("vlm_response") is None
                ):
                    return
                output_entry = _prediction_file_row(
                    result,
                    entry,
                    output_key=output_key,
                )

                with stream_write_lock:
                    with open(predictions_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(output_entry) + "\n")
            except Exception as e:
                logger.warning(f"Failed to append streaming prediction: {e}")

        # Get optional max_workers
        max_workers = context.get("max_workers")

        # Create token tracker
        token_tracker = TokenTracker()

        def run_batch(
            entries: list[dict[str, Any]],
            *,
            already_processed: set[str],
            workers: int | None,
            request_kinds: dict[str, ProviderRequestKind],
        ) -> list[dict[str, Any]]:
            request_kind_by_entry.update(request_kinds)

            def invoke() -> list[dict[str, Any]]:
                return cast(
                    list[dict[str, Any]],
                    batch_classify_assets(
                        vlm=vlm,
                        entries=entries,
                        llm=cast(BaseChatModel, llm),
                        image_base_dir=Path(context["image_base_dir"])
                        if context.get("image_base_dir")
                        else None,
                        system_prompt=system_prompt,
                        invoke_kwargs=vlm_invoke_kwargs,
                        on_progress=on_progress,
                        on_error=on_error,
                        processed_ids=already_processed,
                        on_result=on_result,
                        on_prediction=on_prediction,
                        max_workers=workers,
                        max_retries=max_retries,
                        output_key=output_key,
                        token_tracker=token_tracker,
                        on_provider_attempt=on_provider_attempt,
                    ),
                )

            return cast(
                list[dict[str, Any]],
                run_provider_call_with_journal(
                    invoke,
                    journal=attempt_journal,
                    failure_stage="stage1_provider_transport",
                    diagnostics_artifact_path=str(provider_response_diagnostics_path),
                    on_terminal=bind_terminal,
                ),
            )

        # Run the normal batch, then retry only unresolved entries with one
        # worker. The lower-concurrency completion pass recovers isolated
        # provider failures without repeating successful model calls.
        initial_results = run_batch(
            dataset,
            already_processed=processed_ids,
            workers=max_workers,
            request_kinds=dict.fromkeys(
                expected_dataset_ids - processed_ids, "initial"
            ),
        )
        attempted_ids = expected_dataset_ids - processed_ids
        latest_results = _index_batch_results(
            initial_results,
            expected_ids=attempted_ids,
            phase="initial inference",
            output_key=output_key,
        )
        record_contract_outcomes(latest_results)
        successful_results: dict[str, dict[str, Any]] = {
            entry_id: {
                "id": entry_id,
                "vlm_response": row[output_key],
                "status": "success",
            }
            for entry_id, row in resumed_rows_by_id.items()
        }
        successful_results.update(
            {
                entry_id: result
                for entry_id, result in latest_results.items()
                if result["status"] == "success"
            }
        )

        terminal_attempt_floor = current_run_first_attempt_sequence
        for completion_attempt in range(1, completion_retries + 1):
            unresolved_ids = expected_dataset_ids - set(successful_results)
            if not unresolved_ids:
                break
            listener.info(
                f"Retrying {len(unresolved_ids)} unresolved prediction(s) in "
                f"completion pass {completion_attempt}/{completion_retries}"
            )
            retry_entries = [
                dataset_by_id[entry_id]
                for entry_id in ordered_dataset_ids
                if entry_id in unresolved_ids
            ]
            terminal_attempt_floor = attempt_journal.attempt_count() + 1
            retry_results = run_batch(
                retry_entries,
                already_processed=set(),
                workers=1,
                request_kinds={
                    entry_id: (
                        "contract_correction"
                        if latest_results.get(entry_id, {}).get("contract_reason")
                        else "transport_retry"
                    )
                    for entry_id in unresolved_ids
                },
            )
            indexed_retry_results = _index_batch_results(
                retry_results,
                expected_ids=unresolved_ids,
                phase=f"completion inference pass {completion_attempt}",
                output_key=output_key,
            )
            record_contract_outcomes(
                indexed_retry_results,
                attempt_number=completion_attempt + 1,
                minimum_sequence_number=terminal_attempt_floor,
            )
            latest_results.update(indexed_retry_results)
            successful_results.update(
                {
                    entry_id: result
                    for entry_id, result in indexed_retry_results.items()
                    if result["status"] == "success"
                }
            )

        # Log token usage
        token_stats = token_tracker.get_stats()
        logger.info(f"\n{format_token_stats(token_stats)}")

        unresolved_ids = expected_dataset_ids - set(successful_results)
        failed = [
            latest_results.get(
                entry_id,
                {
                    "id": entry_id,
                    "vlm_response": None,
                    "status": "error",
                    "error": "batch returned no result",
                },
            )
            for entry_id in ordered_dataset_ids
            if entry_id in unresolved_ids
        ]
        predictions = [
            successful_results[entry_id]
            for entry_id in ordered_dataset_ids
            if entry_id in successful_results
        ]

        listener.info(
            f"Inference complete: {len(predictions)} successful, {len(failed)} failed"
        )

        if unresolved_ids:
            unresolved_details = []
            contract_failures: list[dict[str, Any]] = []
            for entry_id in ordered_dataset_ids:
                if entry_id not in unresolved_ids:
                    continue
                result = latest_results.get(entry_id)
                if result is None:
                    failure_detail = "batch returned no result"
                    contract_reason = None
                    contract_diagnostics: dict[str, Any] = {}
                else:
                    failure_detail = str(result.get("error") or "unknown error")
                    contract_reason = result.get("contract_reason")
                    contract_diagnostics = dict(
                        result.get("contract_diagnostics") or {}
                    )
                unresolved_details.append(f"{entry_id}: {failure_detail}")
                has_current_transport_failure = (
                    attempt_journal.latest_transport_attempt(
                        (entry_id,),
                        minimum_sequence_number=terminal_attempt_floor,
                    )
                    is not None
                )
                if contract_reason and not has_current_transport_failure:
                    contract_failures.append(
                        {
                            "entry_id": entry_id,
                            "request_kind": request_kind_by_entry.get(
                                entry_id, "initial"
                            ),
                            "outcome": "contract_rejected",
                            "reason": contract_reason,
                            "normalized_diagnostics": contract_diagnostics,
                        }
                    )
            if object_store:
                object_store.set("predictions", predictions)
                object_store.set("failed_predictions", failed)
            if contract_failures:
                failure_message = (
                    "VLM inference incomplete after bounded recovery: expected "
                    f"{len(dataset)} unique predictions, got {len(predictions)}; "
                    "unresolved entries: " + "; ".join(unresolved_details)
                )
                raise_stage1_terminal(
                    contract_failures,
                    message=failure_message,
                )
            transport_terminal_error = evaluate_exhausted_transport_terminal(
                journal=attempt_journal,
                failure_stage="stage1_provider_transport",
                diagnostics_artifact_path=str(provider_response_diagnostics_path),
                entry_ids=(
                    entry_id
                    for entry_id in ordered_dataset_ids
                    if entry_id in unresolved_ids
                ),
                minimum_sequence_number=terminal_attempt_floor,
                on_terminal=bind_terminal,
            )
            if transport_terminal_error is not None:
                raise transport_terminal_error
            raise RuntimeError(
                "VLM inference incomplete after bounded recovery: expected "
                f"{len(dataset)} unique predictions, got {len(predictions)}; "
                "unresolved entries: " + "; ".join(unresolved_details)
            )

        # Seal successful streaming output into deterministic dataset order.
        # This removes transient append order and guarantees exactly one row
        # for each dataset entry before downstream tasks can start.
        if stream_predictions:
            canonical_rows = [
                _prediction_file_row(
                    successful_results[entry_id],
                    dataset_by_id[entry_id],
                    output_key=output_key,
                )
                for entry_id in ordered_dataset_ids
            ]
            write_predictions_jsonl(predictions_path, canonical_rows)

        # Store in object store
        if object_store:
            object_store.set("predictions", predictions)
            object_store.set("failed_predictions", failed)

        # Update context
        context["predictions_count"] = len(predictions)
        context["failed_count"] = len(failed)
        context["inference_complete"] = True
        context["predictions_path"] = str(predictions_path)
        context["token_stats"] = token_stats
        context["output_key"] = output_key
        context["provider_response_diagnostics_path"] = str(
            provider_response_diagnostics_path
        )
        final_persistence = require_provider_attempt_journal_persistence(
            attempt_journal,
            failure_stage="stage1_provider_evidence",
            diagnostics_artifact_path=str(provider_response_diagnostics_path),
            on_terminal=bind_terminal,
        )
        context["provider_response_diagnostics_sha256"] = (
            final_persistence.artifact_sha256
        )

        return context
