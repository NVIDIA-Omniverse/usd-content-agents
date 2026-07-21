# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task for optimizing USD files via REST API."""

import json
import logging
import math
import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, NoReturn

from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task
from world_understanding.config.s3 import WU_S3_BUCKET, WU_S3_PROFILE, WU_S3_REGION
from world_understanding.functions.graphics.scene_optimizer_nvcf import (
    optimize_usd_from_path,
)
from world_understanding.utils.credentials import (
    find_inline_secret_paths,
    redact_sensitive_config,
)
from world_understanding.utils.object_store import ObjectStore

logger = logging.getLogger(__name__)

_OPTIMIZATION_FAILURE_MESSAGE = "USD optimization failed"
_LOCAL_BACKEND_UNAVAILABLE_MESSAGE = (
    "Scene optimization failed: local backend unavailable and no remote "
    "backend is configured. Fix one of: (a) run "
    "`./scripts/fetch_build_resources.sh` to fetch the public Scene Optimizer "
    "Core package, (b) set NVCF_OPTIMIZER_FUNCTION_ID (or "
    "OPTIMIZER_ENDPOINT) for the remote backend, or (c) set "
    "`optimize_usd.enabled: false` in your config."
)
_KNOWN_OPTIMIZER_OPERATIONS = frozenset(
    {
        "deduplicate",
        "deduplicateGeometry",
        "deinstance",
        "split",
        "splitMeshes",
        "utilityFunction",
    }
)
_SUMMARY_COUNT_FIELDS = (
    "total_original_prims",
    "meshes_before_deinstance",
    "meshes_after_deinstance",
    "meshes_deinstanced",
    "meshes_split",
    "instances_tracked",
)
_SUMMARY_OPERATION_FIELDS = ("deinstance", "split", "deduplicate")


class _SafeOptimizationError(RuntimeError):
    """An operator-facing failure whose message contains no backend payload."""


class _SafeOptimizationInputError(ValueError):
    """A value-free public validation failure."""


def _raise_safe_optimization_error(message: str) -> NoReturn:
    """Raise a value-free failure from a frame with no runtime inputs."""
    raise _SafeOptimizationError(message) from None


def _raise_safe_optimization_input_error(message: str) -> NoReturn:
    """Raise value-free validation from a frame with no runtime inputs."""
    raise _SafeOptimizationInputError(message) from None


def _is_local_backend_unavailable(error: RuntimeError | FileNotFoundError) -> bool:
    """Classify a local setup failure without retaining its diagnostic text."""
    if isinstance(error, FileNotFoundError):
        return True
    try:
        message = str(error)
    except Exception:
        return False
    return any(
        marker in message
        for marker in (
            "WU_SO_PACKAGE_DIR",
            "Scene Optimizer package directory missing",
            "Scene Optimizer Core package not found",
            "Scene Optimizer subprocess failed",
        )
    )


class _UnsupportedDurableConfigError(ValueError):
    """A value-free failure for malformed durable optimizer configuration."""


_OMIT_JSON_VALUE = object()
_UNSUPPORTED_DURABLE_CONFIG_MESSAGE = "Unsupported durable optimizer configuration"
_UNSUPPORTED_CONFIG_LOG_VALUE = "<unsupported>"


def _json_value_sort_key(value: Any) -> tuple[int, str]:
    """Return a stable ordering key for an already projected JSON value."""
    type_rank = {
        type(None): 0,
        bool: 1,
        int: 2,
        float: 3,
        str: 4,
        list: 5,
        dict: 6,
    }
    return (
        type_rank[type(value)],
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    )


def _project_json_value(
    value: Any,
    *,
    active_container_ids: set[int] | None = None,
) -> Any:
    """Project config data to JSON primitives or return an omission sentinel.

    Direct Python callers can attach clients, locks, paths, or other live objects
    to optimizer configuration. Those runtime-only leaves are deliberately
    omitted instead of stringified. An ordered container containing an omitted
    item is omitted as a whole so durable list positions never shift. Sets and
    frozensets accept heterogeneous JSON-safe members and use a canonical JSON
    type/value ordering instead of Python's cross-type comparison.

    Malformed config-shaped values fail with one value-free diagnostic. This
    distinguishes invalid durable input from the explicitly supported exclusion
    of opaque runtime objects.
    """
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise _UnsupportedDurableConfigError(
                _UNSUPPORTED_DURABLE_CONFIG_MESSAGE
            ) from None
        return value
    if type(value) in {bytes, bytearray, complex, date, datetime}:
        raise _UnsupportedDurableConfigError(
            _UNSUPPORTED_DURABLE_CONFIG_MESSAGE
        ) from None

    container_type = type(value)
    if container_type not in {dict, list, tuple, set, frozenset}:
        return _OMIT_JSON_VALUE

    active_ids = active_container_ids if active_container_ids is not None else set()
    container_id = id(value)
    if container_id in active_ids:
        raise _UnsupportedDurableConfigError(
            _UNSUPPORTED_DURABLE_CONFIG_MESSAGE
        ) from None
    active_ids.add(container_id)
    try:
        if container_type is dict:
            projected_mapping: dict[str, Any] = {}
            for key, child in value.items():
                if type(key) is not str:
                    raise _UnsupportedDurableConfigError(
                        _UNSUPPORTED_DURABLE_CONFIG_MESSAGE
                    ) from None
                projected_child = _project_json_value(
                    child,
                    active_container_ids=active_ids,
                )
                if projected_child is not _OMIT_JSON_VALUE:
                    projected_mapping[key] = projected_child
            return projected_mapping

        projected_items = []
        for child in value:
            projected_child = _project_json_value(
                child,
                active_container_ids=active_ids,
            )
            if projected_child is _OMIT_JSON_VALUE:
                return _OMIT_JSON_VALUE
            projected_items.append(projected_child)
        if container_type in {set, frozenset}:
            projected_items.sort(key=_json_value_sort_key)
        return projected_items
    finally:
        active_ids.remove(container_id)


def _project_durable_optimization_config(value: Any) -> dict[str, Any]:
    """Return the strict durable schema for redacted optimizer configuration."""
    projected = _project_json_value(value)
    return projected if type(projected) is dict else {}


def _project_optimizer_config_for_diagnostics(
    value: Any,
) -> dict[str, Any] | None:
    """Return a repr-safe config projection or a constant-summary signal."""
    try:
        return _project_durable_optimization_config(value)
    except _UnsupportedDurableConfigError:
        return None


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Write one fsynced JSON document and atomically replace ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _redacted_log_projection(value: Any) -> tuple[Any, bool]:
    """Return a logging-safe value and whether its source held a credential."""
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    return redact_sensitive_config(value), bool(find_inline_secret_paths(value))


def _redacted_derived_path(
    value: os.PathLike[str], *, source_redaction: Any | None
) -> Any:
    """Keep derived path diagnostics redacted when their source was sensitive."""
    if source_redaction is not None:
        return source_redaction
    return _redacted_log_projection(value)[0]


def _nonnegative_float(value: Any) -> float | None:
    """Project an untrusted value to a finite, non-negative float."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized < 0:
        return None
    return normalized


def _nonnegative_int(value: Any) -> int | None:
    """Project an untrusted value to a non-negative integer."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        normalized = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    if not math.isfinite(normalized) or normalized < 0 or not normalized.is_integer():
        return None
    return int(normalized)


def _safe_prim_path(value: Any) -> str | None:
    """Return a conservative USD prim path or ``None`` for backend text."""
    if not isinstance(value, str):
        return None

    from pxr import Sdf

    validation = Sdf.Path.IsValidPathString(value)
    is_valid = validation[0] if isinstance(validation, tuple) else validation
    if not is_valid:
        return None
    path = Sdf.Path(value)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path.IsAbsoluteRootPath():
        return None
    return value


def _prim_exists(stage: Any, path: str) -> bool:
    """Return whether an independently opened stage owns ``path``."""
    if stage is None:
        return False
    try:
        return bool(stage.GetPrimAtPath(path))
    except Exception:  # pragma: no cover - defensive adapter boundary
        return False


def _project_path_list_mapping(
    value: Any,
    *,
    source_stage: Any,
    target_stage: Any,
) -> dict[str, list[str]]:
    """Retain stage-owned prim-path to prim-path-list mappings."""
    if not isinstance(value, dict):
        return {}

    projected: dict[str, list[str]] = {}
    for raw_source, raw_targets in value.items():
        source = _safe_prim_path(raw_source)
        if source is None or not _prim_exists(source_stage, source):
            continue
        if isinstance(raw_targets, str):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list) or not raw_targets:
            continue
        targets = [_safe_prim_path(target) for target in raw_targets]
        if any(
            target is None or not _prim_exists(target_stage, target)
            for target in targets
        ):
            continue
        projected[source] = [target for target in targets if target is not None]
    return projected


def _project_path_mapping(
    value: Any,
    *,
    source_stage: Any,
    target_stage: Any,
) -> dict[str, str]:
    """Retain only stage-owned prim-path to prim-path mappings."""
    if not isinstance(value, dict):
        return {}

    projected: dict[str, str] = {}
    for raw_source, raw_target in value.items():
        source = _safe_prim_path(raw_source)
        target = _safe_prim_path(raw_target)
        if (
            source is not None
            and target is not None
            and _prim_exists(source_stage, source)
            and _prim_exists(target_stage, target)
        ):
            projected[source] = target
    return projected


def _project_correspondence_map(
    value: Any,
    *,
    original_stage: Any,
    optimized_stage: Any,
) -> dict[str, Any]:
    """Project an untrusted correspondence map to the restore schema.

    Only the fields consumed by ``RestoreUSDTask`` and benchmark summaries
    survive. Path fields must also resolve on independently opened input/output
    stages, so backend-controlled text cannot become durable merely by being a
    syntactically valid USD path.
    """
    if not isinstance(value, dict):
        return {}

    projected: dict[str, Any] = {}

    raw_summary = value.get("summary")
    if isinstance(raw_summary, dict):
        summary: dict[str, Any] = {}
        raw_operations = raw_summary.get("operations_run")
        if isinstance(raw_operations, dict):
            operations = {
                name: raw_operations[name]
                for name in _SUMMARY_OPERATION_FIELDS
                if isinstance(raw_operations.get(name), bool)
            }
            if operations:
                summary["operations_run"] = operations
        for field in _SUMMARY_COUNT_FIELDS:
            normalized = _nonnegative_int(raw_summary.get(field))
            if normalized is not None:
                summary[field] = normalized
        if summary:
            projected["summary"] = summary

    split_mapping = _project_path_list_mapping(
        value.get("split_mapping"),
        source_stage=original_stage,
        target_stage=optimized_stage,
    )
    if split_mapping:
        projected["split_mapping"] = split_mapping

    raw_deduplication = value.get("deduplication_mapping")
    if isinstance(raw_deduplication, dict):
        instance_mapping = _project_path_mapping(
            raw_deduplication.get("instance_to_prototype"),
            source_stage=optimized_stage,
            target_stage=optimized_stage,
        )
        if instance_mapping:
            projected["deduplication_mapping"] = {
                "instance_to_prototype": instance_mapping
            }

    raw_full_mapping = value.get("full_mapping")
    if isinstance(raw_full_mapping, dict):
        original_mapping = _project_path_list_mapping(
            raw_full_mapping.get("original_to_prototype"),
            source_stage=original_stage,
            target_stage=optimized_stage,
        )
        if original_mapping:
            projected["full_mapping"] = {"original_to_prototype": original_mapping}

    return projected


def _project_operations(value: Any) -> list[str | dict[str, Any]]:
    """Retain known optimizer operations with typed, value-free fields."""
    if not isinstance(value, list):
        return []

    projected: list[str | dict[str, Any]] = []
    for operation in value:
        if isinstance(operation, str):
            if operation in _KNOWN_OPTIMIZER_OPERATIONS:
                projected.append(operation)
            continue
        if not isinstance(operation, dict):
            continue
        name = operation.get("name")
        if not isinstance(name, str) or name not in _KNOWN_OPTIMIZER_OPERATIONS:
            continue
        safe_operation: dict[str, Any] = {"name": name}
        if isinstance(operation.get("success"), bool):
            safe_operation["success"] = operation["success"]
        elapsed = _nonnegative_float(operation.get("time"))
        if elapsed is not None:
            safe_operation["time"] = elapsed
        projected.append(safe_operation)
    return projected


def _project_backend_metadata(
    result: dict[str, Any],
    *,
    original_stage: Any,
    optimized_stage: Any,
) -> dict[str, Any]:
    """Return the strict durable schema for an untrusted success result."""
    return {
        "optimization_time": _nonnegative_float(result.get("optimization_time")),
        "stage_size_bytes": _nonnegative_int(result.get("stage_size_bytes")),
        "operations_executed": _project_operations(result.get("operations_executed")),
        "correspondence_map": _project_correspondence_map(
            result.get("correspondence_map"),
            original_stage=original_stage,
            optimized_stage=optimized_stage,
        ),
    }


def _restore_optimized_stage_metadata(
    source_stage: Any, output_usd: Path
) -> tuple[dict[str, Any], Any]:
    """Restore source stage metrics that Scene Optimizer does not preserve."""

    if not output_usd.is_file():
        raise _SafeOptimizationError(_OPTIMIZATION_FAILURE_MESSAGE) from None

    from pxr import Usd, UsdGeom

    optimized_stage = Usd.Stage.Open(str(output_usd))
    if optimized_stage is None:
        raise _SafeOptimizationError(_OPTIMIZATION_FAILURE_MESSAGE) from None

    source_root = source_stage.GetRootLayer()
    optimized_root = optimized_stage.GetRootLayer()
    up_axis = UsdGeom.GetStageUpAxis(source_stage)
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(optimized_stage, up_axis)
    UsdGeom.SetStageMetersPerUnit(optimized_stage, meters_per_unit)

    custom_layer_data = dict(optimized_root.customLayerData or {})
    custom_layer_data.update(dict(source_root.customLayerData or {}))
    optimized_root.customLayerData = custom_layer_data

    source_default = source_stage.GetDefaultPrim()
    if source_default:
        optimized_default = optimized_stage.GetPrimAtPath(source_default.GetPath())
        if optimized_default:
            optimized_stage.SetDefaultPrim(optimized_default)

    if source_root.pseudoRoot.HasInfo("kilogramsPerUnit"):
        optimized_root.pseudoRoot.SetInfo(
            "kilogramsPerUnit",
            source_root.pseudoRoot.GetInfo("kilogramsPerUnit"),
        )
    optimized_root.Save()
    return (
        {
            "restored": True,
            "up_axis": str(up_axis),
            "meters_per_unit": meters_per_unit,
            "custom_layer_data_keys": sorted(custom_layer_data),
        },
        optimized_stage,
    )


class OptimizeUSDTask(Task):
    """Task to optimize USD file via REST API.

    This task calls an optimization REST API with the input USD and produces
    an optimized USD file along with metadata about the optimization.

    Input context keys:
        - input_usd_path: Path to the USD file to optimize
        - output_usd_path: Path where optimized USD will be saved
        - optimization_config: Optional dict with API-specific parameters:
            - scene_optimizer_settings: Dict with operation settings:
                - enable_deinstance: bool (default True)
                - enable_split_meshes: bool (default True)
                - enable_deduplicate: bool (default True)
                - deinstance: Dict with deinstance settings
                - split_meshes: Dict with split settings
                - deduplicate: Dict with deduplicate settings
                - generate_report, capture_stats, verbose, etc.
            - flatten_prototypes: bool (default True) - Fully flatten the USD stage
                before optimization. This converts abstract prototypes (over/class)
                to def, inlines all referenced geometry, removes prototype prims,
                and preserves stage metadata (upAxis, metersPerUnit) and shader
                connections.
            - poll_seconds: Optional int for NVCF polling timeout
            - api_key, base_url, s3_bucket, s3_region, s3_profile, timeout

    Output context keys:
        - optimized_usd_path: Path to the optimized USD file
        - optimization_metadata: Dict with optimization statistics/info
        - optimization_success: Boolean indicating success
        - original_usd_path: Path to the original (pre-optimization) USD file
    """

    def run(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Execute USD optimization synchronously.

        This is a wrapper that calls the async implementation.

        Args:
            context: Workflow context with input parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with optimization results

        Raises:
            ValueError: If required parameters are missing
            Exception: If optimization API call fails
        """
        import asyncio

        safe_failure_message: str | None = None
        input_failure_message: str | None = None
        try:
            return asyncio.run(self.arun(context, object_store))
        except _SafeOptimizationInputError as error:
            input_failure_message = str(error)
        except _SafeOptimizationError as error:
            safe_failure_message = str(error)

        del context, object_store, self
        if input_failure_message is not None:
            _raise_safe_optimization_input_error(input_failure_message)
        assert safe_failure_message is not None
        _raise_safe_optimization_error(safe_failure_message)

    async def arun(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Execute optimization while quarantining rejected exception graphs."""
        input_failure_message: str | None = None
        if not context.get("input_usd_path"):
            input_failure_message = "input_usd_path is required in context"
        elif not context.get("output_usd_path"):
            input_failure_message = "output_usd_path is required in context"
        if input_failure_message is not None:
            del context, object_store, self
            _raise_safe_optimization_input_error(input_failure_message)

        safe_failure_message: str | None = None
        try:
            return await self._arun_impl(context, object_store)
        except _SafeOptimizationError as error:
            # Every _SafeOptimizationError is constructed from a code-owned
            # message. Its traceback can still retain the implementation frame,
            # so discard that graph and publish a fresh detached exception.
            safe_failure_message = str(error)
        except Exception:
            safe_failure_message = _OPTIMIZATION_FAILURE_MESSAGE

        assert safe_failure_message is not None
        listener = get_listener(context)
        listener.error(safe_failure_message)
        context["optimization_success"] = False
        context["optimization_error"] = safe_failure_message
        del context, listener, object_store, self
        _raise_safe_optimization_error(safe_failure_message)

    async def _arun_impl(
        self,
        context: dict[str, Any],
        object_store: ObjectStore | None = None,
    ) -> dict[str, Any]:
        """Execute USD optimization asynchronously.

        This overrides the base Task.arun() to provide true async execution
        instead of running sync code in a thread pool.

        Args:
            context: Workflow context with input parameters
            object_store: Optional object store (not used)

        Returns:
            Updated context with optimization results

        Raises:
            ValueError: If required parameters are missing
            Exception: If optimization API call fails
        """
        listener = get_listener(context)

        # Get input parameters
        input_usd = context.get("input_usd_path")
        output_usd = context.get("output_usd_path")
        optimization_config = context.get("optimization_config", {})
        safe_optimization_config = redact_sensitive_config(optimization_config)

        if not input_usd:
            raise ValueError("input_usd_path is required in context")
        if not output_usd:
            raise ValueError("output_usd_path is required in context")

        safe_input_usd_log, input_path_is_sensitive = _redacted_log_projection(
            input_usd
        )
        safe_output_usd_log, output_path_is_sensitive = _redacted_log_projection(
            output_usd
        )
        derived_path_redaction = (
            safe_input_usd_log
            if input_path_is_sensitive
            else safe_output_usd_log
            if output_path_is_sensitive
            else None
        )
        input_usd = Path(input_usd)
        output_usd = Path(output_usd)

        listener.info(f"Optimizing USD: {safe_input_usd_log}")
        listener.info(f"Output will be saved to: {safe_output_usd_log}")

        if optimization_config:
            if "scene_optimizer_settings" in optimization_config:
                listener.info("Using scene optimizer settings:")
                safe_settings = (
                    safe_optimization_config.get("scene_optimizer_settings")
                    if isinstance(safe_optimization_config, dict)
                    else None
                )
                if isinstance(safe_settings, dict):
                    diagnostic_settings = _project_optimizer_config_for_diagnostics(
                        safe_settings
                    )
                else:
                    diagnostic_settings = None
                if diagnostic_settings is not None:
                    # Build every diagnostic from the redacted projection. The
                    # original mapping remains available only to the backend.
                    enabled_ops = self._get_enabled_operations(diagnostic_settings)
                    listener.info(f"  Operations: {' -> '.join(enabled_ops)}")
                    listener.info(
                        "  Generate report: "
                        f"{diagnostic_settings.get('generate_report', True)}"
                    )
                    listener.info(
                        "  Capture stats: "
                        f"{diagnostic_settings.get('capture_stats', True)}"
                    )
                    listener.info(
                        f"  Verbose: {diagnostic_settings.get('verbose', False)}"
                    )
                    listener.info(
                        "  Wait for assets: "
                        f"{diagnostic_settings.get('wait_for_assets', False)}"
                    )
                    listener.info(
                        "  Stage timeout: "
                        f"{diagnostic_settings.get('stage_timeout', 180.0)}s"
                    )
                    listener.info(
                        "  Extract geom subset indices: "
                        f"{diagnostic_settings.get('extract_geom_subset_indices', True)}"
                    )
                else:
                    listener.info(f"  Settings: {_UNSUPPORTED_CONFIG_LOG_VALUE}")
            else:
                diagnostic_config = _project_optimizer_config_for_diagnostics(
                    safe_optimization_config
                )
                listener.info(
                    "Optimization config: "
                    f"{diagnostic_config if diagnostic_config is not None else _UNSUPPORTED_CONFIG_LOG_VALUE}"
                )

        safe_failure_message: str | None = None
        try:
            # Flatten prototypes BEFORE optimization
            # This converts over/class to def, resolves all references, and removes prototypes
            # Default is True since optimize_usd is typically used with pre-flattened scenes
            flatten_prototypes = optimization_config.get("flatten_prototypes", True)
            if not isinstance(flatten_prototypes, bool):
                raise _SafeOptimizationError(
                    "Invalid optimization configuration"
                ) from None

            # Track if we need to use a flattened input file
            actual_input = input_usd
            temp_flattened_input = None
            pre_converted_count = 0

            # Count original prims BEFORE any optimization
            from pxr import Usd, UsdGeom

            original_stage = Usd.Stage.Open(str(input_usd))
            original_prim_count = len(
                [p for p in original_stage.Traverse() if p.IsA(UsdGeom.Mesh)]
            )
            listener.info(
                f"Original prim count (before optimization): {original_prim_count}"
            )

            if flatten_prototypes:
                from world_understanding.utils.usd.prim import (
                    convert_abstract_prototypes_to_def,
                    flatten_prototype_references,
                )

                listener.info("Flattening prototypes before optimization...")
                listener.info("  - Converting abstract prototypes (over/class) to def")
                listener.info("  - Resolving all references (inlining geometry)")
                listener.info("  - Removing prototype prims")

                stage = original_stage  # Reuse the already opened stage

                # Step 1: Convert over/class to def (so they become traversable)
                converted_count = convert_abstract_prototypes_to_def(stage)
                if converted_count > 0:
                    listener.info(
                        f"  Converted {converted_count} abstract prototype(s) to def"
                    )

                # Step 2: Flatten - resolve references and remove prototypes
                flattened_layer = flatten_prototype_references(stage)

                # Save to the output workspace, not beside the source file. Source
                # stages are often mounted read-only in service containers.
                output_usd.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    prefix=f"_flattened_{input_usd.stem}_",
                    suffix=".usd",
                    dir=output_usd.parent,
                    delete=False,
                ) as temp_file:
                    temp_flattened_input = Path(temp_file.name)
                flattened_layer.Export(str(temp_flattened_input))
                actual_input = temp_flattened_input
                pre_converted_count = converted_count

                listener.info(
                    "  Flattened USD saved to: "
                    f"{_redacted_derived_path(temp_flattened_input, source_redaction=derived_path_redaction)}"
                )
            else:
                listener.info(
                    "Skipping prototype flattening (flatten_prototypes=False)"
                )

            # Determine backend: "local" (default) or "remote"
            backend = optimization_config.get("backend", "local")

            async def _run_nvcf() -> dict[str, Any]:
                """Run NVCF cloud backend."""
                listener.info("Calling NVCF optimization API...")
                return await optimize_usd_from_path(
                    input_path=actual_input,
                    output_path=output_usd,
                    api_key=optimization_config.get("api_key"),
                    base_url=optimization_config.get("base_url"),
                    s3_bucket=optimization_config.get("s3_bucket", WU_S3_BUCKET),
                    s3_region=optimization_config.get("s3_region", WU_S3_REGION),
                    s3_profile=optimization_config.get("s3_profile", WU_S3_PROFILE),
                    timeout=optimization_config.get("timeout", 3600),
                    max_retries=optimization_config.get("max_retries", 3),
                    optimization_config=optimization_config,
                )

            try:
                if backend == "local":
                    import asyncio

                    local_backend_unavailable: bool | None = None
                    try:
                        from world_understanding.functions.graphics.scene_optimizer_local import (
                            optimize_usd_local,
                        )

                        listener.info("Running local Scene Optimizer backend...")
                        result = await asyncio.to_thread(
                            optimize_usd_local,
                            input_path=actual_input,
                            output_path=output_usd,
                            optimization_config=optimization_config,
                        )
                    except (RuntimeError, FileNotFoundError) as local_error:
                        # Auto-fallback to NVCF if local backend is unavailable.
                        # Covers: macOS (.so missing → RuntimeError), and
                        # environments where WU_SO_PYTHON binary doesn't exist
                        # (e.g. Python 3.13 distroless image has no python3.12 →
                        # subprocess.run raises FileNotFoundError).
                        local_backend_unavailable = _is_local_backend_unavailable(
                            local_error
                        )

                    # Leave the rejected exception handler before either
                    # publishing a replacement error or invoking the fallback.
                    # Otherwise Python retains the local backend exception in
                    # the replacement/fallback exception's ``__context__``.
                    if local_backend_unavailable is not None:
                        if not local_backend_unavailable:
                            raise _SafeOptimizationError(_OPTIMIZATION_FAILURE_MESSAGE)
                        if not (
                            os.getenv("NVCF_OPTIMIZER_FUNCTION_ID")
                            or os.getenv("OPTIMIZER_ENDPOINT")
                        ):
                            raise _SafeOptimizationError(
                                _LOCAL_BACKEND_UNAVAILABLE_MESSAGE
                            )
                        listener.warning(
                            "Local SO backend unavailable; falling back to NVCF"
                        )
                        result = await _run_nvcf()
                elif backend == "remote":
                    result = await _run_nvcf()
                else:
                    raise _SafeOptimizationError(
                        "Invalid optimization backend; expected 'local' or 'remote'"
                    )
            finally:
                # Clean up temp file if created (even on failure)
                if (
                    temp_flattened_input
                    and temp_flattened_input.exists()
                    and temp_flattened_input.resolve() != output_usd.resolve()
                ):
                    try:
                        temp_flattened_input.unlink()
                        listener.debug(
                            "Cleaned up temp flattened input: "
                            f"{_redacted_derived_path(temp_flattened_input, source_redaction=derived_path_redaction)}"
                        )
                    except FileNotFoundError:
                        pass

            if result.get("status") != "success":
                # Backend diagnostics are untrusted and may reflect request
                # credentials or credentialed URLs. Keep the public/logged
                # failure constant; backend-specific diagnostics belong in a
                # secret-safe backend telemetry channel.
                raise _SafeOptimizationError(_OPTIMIZATION_FAILURE_MESSAGE)

            preserved_stage_metadata, optimized_stage = (
                _restore_optimized_stage_metadata(original_stage, output_usd)
            )

            # Treat the backend response as untrusted. Only the documented,
            # typed result schema may cross into workflow context, logs, or
            # the durable metadata sidecar. Free-form reports are deliberately
            # omitted because they can reflect request credentials.
            metadata = {
                **_project_backend_metadata(
                    result,
                    original_stage=original_stage,
                    optimized_stage=optimized_stage,
                ),
                "prototypes_converted_pre": pre_converted_count,
                "original_prim_count": original_prim_count,
                "preserved_stage_metadata": preserved_stage_metadata,
            }

            # Keep non-sensitive reproducibility settings without turning the
            # metadata sidecar into a credential artifact. Direct-Python runtime
            # objects are omitted instead of being serialized through repr/str.
            metadata["optimization_config"] = _project_durable_optimization_config(
                safe_optimization_config
            )

            metadata_path = output_usd.with_suffix(".metadata.json")
            _write_json_atomic(metadata_path, metadata)

            # Publish workflow success only after the backend output has opened
            # as a USD stage and the complete metadata sidecar is durable.
            context["optimized_usd_path"] = str(output_usd)
            context["optimization_metadata"] = metadata
            context["optimization_success"] = True
            # Save original path for restore_usd step
            context["original_usd_path"] = str(input_usd)
            # Save original prim count for stats reporting
            context["original_prim_count"] = original_prim_count

            listener.info("✓ USD optimization completed")
            listener.info(f"Optimized USD saved to: {safe_output_usd_log}")
            listener.info(
                "Saved metadata to: "
                f"{_redacted_derived_path(metadata_path, source_redaction=derived_path_redaction)}"
            )

        except _SafeOptimizationError as error:
            safe_failure_message = str(error)
        except Exception:
            # Do not propagate arbitrary exception text into logs, workflow
            # context, API responses, or later persisted service metadata.
            safe_failure_message = _OPTIMIZATION_FAILURE_MESSAGE

        if safe_failure_message is not None:
            # Publish and raise only after the rejected exception handler has
            # exited. ``raise ... from None`` inside that handler suppresses
            # display but still leaves the backend error on ``__context__``.
            listener.error(safe_failure_message)
            context["optimization_success"] = False
            context["optimization_error"] = safe_failure_message
            raise _SafeOptimizationError(safe_failure_message) from None

        return context

    def _get_enabled_operations(self, settings: dict[str, Any]) -> list[str]:
        """Build list of enabled operations.

        Matches client_scene_optimizer.py lines 744-750.

        Args:
            settings: Scene optimizer settings dict with snake_case keys
                (enable_deinstance, enable_split_meshes, enable_deduplicate)

        Returns:
            List of enabled operation names
        """
        enabled_ops = []
        if settings.get("enable_deinstance", True):
            enabled_ops.append("deinstance")
        if settings.get("enable_split_meshes", True):
            enabled_ops.append("split")
        if settings.get("enable_deduplicate", True):
            enabled_ops.append("deduplicate")
        return enabled_ops
