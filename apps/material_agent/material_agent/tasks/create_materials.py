# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author run-local materials from explicit fixed-workflow requests."""

from __future__ import annotations

import json
import os
import shlex
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml
from pxr import Sdf
from world_understanding.agentic.events import get_listener
from world_understanding.agentic.tasks import Task

from material_agent.material_library_generation.authoring import (
    MaterialAuthoringRequest,
    MaterialPackage,
    MaterialPackageAuthoringError,
    SourceMaterialReference,
    author_material_package,
)
from material_agent.material_library_generation.conditioning import (
    MaterialConditioningOptions,
    prepare_material_conditioning,
)
from material_agent.material_library_generation.creation import (
    MaterialCreationBackendRegistry,
    create_material_package,
)
from material_agent.material_library_generation.creation_contract import (
    CreatedMaterial,
    CreateMaterialRequest,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    PreparedMaterialConditioning,
)
from material_agent.material_library_generation.fake_backend import (
    FakeMaterialBackendBehavior,
    FakeMaterialCreationBackend,
)
from material_agent.material_library_generation.schema import (
    MaterialRecipe,
    TextureMapSet,
)
from material_agent.tasks.apply_materials_to_usd import (
    clear_color_space_on_empty_asset_inputs,
    remap_asset_paths_in_prim,
)

_FAKE_BACKEND_ALIASES = {"fake", "auto", "auto-for-test"}
_STEP1X_BACKEND_NAME = "step1x_material_anything"
_CANCEL_MONITOR_INTERVAL_SECONDS = 0.05
_BACKEND_UNSET = object()


class AuthorMaterialsTask(Task):
    """Create or modify run-local materials and register them for assignment."""

    def __init__(self) -> None:
        self.name = "AuthorMaterials"
        self.description = "Author run-local materials through shared package contracts"

    def run(
        self, context: dict[str, Any], object_store: Any | None = None
    ) -> dict[str, Any]:
        del object_store
        listener = get_listener(context, logger_name=__name__)

        config_dir = _optional_config_base_dir(context.get("_config_dir"))

        creation_requests = context.get("creation_requests") or ()
        authoring_requests = context.get("authoring_requests") or ()
        modification_requests = context.get("modification_requests") or ()
        if not isinstance(creation_requests, list | tuple):
            raise TypeError("creation_requests must be a list or tuple")
        if not isinstance(authoring_requests, list | tuple):
            raise TypeError("authoring_requests must be a list or tuple")
        if not isinstance(modification_requests, list | tuple):
            raise TypeError("modification_requests must be a list or tuple")
        if (
            not creation_requests
            and not authoring_requests
            and not modification_requests
        ):
            raise ValueError(
                "material authoring requires creation_requests, authoring_requests, "
                "or modification_requests"
            )

        backend_name = (
            _canonical_backend_name(context.get("backend", _BACKEND_UNSET))
            if creation_requests
            else "not_used"
        )

        output_dir = Path(context.get("output_dir", "created_materials")).resolve()
        source_usd = (
            Path(context["source_usd"]).resolve() if creation_requests else None
        )
        predictions_path = _optional_path(context.get("predictions_path"))
        output_predictions_path = Path(
            context.get("output_predictions_path")
            or output_dir / "created_predictions.jsonl"
        ).resolve()

        predictions = _read_predictions(predictions_path)
        registry = (
            _build_backend_registry(context, backend_name)
            if creation_requests
            else None
        )
        if registry is not None:
            _validate_backend_available(registry, backend_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        cancel_event = threading.Event()

        created_materials: list[CreatedMaterial] = []
        authored_packages: list[MaterialPackage] = []
        statuses: list[dict[str, Any]] = []
        assignments = 0
        fail_on_error = bool(context.get("fail_on_error", True))
        overwrite = bool(context.get("overwrite", False))
        material_profile = str(context.get("material_profile", "auto"))
        request_ids_by_material_id: dict[str, str] = {}
        processed_request_ids: set[str] = set()

        if creation_requests:
            assert source_usd is not None
            assert registry is not None
        for index, raw_spec in enumerate(creation_requests):
            if not isinstance(raw_spec, dict):
                raise TypeError("creation_requests entries must be dictionaries")
            assert source_usd is not None
            conditioning_options = _conditioning_options_for_request(
                context,
                request_spec=raw_spec,
                backend_name=backend_name,
                base_dir=config_dir,
            )
            _sync_cancel_event(context, cancel_event)
            request = _build_create_request(
                raw_spec,
                source_usd,
                backend_name,
                base_dir=config_dir,
            )
            material_id = request.recipe.material_id
            previous_request_id = request_ids_by_material_id.setdefault(
                material_id, request.request_id
            )
            if previous_request_id != request.request_id:
                raise ValueError(
                    "create_materials received conflicting requests for "
                    f"material_id {material_id!r}; use distinct recipe ids"
                )
            if request.request_id in processed_request_ids:
                continue
            processed_request_ids.add(request.request_id)
            package_dir = output_dir / "packages" / request.recipe.material_id
            listener.event(
                "material_creation.started",
                {
                    "request_id": request.request_id,
                    "recipe": request.recipe.name,
                    "backend": backend_name,
                    "index": index,
                },
            )
            try:
                created = _create_material_package_with_cancel_monitor(
                    request,
                    package_dir,
                    registry=registry,
                    cancel_event=cancel_event,
                    context=context,
                    output_dir=output_dir,
                    backend_name=backend_name,
                    material_profile=material_profile,
                    overwrite=overwrite,
                    conditioning_options=conditioning_options,
                )
            except MaterialCreationError as exc:
                status = {
                    "status": "error",
                    "request_id": request.request_id,
                    "recipe": request.recipe.name,
                    "backend": exc.backend or backend_name,
                    "code": exc.code.value,
                    "message": str(exc),
                    "diagnostics": [item.to_dict() for item in exc.diagnostics],
                }
                statuses.append(status)
                listener.event("material_creation.failed", status)
                if fail_on_error:
                    raise
                continue

            created_materials.append(created)
            status = {
                "status": "created",
                "request_id": request.request_id,
                "recipe": request.recipe.name,
                "material_name": created.material_list_entry.name,
                "material_id": created.material_id,
                "package_dir": package_dir.as_posix(),
                "material_usd_path": created.material_usd_path.as_posix(),
                "creation_manifest_path": created.creation_manifest_path.as_posix(),
                "cache_hit": bool(created.validation.get("cache_hit")),
                "texture_paths": {
                    key: path.as_posix()
                    for key, path in sorted(created.texture_paths.items())
                },
            }
            statuses.append(status)
            assignments += _assign_prediction(predictions, raw_spec, created)
            listener.event("material_creation.completed", status)

        direct_requests = [(raw_spec, None) for raw_spec in authoring_requests] + [
            (raw_spec, "modify") for raw_spec in modification_requests
        ]
        for index, (raw_spec, forced_operation) in enumerate(direct_requests):
            if not isinstance(raw_spec, dict):
                raise TypeError(
                    "authoring_requests and modification_requests entries must be "
                    "dictionaries"
                )
            _sync_cancel_event(context, cancel_event)
            request = _build_authoring_request(
                raw_spec,
                material_profile=material_profile,
                base_dir=config_dir,
                forced_operation=forced_operation,
            )
            material_id = request.recipe.material_id
            previous_request_id = request_ids_by_material_id.setdefault(
                material_id, request.request_id
            )
            if previous_request_id != request.request_id:
                raise ValueError(
                    "material authoring received conflicting requests for "
                    f"material_id {material_id!r}; use distinct recipe ids"
                )
            if request.request_id in processed_request_ids:
                continue
            processed_request_ids.add(request.request_id)
            package_dir = output_dir / "packages" / material_id
            listener.event(
                "material_authoring.started",
                {
                    "operation": request.operation.value,
                    "request_id": request.request_id,
                    "recipe": request.recipe.name,
                    "index": index,
                },
            )
            try:
                authored = author_material_package(
                    request,
                    package_dir,
                    overwrite=overwrite,
                )
            except (MaterialPackageAuthoringError, FileExistsError) as exc:
                status = {
                    "status": "error",
                    "operation": request.operation.value,
                    "request_id": request.request_id,
                    "recipe": request.recipe.name,
                    "code": "material_authoring_failed",
                    "message": str(exc),
                    "diagnostics": [],
                }
                statuses.append(status)
                listener.event("material_authoring.failed", status)
                if fail_on_error:
                    raise
                continue

            authored_packages.append(authored)
            status = {
                "status": (
                    "modified" if request.operation.value == "modify" else "created"
                ),
                "operation": request.operation.value,
                "request_id": request.request_id,
                "recipe": request.recipe.name,
                "material_name": authored.material_list_entry["name"],
                "material_id": authored.material_id,
                "package_dir": package_dir.as_posix(),
                "material_usd_path": authored.material_usd_path.as_posix(),
                "authoring_manifest_path": (
                    authored.authoring_manifest_path.as_posix()
                ),
                "cache_hit": bool(authored.validation.get("cache_hit")),
                "texture_paths": (
                    {
                        "albedo": authored.textures.albedo.as_posix(),
                        "normal": authored.textures.normal.as_posix(),
                        "orm": authored.textures.orm.as_posix(),
                    }
                    if authored.textures is not None
                    else {}
                ),
            }
            statuses.append(status)
            assignments += _assign_authored_prediction(
                predictions,
                raw_spec,
                authored,
            )
            listener.event("material_authoring.completed", status)

        authored_materials: list[CreatedMaterial | MaterialPackage] = [
            *created_materials,
            *authored_packages,
        ]
        directly_created_materials = sum(
            package.operation.value == "create" for package in authored_packages
        )
        modified_material_count = sum(
            package.operation.value == "modify" for package in authored_packages
        )
        material_library_path = _created_material_library(
            output_dir, authored_materials
        )
        material_entries = [
            _aggregate_material_entry(output_dir, authored)
            for authored in authored_materials
        ]
        materials_data = {
            "library_path": material_library_path.as_posix()
            if material_library_path
            else None,
            "entries": material_entries,
        }
        materials_yaml_path = output_dir / "materials.yaml"
        _write_materials_yaml(
            materials_yaml_path, material_library_path, material_entries
        )
        _write_predictions(output_predictions_path, predictions)
        manifest_path = output_dir / "material_creation_status.json"
        _write_status_manifest(
            manifest_path,
            statuses=statuses,
            materials_yaml_path=materials_yaml_path,
            material_library_path=material_library_path,
            output_predictions_path=output_predictions_path,
        )

        return {
            "output_dir": output_dir.as_posix(),
            "created_material_count": len(created_materials)
            + directly_created_materials,
            "generated_material_count": len(created_materials),
            "directly_authored_material_count": len(authored_packages),
            "modified_material_count": modified_material_count,
            "authored_material_count": len(authored_materials),
            "assignment_count": assignments,
            "created_materials_manifest_path": manifest_path.as_posix(),
            "created_materials_yaml_path": materials_yaml_path.as_posix(),
            "created_material_library_path": (
                material_library_path.as_posix() if material_library_path else None
            ),
            "created_material_entries": material_entries,
            "created_materials_data": materials_data,
            "predictions_path": output_predictions_path.as_posix(),
            "statuses": statuses,
        }


class CreateMaterialsTask(AuthorMaterialsTask):
    """Backward-compatible task name for the shared authoring implementation."""

    def __init__(self) -> None:
        super().__init__()
        self.name = "CreateMaterials"


def _canonical_backend_name(raw_backend: Any) -> str:
    if raw_backend is _BACKEND_UNSET:
        return "fake"
    if raw_backend is None:
        raise ValueError("create_materials backend must be a non-empty string")
    backend_name = str(raw_backend).strip()
    if not backend_name:
        raise ValueError("create_materials backend must be a non-empty string")
    if backend_name in _FAKE_BACKEND_ALIASES:
        return "fake"
    return backend_name


def _build_backend_registry(
    context: dict[str, Any],
    backend_name: str,
) -> MaterialCreationBackendRegistry:
    registry = MaterialCreationBackendRegistry()
    if backend_name == "fake":
        registry.register(
            FakeMaterialCreationBackend(
                context.get("fake_behavior", FakeMaterialBackendBehavior.SUCCESS)
            ),
            make_default=True,
        )
    elif backend_name == _STEP1X_BACKEND_NAME:
        registry.register(_create_step1x_backend(context), make_default=True)
    return registry


def _validate_backend_available(
    registry: MaterialCreationBackendRegistry,
    backend_name: str,
) -> None:
    registry.resolve(backend_name)


def _prepare_material_conditioning(
    request: CreateMaterialRequest,
    *,
    output_dir: Path,
    backend_name: str,
    cancel_event: threading.Event,
    options: MaterialConditioningOptions | None,
) -> PreparedMaterialConditioning | None:
    if backend_name != _STEP1X_BACKEND_NAME:
        return None
    result = prepare_material_conditioning(
        request,
        output_dir / "conditioning",
        options=options,
        cancel_event=cancel_event,
    )
    return result.conditioning


def _create_material_package_with_cancel_monitor(
    request: CreateMaterialRequest,
    package_dir: Path,
    *,
    registry: MaterialCreationBackendRegistry,
    cancel_event: threading.Event,
    context: dict[str, Any],
    output_dir: Path,
    backend_name: str,
    material_profile: str,
    overwrite: bool,
    conditioning_options: MaterialConditioningOptions | None,
) -> CreatedMaterial:
    stop_event: threading.Event | None = None
    monitor_thread: threading.Thread | None = None
    cancel_checker_error: Exception | None = None
    cancel_checker = context.get("cancel_checker")
    if callable(cancel_checker):
        stop_event = threading.Event()

        def monitor_cancel_checker() -> None:
            nonlocal cancel_checker_error
            while not stop_event.wait(_CANCEL_MONITOR_INTERVAL_SECONDS):
                try:
                    should_cancel = cancel_checker()
                except Exception as exc:
                    cancel_checker_error = exc
                    cancel_event.set()
                    return
                if should_cancel:
                    cancel_event.set()
                    return

        monitor_thread = threading.Thread(
            target=monitor_cancel_checker,
            name="CreateMaterialsCancelMonitor",
            daemon=True,
        )
        monitor_thread.start()

    created: CreatedMaterial | None = None
    create_error: Exception | None = None
    try:
        conditioning = _prepare_material_conditioning(
            request,
            output_dir=output_dir,
            backend_name=backend_name,
            cancel_event=cancel_event,
            options=conditioning_options,
        )
        created = create_material_package(
            request,
            package_dir,
            registry=registry,
            conditioning=conditioning,
            cancel_event=cancel_event,
            material_profile=material_profile,
            overwrite=overwrite,
        )
    except Exception as exc:
        create_error = exc
    finally:
        if stop_event is not None:
            stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=1.0)
    # A failed cancellation probe takes precedence over a concurrently completed package.
    if cancel_checker_error is not None:
        raise cancel_checker_error
    if create_error is not None:
        raise create_error
    if created is None:
        raise RuntimeError("create_material_package did not return a material")
    return created


def _conditioning_options_for_request(
    context: Mapping[str, Any],
    *,
    request_spec: Mapping[str, Any],
    backend_name: str,
    base_dir: Path | None,
) -> MaterialConditioningOptions | None:
    if backend_name != _STEP1X_BACKEND_NAME:
        return None
    raw_options = request_spec.get("conditioning", context.get("conditioning"))
    if raw_options is None:
        return None
    if not isinstance(raw_options, Mapping):
        raise TypeError("create_materials conditioning must be a mapping")
    return MaterialConditioningOptions.from_dict(raw_options, base_dir=base_dir)


def _create_step1x_backend(context: dict[str, Any]) -> Any:
    try:
        from apps.texture_gen_step1x_service.backend import Step1XBackendConfig

        from material_agent.material_library_generation.step1x_backend import (
            Step1XMaterialCreationBackend,
            Step1XMaterialCreationConfig,
        )
    except ImportError as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.BACKEND_UNAVAILABLE,
            "Step1X material creation backend is unavailable; install the "
            "material-agent[step1x] optional dependencies.",
            backend=_STEP1X_BACKEND_NAME,
            retryable=False,
        ) from exc

    config_data = _step1x_config_data(context)
    runtime_config = _step1x_runtime_config(
        Step1XBackendConfig,
        config_data,
        base_dir=_optional_config_base_dir(context.get("_config_dir")),
    )
    strength = _step1x_strength(config_data)
    custom_parameters = _step1x_custom_parameters(config_data, runtime_config)
    model_revisions = (
        _string_tuple(
            config_data["model_revisions"],
            field_name="model_revisions",
        )
        if "model_revisions" in config_data
        else Step1XMaterialCreationConfig.model_revisions
    )
    backend_revision = _step1x_backend_revision(
        config_data,
        runtime_config=runtime_config,
        model_revisions=model_revisions,
        strength=strength,
        custom_parameters=custom_parameters,
    )
    material_config_kwargs: dict[str, Any] = {
        "step1x": runtime_config,
        "backend_name": _STEP1X_BACKEND_NAME,
        "backend_revision": backend_revision,
        "model_revisions": model_revisions,
        "strength": strength,
    }
    if custom_parameters:
        material_config_kwargs["custom_parameters"] = custom_parameters

    try:
        config = Step1XMaterialCreationConfig(**material_config_kwargs)
        return Step1XMaterialCreationBackend(config=config)
    except ValueError:
        raise
    except Exception as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.BACKEND_UNAVAILABLE,
            f"Step1X material creation backend could not be constructed: {exc}",
            backend=_STEP1X_BACKEND_NAME,
            retryable=False,
        ) from exc


def _step1x_config_data(context: dict[str, Any]) -> dict[str, Any]:
    config_data: dict[str, Any] = {}
    for key in ("step1x", _STEP1X_BACKEND_NAME):
        raw_config = context.get(key)
        if raw_config is None:
            continue
        if not isinstance(raw_config, dict):
            raise ValueError(f"{key} config must be a dictionary")
        config_data.update(raw_config)
    return config_data


def _step1x_runtime_config(
    config_cls: Any,
    config_data: dict[str, Any],
    *,
    base_dir: Path | None = None,
) -> Any:
    runtime_kwargs: dict[str, Any] = {}
    for key in (
        "runtime_dir",
        "model_dir",
        "cache_dir",
        "output_dir",
        "python_executable",
        "edit_script",
    ):
        if key in config_data:
            runtime_kwargs[key] = _optional_config_path(
                config_data[key],
                field_name=key,
                base_dir=base_dir,
            )
    if "command_template" in config_data:
        value = config_data["command_template"]
        runtime_kwargs["command_template"] = None if value is None else str(value)
    if "timeout_sec" in config_data:
        runtime_kwargs["timeout_sec"] = _int_config_value(
            config_data["timeout_sec"],
            field_name="timeout_sec",
        )
    for key in (
        "validate_assets",
        "skip_material_anything",
        "require_upscaler",
    ):
        if key in config_data:
            runtime_kwargs[key] = _bool_config_value(config_data[key], field_name=key)
    for key in ("extra_args", "required_executables"):
        if key in config_data:
            runtime_kwargs[key] = _string_tuple(config_data[key], field_name=key)

    try:
        return replace(config_cls.from_env(), **runtime_kwargs)
    except ValueError:
        raise
    except Exception as exc:
        raise MaterialCreationError(
            MaterialCreationErrorCode.BACKEND_UNAVAILABLE,
            f"Step1X runtime config could not be constructed: {exc}",
            backend=_STEP1X_BACKEND_NAME,
            retryable=False,
        ) from exc


def _optional_config_base_dir(value: Any) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, str | os.PathLike):
        return Path(value).resolve()
    raise ValueError("create_materials _config_dir must be a path string")


def _optional_config_path(
    value: Any,
    *,
    field_name: str,
    base_dir: Path | None = None,
) -> Path | None:
    if value is None or value == "":
        return None
    if isinstance(value, str | os.PathLike):
        path = Path(value)
        if base_dir is not None and not path.is_absolute():
            return (base_dir / path).resolve()
        return path
    raise ValueError(f"step1x {field_name} must be a path string or null")


def _string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(shlex.split(value))
    if isinstance(value, list | tuple):
        return tuple(str(item) for item in value)
    raise ValueError(f"step1x {field_name} must be a string or list")


def _float_config_value(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"step1x {field_name} must be a number")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"step1x {field_name} must be a number") from exc


def _int_config_value(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"step1x {field_name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"step1x {field_name} must be an integer") from exc


def _bool_config_value(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"step1x {field_name} must be a boolean")


def _step1x_strength(config_data: dict[str, Any]) -> float:
    if "strength" not in config_data:
        return 0.8
    return _float_config_value(config_data["strength"], field_name="strength")


def _step1x_custom_parameters(
    config_data: dict[str, Any],
    runtime_config: Any,
) -> dict[str, Any]:
    raw_custom_parameters = config_data.get("custom_parameters", {})
    if not isinstance(raw_custom_parameters, dict):
        raise ValueError("step1x custom_parameters must be a dictionary")
    custom_parameters = dict(raw_custom_parameters)
    if (
        "skip_material_anything" in config_data
        and "skip_material_anything" not in custom_parameters
    ):
        custom_parameters["skip_material_anything"] = bool(
            runtime_config.skip_material_anything
        )
    return custom_parameters


def _step1x_backend_revision(
    config_data: dict[str, Any],
    *,
    runtime_config: Any,
    model_revisions: tuple[str, ...],
    strength: float,
    custom_parameters: dict[str, Any],
) -> str:
    from material_agent.material_library_generation.step1x_backend import (
        STEP1X_MATERIAL_CREATION_ADAPTER_REVISION,
        resolve_step1x_material_creation_revision,
    )

    configured_revision = config_data.get("backend_revision")
    base_revision = (
        str(configured_revision)
        if configured_revision is not None
        else STEP1X_MATERIAL_CREATION_ADAPTER_REVISION
    )
    return str(
        resolve_step1x_material_creation_revision(
            base_revision,
            step1x=runtime_config,
            model_revisions=model_revisions,
            strength=strength,
            custom_parameters=custom_parameters,
        )
    )


def _build_create_request(
    spec: dict[str, Any],
    source_usd: Path,
    backend_name: str,
    *,
    base_dir: Path | None = None,
) -> CreateMaterialRequest:
    recipe_data = spec.get("recipe")
    if not isinstance(recipe_data, dict):
        raise ValueError("creation request requires a recipe object")
    target_prim_paths = spec.get("target_prim_paths")
    if isinstance(target_prim_paths, str):
        target_prim_paths = (target_prim_paths,)
    if not isinstance(target_prim_paths, list | tuple):
        raise ValueError("creation request requires target_prim_paths")
    reference_image_uris = spec.get("reference_image_uris", ())
    if isinstance(reference_image_uris, str):
        reference_image_uris = (reference_image_uris,)
    return CreateMaterialRequest(
        source_usd=source_usd,
        source_usd_sha256=spec.get("source_usd_sha256"),
        target_prim_paths=tuple(str(path) for path in target_prim_paths),
        recipe=MaterialRecipe.from_dict(recipe_data, base_dir=base_dir),
        reference_image_uris=tuple(
            _resolve_local_reference_uri(uri, base_dir) for uri in reference_image_uris
        ),
        creation_mode=MaterialCreationMode(str(spec.get("creation_mode", "asset_uv"))),
        texture_size=int(spec.get("texture_size", 1024)),
        backend=backend_name,
        seed=spec.get("seed"),
    )


def _build_authoring_request(
    spec: dict[str, Any],
    *,
    material_profile: str,
    base_dir: Path | None = None,
    forced_operation: str | None = None,
) -> MaterialAuthoringRequest:
    recipe_data = spec.get("recipe")
    if not isinstance(recipe_data, dict):
        raise ValueError("authoring request requires a recipe object")

    operation = str(forced_operation or spec.get("operation", "create")).strip()
    if forced_operation is not None and spec.get("operation", operation) != operation:
        raise ValueError(
            f"compatibility modification request operation must be {operation!r}"
        )
    if operation not in {"create", "modify"}:
        raise ValueError("authoring request operation must be 'create' or 'modify'")

    target_prim_paths = spec.get("target_prim_paths", ())
    if isinstance(target_prim_paths, str):
        target_prim_paths = (target_prim_paths,)
    if not isinstance(target_prim_paths, list | tuple):
        raise TypeError("authoring request target_prim_paths must be a sequence")

    textures = _texture_map_set_from_spec(spec.get("textures"), base_dir=base_dir)
    recipe = MaterialRecipe.from_dict(recipe_data, base_dir=base_dir)
    requested_profile = str(spec.get("material_profile", material_profile))

    if operation == "create":
        if spec.get("source") or spec.get("source_material_usd"):
            raise ValueError(
                "create authoring request must not include a source material"
            )
        return MaterialAuthoringRequest(
            operation="create",
            recipe=recipe,
            textures=textures,
            target_prim_paths=tuple(str(path) for path in target_prim_paths),
            material_profile=requested_profile,
        )

    source_data = spec.get("source") or {}
    if not isinstance(source_data, dict):
        raise TypeError("modify authoring request source must be a mapping")
    source_usd_value = source_data.get("usd_path") or spec.get("source_material_usd")
    source_prim_path = source_data.get("material_prim_path") or spec.get(
        "source_material_prim_path"
    )
    if not source_usd_value or not source_prim_path:
        raise ValueError(
            "modify authoring request requires source material USD and prim path"
        )

    return MaterialAuthoringRequest(
        operation="modify",
        recipe=recipe,
        source=SourceMaterialReference(
            usd_path=_resolve_config_path(source_usd_value, base_dir),
            material_prim_path=str(source_prim_path),
            usd_sha256=(
                source_data.get("usd_sha256") or spec.get("source_material_sha256")
            ),
        ),
        textures=textures,
        target_prim_paths=tuple(str(path) for path in target_prim_paths),
        material_profile=requested_profile,
    )


def _texture_map_set_from_spec(
    texture_data: Any,
    *,
    base_dir: Path | None,
) -> TextureMapSet | None:
    if texture_data is None:
        return None
    if not isinstance(texture_data, dict):
        raise TypeError("authoring request textures must be a mapping")
    missing = [
        channel
        for channel in ("albedo", "normal", "orm")
        if not texture_data.get(channel)
    ]
    if missing:
        raise ValueError(
            "authoring request textures require albedo, normal, and orm; "
            f"missing: {', '.join(missing)}"
        )
    return TextureMapSet(
        albedo=_resolve_config_path(texture_data["albedo"], base_dir),
        normal=_resolve_config_path(texture_data["normal"], base_dir),
        orm=_resolve_config_path(texture_data["orm"], base_dir),
    )


def _resolve_config_path(value: Any, base_dir: Path | None) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def _resolve_local_reference_uri(value: Any, base_dir: Path | None) -> str:
    uri = str(value).strip()
    if not uri or base_dir is None or urlparse(uri).scheme:
        return uri
    path = Path(uri)
    if path.is_absolute():
        return uri
    return str((base_dir / path).resolve())


def _read_predictions(predictions_path: Path | None) -> list[dict[str, Any]]:
    if predictions_path is None or not predictions_path.exists():
        return []
    predictions: list[dict[str, Any]] = []
    with predictions_path.open(encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                predictions.append(item)
    return predictions


def _write_predictions(path: Path, predictions: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for item in predictions:
            json.dump(item, stream, sort_keys=True)
            stream.write("\n")


def _assign_prediction(
    predictions: list[dict[str, Any]], spec: dict[str, Any], created: CreatedMaterial
) -> int:
    prediction_ids = _prediction_ids_for_request(spec)
    assigned = 0
    for prediction in predictions:
        if prediction.get("id") not in prediction_ids:
            continue
        materials = prediction.get("materials")
        if not isinstance(materials, dict):
            materials = {}
            prediction["materials"] = materials
        materials["material"] = created.material_list_entry.name
        materials["creation_action"] = "create_new"
        materials["creation_request_id"] = created.provenance.request_id
        materials["creation_manifest"] = created.creation_manifest_path.as_posix()
        prediction["material_creation"] = {
            "action": "create_new",
            "material_name": created.material_list_entry.name,
            "material_id": created.material_id,
            "creation_request_id": created.provenance.request_id,
            "creation_manifest": created.creation_manifest_path.as_posix(),
        }
        assigned += 1
    return assigned


def _assign_authored_prediction(
    predictions: list[dict[str, Any]],
    spec: dict[str, Any],
    authored: MaterialPackage,
) -> int:
    prediction_ids = _prediction_ids_for_request(spec)
    action = "modify_existing" if authored.operation.value == "modify" else "create_new"
    assigned = 0
    for prediction in predictions:
        if prediction.get("id") not in prediction_ids:
            continue
        materials = prediction.get("materials")
        if not isinstance(materials, dict):
            materials = {}
            prediction["materials"] = materials
        material_name = str(authored.material_list_entry["name"])
        materials["material"] = material_name
        materials["creation_action"] = action
        materials["authoring_request_id"] = authored.request_id
        materials["authoring_manifest"] = authored.authoring_manifest_path.as_posix()
        prediction["material_creation"] = {
            "action": action,
            "material_name": material_name,
            "material_id": authored.material_id,
            "authoring_request_id": authored.request_id,
            "authoring_manifest": authored.authoring_manifest_path.as_posix(),
        }
        assigned += 1
    return assigned


def _prediction_ids_for_request(spec: dict[str, Any]) -> set[str]:
    prediction_ids: set[str] = set()
    prediction_id = spec.get("prediction_id") or spec.get("id")
    if isinstance(prediction_id, str) and prediction_id:
        prediction_ids.add(prediction_id)
    target_prim_paths = spec.get("target_prim_paths")
    if isinstance(target_prim_paths, str):
        prediction_ids.add(target_prim_paths)
    elif isinstance(target_prim_paths, list | tuple):
        prediction_ids.update(
            str(path) for path in target_prim_paths if isinstance(path, str) and path
        )
    return prediction_ids


def _aggregate_material_entry(
    output_dir: Path,
    authored: CreatedMaterial | MaterialPackage,
) -> dict[str, Any]:
    if isinstance(authored, CreatedMaterial):
        entry: dict[str, Any] = dict(authored.material_list_entry.to_dict())
        entry["creation_manifest"] = _relative_path(
            authored.creation_manifest_path,
            output_dir,
        )
        return entry
    entry = dict(authored.material_list_entry)
    entry["authoring_manifest"] = _relative_path(
        authored.authoring_manifest_path,
        output_dir,
    )
    return entry


def _created_material_library(
    output_dir: Path,
    created_materials: list[CreatedMaterial | MaterialPackage],
) -> Path | None:
    if not created_materials:
        return None
    if len(created_materials) == 1:
        return Path(created_materials[0].material_usd_path)

    combined_path = output_dir / "created_materials.usda"
    if combined_path.exists():
        combined_path.unlink()
    layer = Sdf.Layer.CreateNew(str(combined_path))
    layer.defaultPrim = "World"
    for created in created_materials:
        source_layer = Sdf.Layer.FindOrOpen(str(created.material_usd_path))
        if not source_layer:
            raise RuntimeError(
                f"Could not open created material library: {created.material_usd_path}"
            )
        source_path = Sdf.Path(created.material_prim_path)
        if not source_layer.GetPrimAtPath(source_path):
            raise RuntimeError(
                f"Created material prim not found: {created.material_prim_path}"
            )
        _ensure_parent_specs(layer, source_path)
        if not Sdf.CopySpec(source_layer, source_path, layer, source_path):
            raise RuntimeError(
                f"Could not copy created material prim: {created.material_prim_path}"
            )
        remap_asset_paths_in_prim(
            layer,
            source_path,
            created.material_usd_path.resolve().parent,
            combined_path.resolve().parent,
        )
        clear_color_space_on_empty_asset_inputs(layer, source_path)
    layer.Save()
    return combined_path


def _ensure_parent_specs(layer: Sdf.Layer, prim_path: Sdf.Path) -> None:
    parents: list[Sdf.Path] = []
    parent = prim_path.GetParentPath()
    while parent != Sdf.Path.absoluteRootPath:
        parents.append(parent)
        parent = parent.GetParentPath()
    for parent_path in reversed(parents):
        if not layer.GetPrimAtPath(parent_path):
            Sdf.CreatePrimInLayer(layer, parent_path)


def _write_materials_yaml(
    path: Path, material_library_path: Path | None, entries: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "library_path": (
            _relative_path(material_library_path, path.parent)
            if material_library_path
            else None
        ),
        "entries": entries,
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def _write_status_manifest(
    path: Path,
    *,
    statuses: list[dict[str, Any]],
    materials_yaml_path: Path,
    material_library_path: Path | None,
    output_predictions_path: Path,
) -> None:
    payload = {
        "schema_version": "material-agent-create-materials-step.v1",
        "statuses": statuses,
        "materials_yaml_path": materials_yaml_path.as_posix(),
        "material_library_path": (
            material_library_path.as_posix() if material_library_path else None
        ),
        "predictions_path": output_predictions_path.as_posix(),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sync_cancel_event(context: dict[str, Any], cancel_event: threading.Event) -> None:
    cancel_checker = context.get("cancel_checker")
    if callable(cancel_checker) and cancel_checker():
        cancel_event.set()


def _optional_path(value: Any) -> Path | None:
    if value is None:
        return None
    return Path(value).resolve()


def _relative_path(path: Path, base_dir: Path) -> str:
    return os.path.relpath(path.resolve(), base_dir.resolve()).replace("\\", "/")
