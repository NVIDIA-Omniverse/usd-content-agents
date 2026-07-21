# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step1X + Material Anything runtime preflight and smoke harness.

The default CLI path is preflight-only. The single-material smoke path calls the
real Step1X material creation backend only when the caller passes an explicit
``--run-real-step1x`` acknowledgement.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from apps.texture_gen_step1x_service.backend import (
    ExternalStep1XRunner,
    Step1XBackend,
    Step1XBackendConfig,
    Step1XRunRequest,
)

from material_agent.material_library_generation.creation_contract import (
    CreateMaterialRequest,
    MaterialColorSpace,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationMode,
    PreparedMaterialConditioning,
)
from material_agent.material_library_generation.schema import MaterialRecipe
from material_agent.material_library_generation.step1x_backend import (
    Step1XMaterialCreationBackend,
    Step1XMaterialCreationConfig,
    result_fingerprint,
)


class Step1XPreflightCategory(StrEnum):
    """Stable readiness categories surfaced by the WP9D harness."""

    RUNTIME = "runtime"
    MODEL_CHECKPOINTS = "model_checkpoints"
    PYTHON_EXECUTABLE = "python_executable"
    MATERIAL_ANYTHING = "material_anything"
    GPU_CUDA = "gpu_cuda"
    COMMAND_TEMPLATE = "command_template"


@dataclass(frozen=True)
class Step1XPreflightIssue:
    """One structured preflight blocker or warning."""

    category: Step1XPreflightCategory
    code: str
    message: str
    severity: str = "error"
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "category": self.category.value,
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
        }
        if self.detail:
            data["detail"] = dict(self.detail)
        return data


@dataclass(frozen=True)
class Step1XPreflightReport:
    """Structured preflight report for Step1X material creation."""

    ready: bool
    status: str
    categories: dict[str, dict[str, Any]]
    issues: tuple[Step1XPreflightIssue, ...]
    health: dict[str, Any]
    command: list[str] | None = None

    @property
    def blockers(self) -> tuple[Step1XPreflightIssue, ...]:
        return tuple(issue for issue in self.issues if issue.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "status": self.status,
            "ready": self.ready,
            "categories": self.categories,
            "blockers": [issue.to_dict() for issue in self.blockers],
            "issues": [issue.to_dict() for issue in self.issues],
            "health": self.health,
        }
        if self.command is not None:
            data["command"] = _redact_command_preview(self.command)
        return data


def preflight_step1x_runtime(
    config: Step1XBackendConfig | None = None,
    *,
    require_material_anything: bool = True,
    require_gpu: bool = False,
) -> Step1XPreflightReport:
    """Check Step1X + Material Anything readiness without running generation."""

    original_config = config or Step1XBackendConfig.from_env()
    effective_config = (
        replace(original_config, skip_material_anything=False)
        if require_material_anything
        else original_config
    )
    backend = Step1XBackend(config=effective_config)
    health = backend.health()
    missing = backend._missing_runtime_inputs()

    issues: list[Step1XPreflightIssue] = []
    if require_material_anything and original_config.skip_material_anything:
        issues.append(
            Step1XPreflightIssue(
                category=Step1XPreflightCategory.MATERIAL_ANYTHING,
                code="material_anything_default_disabled",
                message=(
                    "TEXTURE_STEP1X_SKIP_MA defaults the Step1X service to "
                    "albedo-only requests, but Material Agent material creation "
                    "uses a per-request skip_material_anything=false override. "
                    "Preflight validates Material Anything readiness with that "
                    "effective override."
                ),
                severity="warning",
            )
        )
    issues.extend(_issues_from_missing_inputs(missing))
    material_anything = _material_anything_capability(backend)
    material_anything_issue = _material_anything_capability_issue(
        material_anything,
        required=require_material_anything,
    )
    if material_anything_issue is not None and not _has_category_error(
        issues,
        Step1XPreflightCategory.MATERIAL_ANYTHING,
    ):
        issues.append(material_anything_issue)

    command, command_issue = _probe_command_readiness(effective_config, missing)
    if command_issue is not None:
        issues.append(command_issue)

    gpu_available = health.gpu_available
    if require_gpu and gpu_available is not True:
        issues.append(
            Step1XPreflightIssue(
                category=Step1XPreflightCategory.GPU_CUDA,
                code="gpu_cuda_unavailable",
                message=(
                    "GPU/CUDA availability was not confirmed by nvidia-smi."
                    if gpu_available is None
                    else "GPU/CUDA availability probe reported no GPU."
                ),
                detail={"gpu_available": gpu_available},
            )
        )
    elif gpu_available is None:
        issues.append(
            Step1XPreflightIssue(
                category=Step1XPreflightCategory.GPU_CUDA,
                code="gpu_cuda_unknown",
                message="GPU/CUDA availability could not be detected by nvidia-smi.",
                severity="warning",
            )
        )

    categories = _category_summaries(
        backend=backend,
        config=effective_config,
        original_config=original_config,
        require_material_anything=require_material_anything,
        issues=tuple(issues),
        command=command,
        command_issue=command_issue,
        gpu_available=gpu_available,
        require_gpu=require_gpu,
    )
    ready = not any(issue.severity == "error" for issue in issues)
    return Step1XPreflightReport(
        ready=ready,
        status="ready" if ready else "blocked",
        categories=categories,
        issues=tuple(issues),
        health=health.model_dump(mode="json"),
        command=command,
    )


def run_single_material_smoke(
    request: CreateMaterialRequest,
    conditioning: PreparedMaterialConditioning,
    *,
    output_dir: Path,
    config: Step1XBackendConfig | None = None,
    run_real_step1x: bool = False,
    require_gpu: bool = True,
) -> dict[str, Any]:
    """Run one real Step1X material creation only after explicit opt-in."""

    if not run_real_step1x:
        return _single_material_smoke_disabled_payload(
            config,
            require_gpu=require_gpu,
        )

    runtime_config = config or Step1XBackendConfig.from_env()
    preflight = preflight_step1x_runtime(
        runtime_config,
        require_material_anything=True,
        require_gpu=require_gpu,
    )
    if preflight.blockers:
        return {
            "status": "blocked",
            "blockers": [issue.to_dict() for issue in preflight.blockers],
            "preflight": preflight.to_dict(),
        }

    output_dir = Path(output_dir).resolve()
    backend = Step1XMaterialCreationBackend(
        config=Step1XMaterialCreationConfig(
            step1x=replace(runtime_config, skip_material_anything=False)
        ),
        runner=ExternalStep1XRunner(
            replace(runtime_config, skip_material_anything=False)
        ),
    )
    try:
        result = backend.create(
            request,
            output_dir=output_dir,
            conditioning=conditioning,
            cancel_event=threading.Event(),
        )
    except MaterialCreationError as exc:
        issue = Step1XPreflightIssue(
            category=Step1XPreflightCategory.RUNTIME,
            code=exc.code.value,
            message=str(exc),
            detail={"backend": exc.backend},
        )
        return {
            "status": "failed",
            "blockers": [issue.to_dict()],
            "preflight": preflight.to_dict(),
        }

    return {
        "status": "completed",
        "preflight": preflight.to_dict(),
        "request": request.to_dict(),
        "output_dir": output_dir.as_posix(),
        "artifacts": [artifact.to_dict() for artifact in result.artifacts],
        "degradations": [degradation.to_dict() for degradation in result.degradations],
        "provenance": result.provenance.to_dict(),
        "fingerprint": result_fingerprint(result),
    }


def load_create_material_request(path: Path) -> CreateMaterialRequest:
    """Load a material creation request from the JSON shape emitted by to_dict()."""

    data = _load_json_object(path)
    base_dir = path.parent
    recipe = MaterialRecipe.from_dict(_require_dict(data, "recipe"), base_dir=base_dir)
    refs = tuple(
        _resolve_local_reference(str(uri), base_dir)
        for uri in data.get("reference_image_uris", ())
    )
    source_usd = _resolve_local_path(str(data["source_usd"]), base_dir)
    return CreateMaterialRequest(
        source_usd=source_usd,
        target_prim_paths=tuple(str(path) for path in data["target_prim_paths"]),
        recipe=recipe,
        reference_image_uris=refs,
        creation_mode=MaterialCreationMode(
            data.get("creation_mode", MaterialCreationMode.ASSET_UV.value)
        ),
        texture_size=int(data.get("texture_size", 1024)),
        backend=str(data.get("backend", "step1x_material_anything")),
        seed=data.get("seed"),
        source_usd_sha256=data.get("source_usd_sha256"),
        schema_version=str(data.get("schema_version", "material-agent-create.v1")),
    )


def load_prepared_conditioning(path: Path) -> PreparedMaterialConditioning:
    """Load prepared conditioning from the JSON shape emitted by to_dict()."""

    data = _load_json_object(path)
    base_dir = path.parent
    artifacts = tuple(
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind(str(item["kind"])),
            uri=_resolve_local_reference(str(item["uri"]), base_dir),
            color_space=(
                MaterialColorSpace(str(item["color_space"]))
                if item.get("color_space") is not None
                else None
            ),
            view=item.get("view"),
            sha256=item.get("sha256"),
        )
        for item in data.get("artifacts", ())
    )
    return PreparedMaterialConditioning(
        request_id=str(data["request_id"]),
        target_prim_paths=tuple(str(path) for path in data["target_prim_paths"]),
        artifacts=artifacts,
        reference_image_uris=tuple(
            _resolve_local_reference(str(uri), base_dir)
            for uri in data.get("reference_image_uris", ())
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = _config_from_args(args)

    if args.command == "smoke":
        if not args.run_real_step1x:
            payload = _single_material_smoke_disabled_payload(
                config,
                require_gpu=not args.allow_gpu_unknown,
            )
        else:
            if (
                args.request is None
                or args.conditioning is None
                or args.output_dir is None
            ):
                parser.error(
                    "--request, --conditioning, and --output-dir are required "
                    "with --run-real-step1x"
                )
            request = load_create_material_request(args.request)
            conditioning = load_prepared_conditioning(args.conditioning)
            payload = run_single_material_smoke(
                request,
                conditioning,
                output_dir=args.output_dir,
                config=config,
                run_real_step1x=True,
                require_gpu=not args.allow_gpu_unknown,
            )
    else:
        report = preflight_step1x_runtime(
            config,
            require_material_anything=not getattr(
                args, "allow_material_anything_disabled", False
            ),
            require_gpu=getattr(args, "require_gpu", False),
        )
        payload = report.to_dict()

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload.get("status") in {"ready", "completed"} else 2


def _issues_from_missing_inputs(missing: list[str]) -> list[Step1XPreflightIssue]:
    return [_issue_from_missing_input(item) for item in missing]


def _single_material_smoke_disabled_payload(
    config: Step1XBackendConfig | None,
    *,
    require_gpu: bool,
) -> dict[str, Any]:
    preflight = preflight_step1x_runtime(config, require_gpu=require_gpu)
    blocker = Step1XPreflightIssue(
        category=Step1XPreflightCategory.RUNTIME,
        code="real_smoke_not_explicitly_enabled",
        message=(
            "Single-material smoke is disabled by default; pass "
            "--run-real-step1x to execute the real backend."
        ),
    )
    return {
        "status": "blocked",
        "blockers": [blocker.to_dict()],
        "preflight": preflight.to_dict(),
    }


def _issue_from_missing_input(message: str) -> Step1XPreflightIssue:
    upper = message.upper()
    if "MATERIAL ANYTHING" in upper:
        category = Step1XPreflightCategory.MATERIAL_ANYTHING
        code = "material_anything_unavailable"
    elif (
        "MODEL" in upper
        or "CHECKPOINT" in upper
        or "PRETRAINED_MODELS" in upper
        or "CONTROL_SD15_DEPTH" in upper
    ):
        category = Step1XPreflightCategory.MODEL_CHECKPOINTS
        code = "model_or_checkpoint_missing"
    elif (
        "TEXTURE_STEP1X_PYTHON" in upper
        or "PYTHON MODULE" in upper
        or "TEXTURE_STEP1X_REQUIRED_EXECUTABLES" in upper
        or "EXECUTABLE" in upper
    ):
        category = Step1XPreflightCategory.PYTHON_EXECUTABLE
        code = "python_or_executable_missing"
    else:
        category = Step1XPreflightCategory.RUNTIME
        code = "runtime_missing"
    return Step1XPreflightIssue(category=category, code=code, message=message)


def _probe_command_readiness(
    config: Step1XBackendConfig,
    missing_inputs: list[str],
) -> tuple[list[str] | None, Step1XPreflightIssue | None]:
    request = Step1XRunRequest(
        prompt="step1x preflight material smoke",
        seed=482,
        strength=0.8,
        texture_size=1024,
        source_asset_uri=Path("/tmp/step1x_preflight_source.usda").as_uri(),
        source_asset_path=Path("/tmp/step1x_preflight_source.usda"),
        source_albedo_path=Path("/tmp/step1x_preflight_source_albedo.png"),
        reference_image_uris=(),
        turntable_video_uri=None,
        multiview_image_uris=(),
        target=None,
        scope=None,
        job_id="step1x-preflight",
        output_dir=Path("/tmp/step1x_preflight_output"),
        runtime_dir=config.runtime_dir,
        model_dir=config.model_dir,
        cache_dir=config.cache_dir,
        custom_parameters={"skip_material_anything": False},
    )
    try:
        command = ExternalStep1XRunner(config)._build_command(
            request,
            Path("/tmp/step1x_preflight_source.usda"),
        )
    except (RuntimeError, ValueError) as exc:
        return None, Step1XPreflightIssue(
            category=Step1XPreflightCategory.COMMAND_TEMPLATE,
            code="command_not_buildable",
            message=str(exc),
        )

    if not command:
        return command, Step1XPreflightIssue(
            category=Step1XPreflightCategory.COMMAND_TEMPLATE,
            code="command_empty",
            message="Step1X command template rendered an empty command.",
        )

    command_blockers = tuple(
        item
        for item in missing_inputs
        if "TEXTURE_STEP1X_RUNTIME_DIR" in item
        or "TEXTURE_STEP1X_EDIT_SCRIPT" in item
        or "TEXTURE_STEP1X_PYTHON" in item
        or "TEXTURE_STEP1X_REQUIRED_EXECUTABLES" in item
    )
    if command_blockers:
        return command, Step1XPreflightIssue(
            category=Step1XPreflightCategory.COMMAND_TEMPLATE,
            code="command_prerequisites_missing",
            message="Step1X command cannot run until command prerequisites are ready.",
            detail={"missing": list(command_blockers)},
        )
    return command, None


def _category_summaries(
    *,
    backend: Step1XBackend,
    config: Step1XBackendConfig,
    original_config: Step1XBackendConfig,
    require_material_anything: bool,
    issues: tuple[Step1XPreflightIssue, ...],
    command: list[str] | None,
    command_issue: Step1XPreflightIssue | None,
    gpu_available: bool | None,
    require_gpu: bool,
) -> dict[str, dict[str, Any]]:
    issues_by_category = {
        category: [issue for issue in issues if issue.category is category]
        for category in Step1XPreflightCategory
    }
    capabilities = backend.capabilities().model_dump(mode="json")
    runtime_info = capabilities.get("external_runtime", {})
    material_anything = capabilities.get("material_anything", {})
    summaries: dict[str, dict[str, Any]] = {
        Step1XPreflightCategory.RUNTIME.value: {
            "ready": not _has_errors(
                issues_by_category[Step1XPreflightCategory.RUNTIME]
            ),
            "runtime_dir": runtime_info.get("runtime_dir"),
            "edit_script": runtime_info.get("edit_script"),
            "runtime_source": runtime_info.get("runtime_source"),
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[Step1XPreflightCategory.RUNTIME]
            ],
        },
        Step1XPreflightCategory.MODEL_CHECKPOINTS.value: {
            "ready": not _has_errors(
                issues_by_category[Step1XPreflightCategory.MODEL_CHECKPOINTS]
            ),
            "model_dir": runtime_info.get("model_dir"),
            "weights_policy": runtime_info.get("weights_policy"),
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[
                    Step1XPreflightCategory.MODEL_CHECKPOINTS
                ]
            ],
        },
        Step1XPreflightCategory.PYTHON_EXECUTABLE.value: {
            "ready": not _has_errors(
                issues_by_category[Step1XPreflightCategory.PYTHON_EXECUTABLE]
            ),
            "python_executable": str(config.python_executable)
            if config.python_executable is not None
            else None,
            "required_executables": runtime_info.get("required_executables", []),
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[
                    Step1XPreflightCategory.PYTHON_EXECUTABLE
                ]
            ],
        },
        Step1XPreflightCategory.MATERIAL_ANYTHING.value: {
            **material_anything,
            "default_enabled": not original_config.skip_material_anything,
            "effective_enabled": not config.skip_material_anything,
            "enabled": not config.skip_material_anything,
            "required": require_material_anything,
            "ready": _material_anything_capability_ready(material_anything)
            and not _has_errors(
                issues_by_category[Step1XPreflightCategory.MATERIAL_ANYTHING]
            ),
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[
                    Step1XPreflightCategory.MATERIAL_ANYTHING
                ]
            ],
        },
        Step1XPreflightCategory.GPU_CUDA.value: {
            "ready": not _has_errors(
                issues_by_category[Step1XPreflightCategory.GPU_CUDA]
            ),
            "required": require_gpu,
            "gpu_available": gpu_available,
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[Step1XPreflightCategory.GPU_CUDA]
            ],
        },
        Step1XPreflightCategory.COMMAND_TEMPLATE.value: {
            "ready": command_issue is None,
            "configured": config.command_template is not None,
            "rendered": command is not None,
            "command_preview": _redact_command_preview(command)
            if command is not None
            else None,
            "issues": [
                issue.to_dict()
                for issue in issues_by_category[
                    Step1XPreflightCategory.COMMAND_TEMPLATE
                ]
            ],
        },
    }
    return summaries


def _has_errors(issues: list[Step1XPreflightIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def _has_category_error(
    issues: list[Step1XPreflightIssue],
    category: Step1XPreflightCategory,
) -> bool:
    return any(
        issue.category is category and issue.severity == "error" for issue in issues
    )


def _material_anything_capability(backend: Step1XBackend) -> dict[str, Any]:
    capability = (
        backend.capabilities()
        .model_dump(mode="json")
        .get(
            "material_anything",
            {},
        )
    )
    return capability if isinstance(capability, dict) else {}


def _material_anything_capability_issue(
    material_anything: dict[str, Any],
    *,
    required: bool,
) -> Step1XPreflightIssue | None:
    if not required or _material_anything_capability_ready(material_anything):
        return None

    missing = [
        str(item) for item in material_anything.get("missing", ()) if str(item).strip()
    ]
    message = (
        "Material Anything readiness is required but capability check is not ready."
    )
    detail: dict[str, Any] = {}
    if missing:
        message = (
            "Material Anything readiness is required but inputs are missing: "
            + "; ".join(missing)
        )
        detail["missing"] = missing
    return Step1XPreflightIssue(
        category=Step1XPreflightCategory.MATERIAL_ANYTHING,
        code="material_anything_unavailable",
        message=message,
        detail=detail,
    )


def _material_anything_capability_ready(material_anything: dict[str, Any]) -> bool:
    if "ready" in material_anything:
        return bool(material_anything["ready"])
    if "available" in material_anything:
        return bool(material_anything["available"])
    return True


def _redact_command_preview(command: list[str]) -> list[str]:
    redacted: list[str] = []
    redact_next = False
    for token in command:
        if redact_next:
            redacted.append("<redacted>")
            redact_next = False
            continue

        name, separator, _value = token.partition("=")
        if separator and _is_sensitive_command_name(name):
            redacted.append(f"{name}=<redacted>")
            continue

        redacted.append(token)
        redact_next = _is_sensitive_command_name(token)
    return redacted


def _is_sensitive_command_name(token: str) -> bool:
    normalized = token.lower().lstrip("-").replace("_", "-")
    exact_names = {
        "api-key",
        "apikey",
        "access-key",
        "secret-key",
        "token",
        "password",
        "passwd",
        "secret",
        "credential",
        "credentials",
    }
    if normalized in exact_names:
        return True
    return any(
        sensitive in normalized
        for sensitive in (
            "api-key",
            "apikey",
            "secret-key",
            "token",
            "password",
            "passwd",
            "secret",
            "credential",
        )
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _require_dict(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data[key]
    if not isinstance(value, dict):
        raise TypeError(f"{key} must be a JSON object")
    return value


def _resolve_local_reference(value: str, base_dir: Path) -> str:
    parsed = urlparse(value)
    if parsed.scheme:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((base_dir / path).resolve())


def _resolve_local_path(value: str, base_dir: Path) -> Path:
    parsed = urlparse(value)
    if parsed.scheme and parsed.scheme != "file":
        raise ValueError(f"source_usd must be a local path or file URI: {value}")
    if parsed.scheme == "file":
        return Path(parsed.path).resolve()
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _config_from_args(args: argparse.Namespace) -> Step1XBackendConfig:
    config = Step1XBackendConfig.from_env()
    required_executables = tuple(
        getattr(args, "required_executable", None) or config.required_executables
    )
    return replace(
        config,
        runtime_dir=getattr(args, "runtime_dir", None) or config.runtime_dir,
        model_dir=getattr(args, "model_dir", None) or config.model_dir,
        cache_dir=getattr(args, "cache_dir", None) or config.cache_dir,
        python_executable=(
            getattr(args, "python_executable", None) or config.python_executable
        ),
        edit_script=getattr(args, "edit_script", None) or config.edit_script,
        command_template=(
            getattr(args, "command_template", None) or config.command_template
        ),
        timeout_sec=getattr(args, "timeout_sec", None) or config.timeout_sec,
        required_executables=required_executables,
    )


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--python-executable", type=Path)
    parser.add_argument("--edit-script", type=Path)
    parser.add_argument("--command-template")
    parser.add_argument("--timeout-sec", type=int)
    parser.add_argument(
        "--required-executable",
        action="append",
        help="Executable that must be available on PATH; may be repeated.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    preflight = subparsers.add_parser(
        "preflight",
        help="Check Step1X runtime readiness without running generation.",
    )
    _add_config_args(preflight)
    preflight.add_argument("--require-gpu", action="store_true")
    preflight.add_argument(
        "--allow-material-anything-disabled",
        action="store_true",
        help="Do not require Material Anything readiness in the preflight result.",
    )

    smoke = subparsers.add_parser(
        "smoke",
        help="Run one real Step1X material creation when explicitly enabled.",
    )
    _add_config_args(smoke)
    smoke.add_argument("--request", type=Path)
    smoke.add_argument("--conditioning", type=Path)
    smoke.add_argument("--output-dir", type=Path)
    smoke.add_argument("--run-real-step1x", action="store_true")
    smoke.add_argument(
        "--allow-gpu-unknown",
        action="store_true",
        help="Do not block smoke solely because nvidia-smi could not confirm a GPU.",
    )

    parser.set_defaults(command="preflight")
    return parser


if __name__ == "__main__":
    sys.exit(main())
