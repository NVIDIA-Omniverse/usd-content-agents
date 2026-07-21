# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Step1X backend adapter for the Texture Variation API.

This module does not import or initialize Step1X at service startup. The
boundary is the ``Step1XRunner`` protocol: internal deployments can use the
source-only runtime under ``apps/texture_gen_step1x_service/internal`` while
public or custom deployments can mount an external Step1X runtime. Unit tests
inject lightweight fakes through the same runner interface.
"""

from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from apps.texture_gen_service_common import (
    BackendCapabilities,
    BackendHealth,
    CreateJobRequest,
    GeneratedTextures,
    GenerationResult,
    MapArtifact,
    TextureGenerationBackend,
    TextureGenerationBackendError,
    TextureTarget,
    local_path_from_file_uri,
)
from apps.texture_gen_service_common.usd_package import (
    ArchiveSizeLimitExceeded,
    extract_usdz_member_to_dir,
    package_member_cache_name,
    parse_package_member_asset_path,
)

_IMAGE_SUFFIXES = {
    ".bmp",
    ".exr",
    ".jpg",
    ".jpeg",
    ".png",
    ".tga",
    ".tif",
    ".tiff",
}
_ALBEDO_INPUTS = {
    "albedo",
    "basecolor",
    "base_color",
    "base_color_texture_file",
    "basecolormap",
    "diffuse",
    "diffusecolor",
    "diffuse_texture",
    "diffuse_texture_file",
}
_NORMAL_INPUTS = {
    "detail_normalmap_texture",
    "normal",
    "normal_texture",
    "normal_texture_file",
    "normalmap",
    "normalmap_texture",
}
_ORM_INPUTS = {
    "arm",
    "ao_roughness_metallic",
    "metallic",
    "metallicroughness",
    "metallic_roughness",
    "occlusion",
    "occlusionroughnessmetallic",
    "orm",
    "orm_texture",
    "ormtexture",
    "roughness",
}
_SECRET_KEY_NAMES = {
    "api_key",
    "apikey",
    "auth_token",
    "token",
    "secret",
    "password",
    "credential",
}
_SECRET_KEY_SUFFIXES = (
    "_api_key",
    "_apikey",
    "_auth_token",
    "_token",
    "_secret",
    "_password",
    "_credential",
)
_GPU_CACHE_TTL_SEC = 60.0
_GPU_CACHE_LOCK = threading.Lock()
_GPU_CACHE_VALUE: bool | None = None
_GPU_CACHE_AT = 0.0
_BUNDLED_RUNTIME_RELATIVE = Path("internal") / "texture_editing_runtime"
_BUNDLED_RUNTIME_SOURCE_PATHS = (
    ("Step1X-3D source", Path("third_party") / "Step1X-3D" / "step1x3d_texture"),
    ("MaterialAnything source", Path("third_party") / "MaterialAnything" / "scripts"),
)
_COMPOSE_RUNTIME_MARKER = ".texture-agent-runtime.json"
_MAX_PACKAGE_ASSET_BYTES = 512 * 1024 * 1024
_MATERIAL_ANYTHING_REQUIRED_MODULES = (
    "kaolin",
    "pytorch3d",
    "torch",
    "torchvision",
    "PIL",
    "numpy",
    "diffusers",
    "transformers",
    "open_clip",
    "cv2",
    "trimesh",
    "xatlas",
    "sklearn",
    "skimage",
    "scipy",
    "matplotlib",
    "imageio",
    "tqdm",
    "cupy",
    "einops",
    "gradio",
    "pkg_resources",
    "pytorch_lightning",
    "omegaconf",
    "pymeshlab",
)


class _SafeFormatDict(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return ""


class Step1XExtractionError(RuntimeError):
    """Raised when a package member cannot be extracted for Step1X input."""


@dataclass(frozen=True)
class Step1XBackendConfig:
    """Configuration for a Step1X runtime."""

    runtime_dir: Path | None = None
    model_dir: Path | None = None
    cache_dir: Path | None = None
    output_dir: Path | None = None
    python_executable: Path | None = None
    edit_script: Path | None = None
    command_template: str | None = None
    timeout_sec: int = 3600
    validate_assets: bool = True
    skip_material_anything: bool = True
    require_upscaler: bool = False
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    required_executables: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_env(cls) -> Step1XBackendConfig:
        """Create config from explicit Step1X service environment variables."""
        runtime_dir = _optional_path("TEXTURE_STEP1X_RUNTIME_DIR")
        if runtime_dir is None:
            runtime_dir = _bundled_runtime_dir()
        return cls(
            runtime_dir=runtime_dir,
            model_dir=_optional_path("TEXTURE_STEP1X_MODEL_DIR"),
            cache_dir=_optional_path("TEXTURE_STEP1X_CACHE_DIR"),
            output_dir=_optional_path("TEXTURE_OUTPUT_DIR"),
            python_executable=_optional_path("TEXTURE_STEP1X_PYTHON"),
            edit_script=_optional_path("TEXTURE_STEP1X_EDIT_SCRIPT"),
            command_template=_optional_str("TEXTURE_STEP1X_COMMAND_TEMPLATE"),
            timeout_sec=_optional_int("TEXTURE_STEP1X_TIMEOUT_SEC", default=3600),
            validate_assets=_optional_bool("TEXTURE_STEP1X_VALIDATE_ASSETS", True),
            skip_material_anything=_optional_bool("TEXTURE_STEP1X_SKIP_MA", True),
            require_upscaler=_optional_bool(
                "TEXTURE_STEP1X_REQUIRE_UPSCALER",
                False,
            ),
            extra_args=tuple(
                shlex.split(os.environ.get("TEXTURE_STEP1X_EXTRA_ARGS", ""))
            ),
            required_executables=_optional_executables(
                "TEXTURE_STEP1X_REQUIRED_EXECUTABLES",
                default=("uv",),
            ),
        )


@dataclass(frozen=True)
class Step1XScopeInfo:
    """USD target-scope data resolved before invoking Step1X."""

    source_asset_path: Path
    material_path: str | None = None
    material_name: str | None = None
    prim_paths: tuple[str, ...] = ()
    source_albedo_path: Path | None = None
    source_normal_path: Path | None = None
    source_orm_path: Path | None = None
    diagnostics: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class Step1XRunRequest:
    """Normalized request passed to an external Step1X runner."""

    prompt: str | None
    seed: int | None
    strength: float
    texture_size: int | None
    source_asset_uri: str
    source_asset_path: Path | None
    source_albedo_path: Path | None
    reference_image_uris: tuple[str, ...]
    turntable_video_uri: str | None
    multiview_image_uris: tuple[str, ...]
    target: TextureTarget | None
    scope: Step1XScopeInfo | None
    job_id: str
    output_dir: Path
    runtime_dir: Path | None
    model_dir: Path | None
    cache_dir: Path | None
    custom_parameters: dict[str, Any]


@dataclass(frozen=True)
class Step1XRunResult:
    """Raw result returned by a Step1X runner implementation."""

    albedo_uri: str
    variant_asset_uri: str | None = None
    normal_uri: str | None = None
    orm_uri: str | None = None
    width: int | None = None
    height: int | None = None
    metadata: dict[str, Any] | None = None
    auxiliary_artifacts: dict[str, Any] | None = None


def _scope_metadata(scope: Step1XScopeInfo) -> dict[str, Any]:
    metadata = {
        "material_path": scope.material_path,
        "material_name": scope.material_name,
        "prim_paths": list(scope.prim_paths),
        "source_albedo_path": (
            str(scope.source_albedo_path) if scope.source_albedo_path else None
        ),
        "source_normal_path": (
            str(scope.source_normal_path) if scope.source_normal_path else None
        ),
        "source_orm_path": str(scope.source_orm_path)
        if scope.source_orm_path
        else None,
    }
    if scope.diagnostics:
        metadata["diagnostics"] = list(scope.diagnostics)
    return metadata


class Step1XRunner(Protocol):
    """Mockable seam for invoking an external Step1X runtime."""

    def run(
        self,
        request: Step1XRunRequest,
        *,
        cancel_event: threading.Event,
    ) -> Step1XRunResult:
        """Generate texture maps using Step1X."""
        ...


class ExternalStep1XRunner:
    """Invoke an external Step1X-compatible CLI without vendoring it."""

    def __init__(self, config: Step1XBackendConfig) -> None:
        self.config = config

    def run(
        self,
        request: Step1XRunRequest,
        *,
        cancel_event: threading.Event,
    ) -> Step1XRunResult:
        if cancel_event.is_set():
            raise RuntimeError("Step1X job was cancelled before launch.")
        if not request.prompt or not request.prompt.strip():
            raise RuntimeError("STEP1X_PROMPT_MISSING: text_prompt is required.")

        source_asset_path = request.source_asset_path or _local_path_from_uri(
            request.source_asset_uri
        )
        if source_asset_path is None:
            raise RuntimeError(
                "STEP1X_ASSET_UNREACHABLE: only local file:// source_asset_uri "
                "values are supported by the default runner."
            )

        request.output_dir.mkdir(parents=True, exist_ok=True)
        command_source_asset_path = (
            _prepare_scoped_usd(request.scope, request.output_dir)
            if request.scope is not None
            else source_asset_path
        )
        stdout_path = request.output_dir / "step1x_stdout.log"
        stderr_path = request.output_dir / "step1x_stderr.log"
        cmd = self._build_command(request, command_source_asset_path)
        start = time.monotonic()

        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            process = subprocess.Popen(
                cmd,
                cwd=str(self._working_dir()),
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            while True:
                if cancel_event.is_set():
                    _terminate_process(process)
                    raise RuntimeError("Step1X job was cancelled while running.")
                remaining = self.config.timeout_sec - (time.monotonic() - start)
                if remaining <= 0:
                    _terminate_process(process)
                    raise RuntimeError(
                        "STEP1X_TIMEOUT: external Step1X command exceeded "
                        f"{self.config.timeout_sec} seconds."
                    )
                try:
                    process.wait(timeout=min(0.2, remaining))
                    break
                except subprocess.TimeoutExpired:
                    continue

        elapsed = time.monotonic() - start
        if process.returncode != 0:
            stderr_tail = _tail_text(stderr_path)
            raise RuntimeError(
                "STEP1X_COMMAND_FAILED: external Step1X command exited with "
                f"{process.returncode}.\n{stderr_tail}"
            )

        albedo_path = _first_existing(
            request.output_dir,
            (
                "final_albedo.png",
                "edited_albedo.png",
                "albedo.png",
                "texture.png",
            ),
        )
        if albedo_path is None:
            raise RuntimeError(
                "STEP1X_OUTPUT_MISSING: external Step1X command did not produce "
                "a final_albedo.png, edited_albedo.png, albedo.png, or texture.png."
            )
        width, height = _validate_albedo(albedo_path)

        orm_path = _first_existing(
            request.output_dir,
            ("final_orm.png", "orm.png", "packed_orm.png"),
        )
        normal_path = _first_existing(
            request.output_dir,
            ("final_normal.png", "normal.png", "normal_map.png"),
        )
        variant_asset_path = _first_matching(request.output_dir, "edited_*.usd*")

        metadata: dict[str, Any] = {
            "runner": "external_step1x_cli",
            "command": _redact_command(cmd),
            "elapsed_sec": round(elapsed, 3),
            "stdout_log": stdout_path.as_uri(),
            "stderr_log": stderr_path.as_uri(),
            "command_source_asset_uri": command_source_asset_path.as_uri(),
        }
        if request.scope is not None:
            metadata["scoped_source_asset_uri"] = command_source_asset_path.as_uri()
            metadata["scope"] = _scope_metadata(request.scope)

        return Step1XRunResult(
            albedo_uri=albedo_path.as_uri(),
            variant_asset_uri=variant_asset_path.as_uri()
            if variant_asset_path is not None
            else None,
            normal_uri=normal_path.as_uri() if normal_path is not None else None,
            orm_uri=orm_path.as_uri() if orm_path is not None else None,
            width=width,
            height=height,
            metadata=metadata,
            auxiliary_artifacts={
                "stdout_log": stdout_path.as_uri(),
                "stderr_log": stderr_path.as_uri(),
            },
        )

    def _build_command(
        self,
        request: Step1XRunRequest,
        source_asset_path: Path,
    ) -> list[str]:
        if self.config.command_template:
            mapping = {
                "source_asset": str(source_asset_path),
                "source_asset_uri": source_asset_path.as_uri(),
                "source_albedo": str(request.source_albedo_path or ""),
                "prompt": request.prompt or "",
                "output_dir": str(request.output_dir),
                "strength": str(request.strength),
                "seed": str(_default_seed(request.seed)),
                "texture_size": str(request.texture_size or 1536),
                "runtime_dir": str(request.runtime_dir or ""),
                "model_dir": str(request.model_dir or ""),
                "cache_dir": str(request.cache_dir or ""),
                "target_material_path": (
                    request.scope.material_path if request.scope else ""
                ),
                "target_material_name": (
                    request.scope.material_name if request.scope else ""
                ),
                "target_prim_paths": ",".join(request.scope.prim_paths)
                if request.scope
                else "",
                "reference_image_uris": ",".join(request.reference_image_uris),
                "turntable_video_uri": request.turntable_video_uri or "",
                "multiview_image_uris": ",".join(request.multiview_image_uris),
            }
            return [
                substituted
                for token in shlex.split(self.config.command_template)
                if (substituted := token.format_map(_SafeFormatDict(mapping)))
            ]

        script_path = self._edit_script()
        python_executable = self._python_executable()
        cmd = [
            str(python_executable),
            str(script_path),
            "--usd",
            str(source_asset_path),
            "--prompt",
            request.prompt or "",
            "--output",
            str(request.output_dir),
            "--strength",
            str(request.strength),
            "--seed",
            str(_default_seed(request.seed)),
            "--resolution",
            str(request.texture_size or 1536),
        ]

        custom = request.custom_parameters
        for key, cli_name in (
            ("steps", "--steps"),
            ("guidance", "--guidance"),
            ("ma_steps", "--ma-steps"),
            ("gpu", "--gpu"),
        ):
            value = custom.get(key)
            if value is not None:
                cmd.extend([cli_name, str(value)])

        skip_ma = _coerce_bool(
            custom.get("skip_material_anything"),
            default=self.config.skip_material_anything,
        )
        if skip_ma:
            cmd.append("--skip-ma")
        for key, cli_name in (
            ("upscale", "--upscale"),
            ("debug", "--debug"),
        ):
            if _coerce_bool(custom.get(key), default=False):
                cmd.append(cli_name)
        cmd.extend(self.config.extra_args)
        return cmd

    def _working_dir(self) -> Path:
        if self.config.runtime_dir is not None:
            return self.config.runtime_dir
        if self.config.edit_script is not None:
            return self.config.edit_script.parent
        return Path.cwd()

    def _edit_script(self) -> Path:
        if self.config.edit_script is not None:
            return self.config.edit_script
        if self.config.runtime_dir is None:
            raise RuntimeError("TEXTURE_STEP1X_RUNTIME_DIR is required.")
        return self.config.runtime_dir / "edit_texture.py"

    def _python_executable(self) -> Path:
        if self.config.python_executable is not None:
            return self.config.python_executable
        if self.config.runtime_dir is not None:
            runtime_python = self.config.runtime_dir / ".venv_gen" / "bin" / "python"
            if runtime_python.exists():
                return runtime_python
            runtime_python = self.config.runtime_dir / ".venv" / "bin" / "python"
            if runtime_python.exists():
                return runtime_python
        return Path(sys.executable)


class Step1XBackend(TextureGenerationBackend):
    """Texture Variation API backend adapter for an external Step1X runtime."""

    def __init__(
        self,
        *,
        config: Step1XBackendConfig | None = None,
        runner: Step1XRunner | None = None,
    ) -> None:
        self.config = config or Step1XBackendConfig.from_env()
        self.runner = runner or ExternalStep1XRunner(self.config)

    @property
    def name(self) -> str:
        return "step1x"

    def capabilities(self) -> BackendCapabilities:
        template_conditioning = self.config.command_template is not None
        return BackendCapabilities(
            image_conditioning=template_conditioning,
            multiview=template_conditioning,
            normal_map=False,
            orm=False,
            masks=False,
            coverage=False,
            geometry_output="source_asset",
            external_runtime=self._external_runtime_info(),
            material_anything=self._material_anything_info(),
            upscaler=self._upscaler_info(),
        )

    def health(self) -> BackendHealth:
        missing = self._missing_runtime_inputs()
        if missing:
            return BackendHealth(
                status="not_ready",
                ready=False,
                warmup_complete=False,
                gpu_available=_detect_gpu_available(),
                capabilities=self.capabilities(),
                error="Missing Step1X runtime configuration: " + ", ".join(missing),
            )
        return BackendHealth(
            status="healthy",
            ready=True,
            warmup_complete=True,
            gpu_available=_detect_gpu_available(),
            capabilities=self.capabilities(),
        )

    def generate(
        self,
        request: CreateJobRequest,
        *,
        job_id: str,
        output_dir: Path,
        cancel_event: threading.Event,
    ) -> GenerationResult:
        self._raise_if_requested_features_unavailable(request, job_id)
        try:
            scope = (
                _inspect_step1x_scope(
                    request.source_asset_uri,
                    request.target,
                    output_dir=output_dir,
                    texture_size=request.configuration.texture_size,
                )
                if self.config.validate_assets
                else None
            )
        except RuntimeError as exc:
            code = _step1x_error_code(str(exc))
            if code is not None:
                raise TextureGenerationBackendError(
                    str(exc),
                    result=self._failure_result(
                        request,
                        job_id,
                        code=code,
                        message=str(exc),
                    ),
                ) from exc
            raise
        source_asset_path = (
            scope.source_asset_path
            if scope is not None
            else _local_path_from_uri(request.source_asset_uri, require_exists=False)
        )
        run_request = Step1XRunRequest(
            prompt=request.conditioning.text_prompt,
            seed=request.configuration.seed,
            strength=request.configuration.strength,
            texture_size=request.configuration.texture_size,
            source_asset_uri=request.source_asset_uri,
            source_asset_path=source_asset_path,
            source_albedo_path=scope.source_albedo_path if scope is not None else None,
            reference_image_uris=tuple(request.conditioning.reference_image_uris),
            turntable_video_uri=request.conditioning.turntable_video_uri,
            multiview_image_uris=tuple(request.conditioning.multiview_image_uris),
            target=request.target,
            scope=scope,
            job_id=job_id,
            output_dir=output_dir,
            runtime_dir=self.config.runtime_dir,
            model_dir=self.config.model_dir,
            cache_dir=self.config.cache_dir,
            custom_parameters=dict(request.configuration.custom_parameters),
        )
        try:
            raw_result = self.runner.run(run_request, cancel_event=cancel_event)
        except RuntimeError as exc:
            code = _step1x_error_code(str(exc))
            if code is not None:
                raise TextureGenerationBackendError(
                    str(exc),
                    result=self._failure_result(
                        request,
                        job_id,
                        code=code,
                        message=str(exc),
                    ),
                ) from exc
            raise
        if scope is not None:
            metadata = dict(raw_result.metadata or {})
            existing_scope = metadata.get("scope")
            if isinstance(existing_scope, dict):
                scope_metadata = {**existing_scope, **_scope_metadata(scope)}
            else:
                scope_metadata = _scope_metadata(scope)
            metadata["scope"] = scope_metadata
            raw_result = replace(raw_result, metadata=metadata)
            raw_result = _preserve_source_normal_if_missing(
                raw_result,
                scope=scope,
                output_dir=output_dir,
            )
        return self._to_generation_result(request, job_id, raw_result)

    def _raise_if_requested_features_unavailable(
        self,
        request: CreateJobRequest,
        job_id: str,
    ) -> None:
        if self.config.command_template:
            return

        custom = request.configuration.custom_parameters
        skip_ma = _coerce_bool(
            custom.get("skip_material_anything"),
            default=self.config.skip_material_anything,
        )
        if not skip_ma:
            missing = self._missing_material_anything_inputs()
            if missing:
                message = (
                    "STEP1X_MATERIAL_ANYTHING_UNAVAILABLE: request set "
                    "skip_material_anything=false, but Material Anything inputs "
                    f"are not ready: {', '.join(missing)}"
                )
                raise TextureGenerationBackendError(
                    message,
                    result=self._failure_result(
                        request,
                        job_id,
                        code="STEP1X_MATERIAL_ANYTHING_UNAVAILABLE",
                        message=message,
                    ),
                )

        if _coerce_bool(custom.get("upscale"), default=False):
            upscaler_info = self._upscaler_info()
            if not upscaler_info["ready"]:
                missing = ", ".join(upscaler_info["missing"])
                message = (
                    "STEP1X_UPSCALER_UNAVAILABLE: request set upscale=true, "
                    f"but upscaler inputs are not ready: {missing}"
                )
                raise TextureGenerationBackendError(
                    message,
                    result=self._failure_result(
                        request,
                        job_id,
                        code="STEP1X_UPSCALER_UNAVAILABLE",
                        message=message,
                    ),
                )

    def _failure_result(
        self,
        request: CreateJobRequest,
        job_id: str,
        *,
        code: str,
        message: str,
    ) -> GenerationResult:
        metadata: dict[str, Any] = {"backend_name": self.name}
        if request.target is not None:
            metadata["target"] = request.target.model_dump(exclude_none=True)
        return GenerationResult(
            variant_asset_uri=request.source_asset_uri,
            variant_name=request.configuration.variant_name or job_id,
            generated_textures=GeneratedTextures(),
            maps={},
            metadata=metadata,
            diagnostics=[
                {
                    "code": code,
                    "severity": "error",
                    "message": message,
                }
            ],
        )

    def _missing_runtime_inputs(self) -> list[str]:
        missing: list[str] = []
        if self.config.command_template:
            optional_paths = (
                (
                    "TEXTURE_STEP1X_RUNTIME_DIR",
                    self.config.runtime_dir,
                    True,
                    True,
                ),
                ("TEXTURE_STEP1X_MODEL_DIR", self.config.model_dir, True, False),
            )
            for label, path, require_read, require_execute in optional_paths:
                issue = _path_readiness_issue(
                    label,
                    path,
                    require_read=require_read,
                    require_execute=require_execute,
                    optional=True,
                )
                if issue is not None:
                    missing.append(issue)
            missing.extend(self._missing_required_executables())
            return missing

        required_paths = (("TEXTURE_STEP1X_RUNTIME_DIR", self.config.runtime_dir),)
        for label, path in required_paths:
            issue = _path_readiness_issue(
                label,
                path,
                require_read=True,
                require_execute=True,
            )
            if issue is not None:
                missing.append(issue)
        edit_script = self._configured_edit_script()
        issue = _path_readiness_issue(
            "TEXTURE_STEP1X_EDIT_SCRIPT",
            edit_script,
            require_read=True,
        )
        if issue is not None:
            missing.append(issue)
        for label, path in (
            ("TEXTURE_STEP1X_MODEL_DIR", self.config.model_dir),
            ("TEXTURE_STEP1X_PYTHON", self.config.python_executable),
        ):
            issue = _path_readiness_issue(
                label,
                path,
                require_read=label == "TEXTURE_STEP1X_MODEL_DIR",
                require_execute=label == "TEXTURE_STEP1X_PYTHON",
                optional=True,
            )
            if issue is not None:
                missing.append(issue)
        missing.extend(self._missing_required_executables())
        if self._runtime_source() == "repo_internal" and self.config.runtime_dir:
            for label, relative_path in _BUNDLED_RUNTIME_SOURCE_PATHS:
                source_path = self.config.runtime_dir / relative_path
                issue = _path_readiness_issue(label, source_path, require_read=True)
                if issue is not None:
                    missing.append(f"{issue}; run setup_env.sh")
        if not self.config.skip_material_anything:
            missing.extend(self._missing_material_anything_inputs())
        if self.config.require_upscaler:
            upscaler_info = self._upscaler_info()
            if not upscaler_info["ready"]:
                missing.extend(f"upscaler {item}" for item in upscaler_info["missing"])
        return missing

    def _missing_required_executables(self) -> list[str]:
        missing: list[str] = []
        for executable in _required_executable_status(self.config.required_executables):
            if not executable["available"]:
                missing.append(
                    "TEXTURE_STEP1X_REQUIRED_EXECUTABLES "
                    f"(not found on PATH: {executable['name']})"
                )
        return missing

    def _configured_edit_script(self) -> Path | None:
        if self.config.edit_script is not None:
            return self.config.edit_script
        if self.config.runtime_dir is not None:
            return self.config.runtime_dir / "edit_texture.py"
        return None

    def _external_runtime_info(self) -> dict[str, Any]:
        edit_script = self._configured_edit_script()
        runtime_source = self._runtime_source()
        return {
            "api_service": "repo_owned",
            "step1x_runtime": runtime_source,
            "runtime_source": runtime_source,
            "runtime_dir": str(self.config.runtime_dir)
            if self.config.runtime_dir is not None
            else None,
            "edit_script": str(edit_script) if edit_script is not None else None,
            "edit_script_configured": self.config.edit_script is not None,
            "command_template_configured": self.config.command_template is not None,
            "model_dir": str(self.config.model_dir)
            if self.config.model_dir is not None
            else None,
            "cache_dir": str(self.config.cache_dir)
            if self.config.cache_dir is not None
            else None,
            "validate_assets": self.config.validate_assets,
            "skip_material_anything_default": self.config.skip_material_anything,
            "require_upscaler": self.config.require_upscaler,
            "weights_policy": "downloadable_not_committed",
            "required_executables": _required_executable_status(
                self.config.required_executables
            ),
        }

    def _material_anything_info(self) -> dict[str, Any]:
        paths = _material_anything_paths(self.config.runtime_dir)
        missing = self._missing_material_anything_inputs()
        return {
            "enabled_by_default": not self.config.skip_material_anything,
            "ready": not missing,
            "missing": missing,
            "paths": {label: str(path) for label, path in paths.items()},
            "required_assets": [
                "third_party/MaterialAnything/pretrained_models/material_estimator",
                "third_party/MaterialAnything/pretrained_models/material_refiner",
                "third_party/MaterialAnything/models/ControlNet/models/control_sd15_depth.pth",
            ],
        }

    def _upscaler_info(self) -> dict[str, Any]:
        paths = _upscaler_paths(self.config.runtime_dir)
        missing = _missing_upscaler_inputs(
            self.config.runtime_dir,
            python_executable=self._runtime_python_executable(),
        )
        can_auto_download = _upscaler_auto_download_writable(self.config.runtime_dir)
        return {
            "backend": _upscaler_backend(),
            "available_backends": ["swin2sr", "auto", "ncnn-vulkan"],
            "required_for_ready": self.config.require_upscaler,
            "ready": not missing or can_auto_download,
            "missing": missing,
            "auto_download_writable": can_auto_download,
            "model_policy": "huggingface_cache_downloadable_not_committed",
            "swin2sr_models": {
                "x2": "caidas/swin2SR-classical-sr-x2-64",
                "x4": "caidas/swin2SR-realworld-sr-x4-64-bsrgan-psnr",
            },
            "paths": {label: str(path) for label, path in paths.items()},
        }

    def _missing_material_anything_inputs(self) -> list[str]:
        return _missing_material_anything_inputs(
            self.config.runtime_dir,
            python_executable=self._runtime_python_executable(),
        )

    def _runtime_python_executable(self) -> Path | None:
        if self.config.python_executable is not None:
            return self.config.python_executable
        if self.config.runtime_dir is None:
            return None
        for relative_path in (
            Path(".venv_gen") / "bin" / "python",
            Path(".venv") / "bin" / "python",
        ):
            python_path = self.config.runtime_dir / relative_path
            if python_path.exists():
                return python_path
        return None

    def _runtime_source(self) -> str:
        runtime_dir = self.config.runtime_dir
        if runtime_dir is not None and _is_compose_managed_runtime_marker(
            runtime_dir / _COMPOSE_RUNTIME_MARKER
        ):
            return "compose_managed"
        bundled_runtime = _bundled_runtime_dir()
        if runtime_dir is not None and bundled_runtime is not None:
            try:
                if runtime_dir.resolve() == bundled_runtime.resolve():
                    return "repo_internal"
            except OSError:
                pass
        return "operator_mounted"

    def _to_generation_result(
        self,
        request: CreateJobRequest,
        job_id: str,
        raw_result: Step1XRunResult,
    ) -> GenerationResult:
        degraded_channels = [
            name
            for name, uri in (
                ("normal", raw_result.normal_uri),
                ("orm", raw_result.orm_uri),
            )
            if uri is None
        ]
        metadata = dict(raw_result.metadata or {})
        metadata.setdefault("backend_name", self.name)
        if request.target is not None:
            metadata.setdefault(
                "target",
                request.target.model_dump(exclude_none=True),
            )
        if degraded_channels:
            metadata["degraded_channels"] = degraded_channels

        maps = {
            "albedo": MapArtifact(
                uri=raw_result.albedo_uri,
                width=raw_result.width,
                height=raw_result.height,
                colorspace="srgb",
            )
        }
        if raw_result.normal_uri is not None:
            maps["normal"] = MapArtifact(
                uri=raw_result.normal_uri,
                width=raw_result.width,
                height=raw_result.height,
                colorspace="linear",
            )
        if raw_result.orm_uri is not None:
            maps["orm"] = MapArtifact(
                uri=raw_result.orm_uri,
                width=raw_result.width,
                height=raw_result.height,
                colorspace="linear",
                packing="occlusion_roughness_metallic",
            )

        diagnostics: list[dict[str, Any]] = []
        scope_diagnostics = metadata.get("scope", {}).get("diagnostics")
        if isinstance(scope_diagnostics, list):
            diagnostics.extend(
                item for item in scope_diagnostics if isinstance(item, dict)
            )
        if degraded_channels:
            diagnostics.append(
                {
                    "code": "STEP1X_MAPS_DEGRADED",
                    "severity": "warning",
                    "message": "Step1X output omitted optional PBR maps.",
                    "channels": degraded_channels,
                }
            )

        return GenerationResult(
            variant_asset_uri=raw_result.variant_asset_uri or request.source_asset_uri,
            variant_name=request.configuration.variant_name or job_id,
            generated_textures=GeneratedTextures(
                albedo=raw_result.albedo_uri,
                normal=raw_result.normal_uri,
                orm=raw_result.orm_uri,
            ),
            maps=maps,
            auxiliary_artifacts=dict(raw_result.auxiliary_artifacts or {}),
            metadata=metadata,
            diagnostics=diagnostics,
        )


def _preserve_source_normal_if_missing(
    raw_result: Step1XRunResult,
    *,
    scope: Step1XScopeInfo,
    output_dir: Path,
) -> Step1XRunResult:
    """Return the source normal as the generated normal when editing preserves UVs."""
    if raw_result.normal_uri is not None or scope.source_normal_path is None:
        return raw_result

    source_normal = scope.source_normal_path
    if not source_normal.is_file():
        return raw_result

    preserved_normal = output_dir / "final_normal.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    target_size = (
        (raw_result.width, raw_result.height)
        if isinstance(raw_result.width, int)
        and isinstance(raw_result.height, int)
        and raw_result.width > 0
        and raw_result.height > 0
        else None
    )
    try:
        _write_preserved_normal_png(
            source_normal,
            preserved_normal,
            size=target_size,
        )
    except (OSError, ValueError):
        return raw_result

    metadata = dict(raw_result.metadata or {})
    metadata.setdefault("preserved_channels", [])
    preserved_channels = metadata["preserved_channels"]
    if isinstance(preserved_channels, list) and "normal" not in preserved_channels:
        preserved_channels.append("normal")
    metadata["normal_source"] = "source_preserved"
    metadata["source_normal_uri"] = source_normal.as_uri()

    auxiliary_artifacts = dict(raw_result.auxiliary_artifacts or {})
    auxiliary_artifacts["preserved_normal"] = preserved_normal.as_uri()

    return replace(
        raw_result,
        normal_uri=preserved_normal.as_uri(),
        metadata=metadata,
        auxiliary_artifacts=auxiliary_artifacts,
    )


def _write_preserved_normal_png(
    source_normal: Path,
    output_path: Path,
    *,
    size: tuple[int, int] | None,
) -> None:
    """Convert a preserved source normal into the service PNG/size contract."""
    from PIL import Image

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with Image.open(source_normal) as image:
            normal = image.convert("RGB")
            if size is None:
                size = normal.size
            if size[0] <= 0 or size[1] <= 0:
                raise ValueError(f"invalid normal size: {size}")
            if normal.size != size:
                resampling = getattr(Image, "Resampling", Image).LANCZOS
                normal = normal.resize(size, resampling)
            normal.save(tmp_path, format="PNG")
        tmp_path.replace(output_path)
    finally:
        with suppress(OSError):
            tmp_path.unlink()


def _optional_path(env_name: str) -> Path | None:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    return Path(value).expanduser()


def _bundled_runtime_dir() -> Path | None:
    runtime_dir = Path(__file__).parent / _BUNDLED_RUNTIME_RELATIVE
    if (runtime_dir / "edit_texture.py").exists():
        return runtime_dir
    return None


def _optional_str(env_name: str) -> str | None:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return None
    return value.strip()


def _optional_int(env_name: str, *, default: int) -> int:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _optional_bool(env_name: str, default: bool) -> bool:
    value = os.environ.get(env_name)
    if value is None or not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return bool(value)
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"1", "true", "yes", "on"}:
            return True
        if stripped in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _optional_executables(
    env_name: str, *, default: tuple[str, ...]
) -> tuple[str, ...]:
    value = os.environ.get(env_name)
    if value is None:
        return default
    if not value.strip():
        return ()
    return tuple(
        executable
        for executable in shlex.split(value.replace(",", " "))
        if executable.strip()
    )


def _required_executable_status(
    executables: tuple[str, ...],
) -> list[dict[str, Any]]:
    status: list[dict[str, Any]] = []
    for executable in executables:
        name = executable.strip()
        if not name:
            continue
        resolved = shutil.which(name)
        status.append(
            {
                "name": name,
                "available": resolved is not None,
                "path": resolved,
            }
        )
    return status


def _is_compose_managed_runtime_marker(marker_path: Path) -> bool:
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return marker.get("runtime_source") == "compose_managed"


def _detect_gpu_available() -> bool | None:
    global _GPU_CACHE_AT, _GPU_CACHE_VALUE
    now = time.monotonic()
    with _GPU_CACHE_LOCK:
        if now - _GPU_CACHE_AT <= _GPU_CACHE_TTL_SEC:
            return _GPU_CACHE_VALUE
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        value = None
    else:
        value = result.returncode == 0 and bool(result.stdout.strip())
    with _GPU_CACHE_LOCK:
        _GPU_CACHE_VALUE = value
        _GPU_CACHE_AT = now
    return value


def _material_anything_paths(runtime_dir: Path | None) -> dict[str, Path]:
    if runtime_dir is None:
        return {}
    ma_dir = runtime_dir / "third_party" / "MaterialAnything"
    return {
        "source": ma_dir / "scripts" / "generate_texture_pbr_3d.py",
        "material_estimator": ma_dir / "pretrained_models" / "material_estimator",
        "material_refiner": ma_dir / "pretrained_models" / "material_refiner",
        "controlnet_depth": ma_dir
        / "models"
        / "ControlNet"
        / "models"
        / "control_sd15_depth.pth",
    }


def _path_readiness_issue(
    label: str,
    path: Path | None,
    *,
    require_read: bool = False,
    require_execute: bool = False,
    require_directory: bool = False,
    optional: bool = False,
) -> str | None:
    if path is None:
        return None if optional else label
    try:
        exists = path.exists()
    except OSError as exc:
        return f"{label} (not accessible: {path}: {_format_os_error(exc)})"
    if not exists:
        return f"{label} (not found: {path})"
    if require_directory:
        try:
            is_directory = path.is_dir()
        except OSError as exc:
            return f"{label} (not accessible: {path}: {_format_os_error(exc)})"
        if not is_directory:
            return f"{label} (not a directory: {path})"

    access_issues: list[str] = []
    if require_read and not os.access(path, os.R_OK):
        access_issues.append("readable")
    if require_execute and not os.access(path, os.X_OK):
        access_issues.append("executable")
    if access_issues:
        return f"{label} (not {'/'.join(access_issues)}: {path})"
    return None


def _format_os_error(exc: OSError) -> str:
    return exc.strerror or str(exc)


def _missing_material_anything_inputs(
    runtime_dir: Path | None,
    *,
    python_executable: Path | None = None,
) -> list[str]:
    if runtime_dir is None:
        return ["TEXTURE_STEP1X_RUNTIME_DIR (required for Material Anything)"]
    missing: list[str] = []
    for label, path in _material_anything_paths(runtime_dir).items():
        issue = _path_readiness_issue(
            f"Material Anything {label}",
            path,
            require_read=True,
        )
        if issue is not None:
            missing.append(issue)
    if python_executable is not None:
        missing.extend(
            f"Material Anything {item}"
            for item in _missing_python_modules_in_runtime(
                python_executable,
                _MATERIAL_ANYTHING_REQUIRED_MODULES,
            )
        )
    return missing


def _upscaler_paths(runtime_dir: Path | None) -> dict[str, Path]:
    if runtime_dir is None:
        return {}
    bin_dir = runtime_dir / "bin"
    return {
        "module": runtime_dir / "src" / "texture_edit" / "upscaler.py",
        "ncnn_binary": bin_dir / "realesrgan-ncnn-vulkan",
        "ncnn_models": bin_dir / "models",
    }


def _missing_upscaler_inputs(
    runtime_dir: Path | None,
    *,
    python_executable: Path | None = None,
) -> list[str]:
    if runtime_dir is None:
        return ["TEXTURE_STEP1X_RUNTIME_DIR (required for upscaler)"]
    missing: list[str] = []
    paths = _upscaler_paths(runtime_dir)
    issue = _path_readiness_issue("module", paths["module"], require_read=True)
    if issue is not None:
        missing.append(issue)
    backend = _upscaler_backend()
    if backend not in _VALID_UPSCALER_BACKENDS:
        missing.append(
            "backend "
            f"(unsupported TEXTURE_UPSCALER_BACKEND={backend!r}; "
            "expected auto, swin2sr, or ncnn-vulkan variants)"
        )
        return missing
    if backend in {"swin2sr", "swin2sr-pytorch", "transformers"}:
        missing.extend(_missing_swin2sr_upscaler_inputs(python_executable))
    elif backend in {"ncnn", "ncnn-vulkan", "vulkan"}:
        missing.extend(_missing_ncnn_upscaler_inputs(paths))
    else:
        swin2sr_missing = _missing_swin2sr_upscaler_inputs(python_executable)
        ncnn_missing = _missing_ncnn_upscaler_inputs(paths)
        if swin2sr_missing and ncnn_missing:
            missing.extend([f"swin2sr {item}" for item in swin2sr_missing])
            missing.extend([f"ncnn-vulkan {item}" for item in ncnn_missing])
    return missing


def _missing_swin2sr_upscaler_inputs(
    python_executable: Path | None = None,
) -> list[str]:
    required_modules = ("torch", "transformers", "PIL", "numpy")
    if python_executable is not None:
        return _missing_python_modules_in_runtime(python_executable, required_modules)
    return [
        f"python module {module_name} (not importable)"
        for module_name in required_modules
        if find_spec(module_name) is None
    ]


def _missing_python_modules_in_runtime(
    python_executable: Path,
    module_names: tuple[str, ...],
) -> list[str]:
    issue = _path_readiness_issue(
        "TEXTURE_STEP1X_PYTHON",
        python_executable,
        require_execute=True,
    )
    if issue is not None:
        return [issue]
    script = (
        "import importlib, json\n"
        f"modules = {list(module_names)!r}\n"
        "missing = []\n"
        "for name in modules:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception:\n"
        "        missing.append(name)\n"
        "print('WU_MISSING_MODULES_JSON=' + json.dumps(missing))\n"
    )
    timeout_sec = _runtime_module_probe_timeout_sec()
    try:
        result = subprocess.run(
            [str(python_executable), "-c", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except subprocess.TimeoutExpired as exc:
        timeout = (
            f"{exc.timeout:g}s"
            if isinstance(exc.timeout, int | float)
            else "unknown timeout"
        )
        return [f"runtime python module probe failed (timed out after {timeout})"]
    except OSError as exc:
        return [f"runtime python module probe failed ({_format_os_error(exc)})"]
    except subprocess.SubprocessError as exc:
        return [f"runtime python module probe failed ({exc})"]
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        if message:
            return [f"runtime python module probe failed ({message})"]
        return [f"runtime python module probe failed (exit {result.returncode})"]
    prefix = "WU_MISSING_MODULES_JSON="
    json_payload = None
    for line in reversed((result.stdout or "").splitlines()):
        if line.startswith(prefix):
            json_payload = line[len(prefix) :]
            break
    if json_payload is None:
        return ["runtime python module probe failed (missing module probe output)"]
    try:
        missing = json.loads(json_payload)
    except json.JSONDecodeError as exc:
        return [f"runtime python module probe failed ({exc})"]
    if not isinstance(missing, list):
        return ["runtime python module probe failed (unexpected output)"]
    return [f"python module {name} (not importable)" for name in missing]


def _runtime_module_probe_timeout_sec() -> float:
    for env_name in (
        "TEXTURE_STEP1X_RUNTIME_MODULE_PROBE_TIMEOUT_SEC",
        "TEXTURE_STEP1X_PREFLIGHT_TIMEOUT_SEC",
    ):
        value = os.environ.get(env_name)
        if value is None or not value.strip():
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return 30.0


def _missing_ncnn_upscaler_inputs(paths: dict[str, Path]) -> list[str]:
    missing: list[str] = []
    issue = _path_readiness_issue(
        "binary",
        paths["ncnn_binary"],
        require_execute=True,
    )
    if issue is not None:
        missing.append(issue)
    issue = _path_readiness_issue(
        "models",
        paths["ncnn_models"],
        require_read=True,
        require_directory=True,
    )
    if issue is not None:
        missing.append(issue)
    return missing


def _upscaler_backend() -> str:
    value = os.environ.get("TEXTURE_UPSCALER_BACKEND")
    if value is not None and value.strip():
        return value.strip().lower()
    legacy = os.environ.get("TEXTURE_REALESRGAN_BACKEND")
    if legacy is not None and legacy.strip().lower() in {
        "ncnn",
        "ncnn-vulkan",
        "vulkan",
    }:
        return "ncnn-vulkan"
    return "swin2sr"


_VALID_UPSCALER_BACKENDS = frozenset(
    {
        "auto",
        "swin2sr",
        "swin2sr-pytorch",
        "transformers",
        "ncnn",
        "ncnn-vulkan",
        "vulkan",
    }
)


def _upscaler_auto_download_writable(runtime_dir: Path | None) -> bool:
    if runtime_dir is None:
        return False
    backend = _upscaler_backend()
    if backend not in _VALID_UPSCALER_BACKENDS:
        return False
    if backend in {"swin2sr", "swin2sr-pytorch", "transformers"}:
        return False
    module_path = _upscaler_paths(runtime_dir).get("module")
    if _path_readiness_issue("module", module_path, require_read=True) is not None:
        return False
    bin_dir = runtime_dir / "bin"
    try:
        bin_dir_exists = bin_dir.exists()
    except OSError:
        return False
    if bin_dir_exists:
        return os.access(bin_dir, os.W_OK)
    return os.access(runtime_dir, os.W_OK)


def _default_seed(seed: int | None) -> int:
    return 42 if seed is None else seed


def _step1x_error_code(message: str) -> str | None:
    code = message.split(":", 1)[0].strip()
    if code.startswith("STEP1X_"):
        return code
    return None


def _local_path_from_uri(uri: str, *, require_exists: bool = True) -> Path | None:
    path = local_path_from_file_uri(uri)
    if path is None:
        return None
    if require_exists and not path.exists():
        raise RuntimeError(
            f"STEP1X_ASSET_UNREACHABLE: source asset is not visible: {path}"
        )
    return path.resolve() if path.exists() else path


def _inspect_step1x_scope(
    source_asset_uri: str,
    target: TextureTarget | None,
    *,
    output_dir: Path | None = None,
    texture_size: int | None = None,
) -> Step1XScopeInfo:
    source_asset_path = _local_path_from_uri(source_asset_uri)
    if source_asset_path is None:
        raise RuntimeError(
            "STEP1X_ASSET_UNREACHABLE: only local file:// source_asset_uri values "
            "are supported by the Step1X backend."
        )
    if target is not None and target.mode != "per_material":
        raise RuntimeError(
            f"STEP1X_SCOPE_UNSUPPORTED: target.mode={target.mode!r} is not "
            "supported; use 'per_material'."
        )

    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.Open(str(source_asset_path))
    if stage is None:
        raise RuntimeError(f"STEP1X_ASSET_INVALID: failed to open {source_asset_path}")

    material_path = _resolve_target_material_path(stage, target)
    prim_paths = _resolve_target_prim_paths(stage, target)
    bound_material_paths = _bound_material_paths(stage, prim_paths)
    if material_path is None and len(bound_material_paths) == 1:
        material_path = next(iter(bound_material_paths))
    if material_path is None:
        raise RuntimeError(
            "STEP1X_SCOPE_INVALID: target material could not be resolved from "
            "material_path, material_name, or prim bindings."
        )
    if bound_material_paths and material_path not in bound_material_paths:
        raise RuntimeError(
            "STEP1X_SCOPE_INVALID: selected material "
            f"{material_path} is not bound to target prims "
            f"{sorted(bound_material_paths)}."
        )

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsA(UsdShade.Material):
        raise RuntimeError(
            f"STEP1X_SCOPE_INVALID: material prim not found: {material_path}"
        )

    texture_paths = _find_material_texture_paths(
        material_prim=material_prim,
        source_asset_path=source_asset_path,
        extraction_root=output_dir,
        sdf=Sdf,
    )
    diagnostics: list[dict[str, Any]] = []
    albedo_path = texture_paths.get("albedo")
    if albedo_path is None or not albedo_path.exists():
        synthesized_path = _synthesize_source_albedo_texture(
            material_prim=material_prim,
            material_name=target.material_name if target else material_prim.GetName(),
            output_dir=output_dir,
            texture_size=texture_size,
        )
        if synthesized_path is None:
            if albedo_path is None:
                raise RuntimeError(
                    "STEP1X_TEXTURE_MISSING: no albedo texture found for "
                    f"{material_path}."
                )
            raise RuntimeError(
                f"STEP1X_TEXTURE_MISSING: albedo texture is not visible: {albedo_path}"
            )
        diagnostics.append(
            {
                "code": "STEP1X_SOURCE_ALBEDO_SYNTHESIZED",
                "severity": "warning",
                "message": (
                    "Step1X source material had no visible albedo texture; "
                    "generated a flat source albedo from the material base color."
                ),
                "material_path": material_path,
                "source_albedo_path": str(synthesized_path),
                "missing_albedo_path": str(albedo_path) if albedo_path else None,
            }
        )
        albedo_path = synthesized_path

    scope = Step1XScopeInfo(
        source_asset_path=source_asset_path,
        material_path=material_path,
        material_name=target.material_name if target else material_prim.GetName(),
        prim_paths=tuple(prim_paths),
        source_albedo_path=albedo_path,
        source_normal_path=texture_paths.get("normal"),
        source_orm_path=texture_paths.get("orm"),
        diagnostics=tuple(diagnostics),
    )
    _validate_step1x_scope_uvs(stage, scope)
    return scope


def _validate_step1x_scope_uvs(stage: Any, scope: Step1XScopeInfo) -> None:
    from pxr import UsdGeom

    for source_prim in _target_mesh_prims(stage, scope):
        source_mesh = UsdGeom.Mesh(source_prim)
        mesh_points = list(source_mesh.GetPointsAttr().Get() or [])
        mesh_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        source_st = UsdGeom.PrimvarsAPI(source_mesh.GetPrim()).GetPrimvar("st")
        _resolve_face_varying_st_values(
            source_st,
            point_count=len(mesh_points),
            face_vertex_indices=mesh_indices,
        )


def _prepare_scoped_usd(scope: Step1XScopeInfo, output_dir: Path) -> Path:
    """Create a minimal USD containing only the selected material and meshes."""
    if scope.source_albedo_path is None:
        raise RuntimeError("STEP1X_TEXTURE_MISSING: selected scope has no albedo.")

    from pxr import Sdf, Usd, UsdGeom, UsdShade

    source_stage = Usd.Stage.Open(str(scope.source_asset_path))
    if source_stage is None:
        raise RuntimeError(
            f"STEP1X_ASSET_INVALID: failed to open {scope.source_asset_path}"
        )

    scoped_path = output_dir / "step1x_scoped_source.usda"
    scoped_stage = Usd.Stage.CreateNew(str(scoped_path))
    UsdGeom.SetStageUpAxis(scoped_stage, UsdGeom.GetStageUpAxis(source_stage))
    UsdGeom.SetStageMetersPerUnit(
        scoped_stage,
        UsdGeom.GetStageMetersPerUnit(source_stage),
    )
    UsdGeom.Xform.Define(scoped_stage, "/World")
    UsdGeom.Scope.Define(scoped_stage, "/World/Looks")

    material = UsdShade.Material.Define(scoped_stage, "/World/Looks/SelectedMaterial")
    shader = UsdShade.Shader.Define(
        scoped_stage,
        "/World/Looks/SelectedMaterial/Shader",
    )
    shader_id = (
        _source_material_surface_shader_id(source_stage, scope.material_path)
        or "UsdPreviewSurface"
    )
    shader.CreateIdAttr(shader_id)
    if shader_id == "UsdPreviewSurface":
        uv_reader = UsdShade.Shader.Define(
            scoped_stage,
            "/World/Looks/SelectedMaterial/AlbedoUVReader",
        )
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        albedo_texture = UsdShade.Shader.Define(
            scoped_stage,
            "/World/Looks/SelectedMaterial/AlbedoTexture",
        )
        albedo_texture.CreateIdAttr("UsdUVTexture")
        albedo_texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(scope.source_albedo_path))
        )
        albedo_texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(
            "sRGB"
        )
        albedo_texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader.ConnectableAPI(),
            "result",
        )
        albedo_texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            albedo_texture.ConnectableAPI(),
            "rgb",
        )
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(scope.source_albedo_path))
        )
    if scope.source_normal_path is not None:
        shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(scope.source_normal_path))
        )
    if scope.source_orm_path is not None:
        shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(str(scope.source_orm_path))
        )
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

    target_prims = _target_mesh_prims(source_stage, scope)
    if not target_prims:
        raise RuntimeError(
            "STEP1X_SCOPE_INVALID: no mesh prims were found for selected scope."
        )

    dest_mesh = UsdGeom.Mesh.Define(scoped_stage, "/World/SelectedMesh")
    _copy_merged_meshes_for_step1x(
        [UsdGeom.Mesh(source_prim) for source_prim in target_prims],
        dest_mesh,
        UsdGeom,
    )
    UsdShade.MaterialBindingAPI.Apply(dest_mesh.GetPrim()).Bind(material)
    scoped_stage.Save()
    return scoped_path


def _source_material_surface_shader_id(
    stage: Any, material_path: str | None
) -> str | None:
    if material_path is None:
        return None

    from pxr import UsdShade

    material_prim = stage.GetPrimAtPath(material_path)
    if not material_prim or not material_prim.IsA(UsdShade.Material):
        return None

    for attr in material_prim.GetAttributes():
        attr_name = attr.GetName()
        if not attr_name.startswith("outputs:") or not attr_name.endswith("surface"):
            continue
        for connection in attr.GetConnections():
            shader_prim = stage.GetPrimAtPath(connection.GetPrimPath())
            if not shader_prim or not shader_prim.IsA(UsdShade.Shader):
                continue
            shader_id = UsdShade.Shader(shader_prim).GetIdAttr().Get()
            if shader_id:
                return str(shader_id)
    return None


def _target_mesh_prims(stage: Any, scope: Step1XScopeInfo) -> list[Any]:
    from pxr import UsdGeom, UsdShade

    candidates: list[Any] = []
    if scope.prim_paths:
        for prim_path in scope.prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                continue
            for candidate in _iter_prim_subtree(prim):
                if not candidate.IsA(UsdGeom.Mesh):
                    continue
                material, _ = UsdShade.MaterialBindingAPI(
                    candidate
                ).ComputeBoundMaterial()
                if scope.material_path:
                    if not material or str(material.GetPath()) != scope.material_path:
                        continue
                elif not material:
                    continue
                candidates.append(candidate)
        return candidates

    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if (
            material
            and material.GetPrim()
            and str(material.GetPath()) == scope.material_path
        ):
            candidates.append(prim)
    return candidates


def _copy_merged_meshes_for_step1x(
    source_meshes: list[Any],
    dest_mesh: Any,
    usd_geom: Any,
) -> None:
    """Merge selected material meshes into one OBJ-exportable mesh.

    The reference Step1X script exports the first mesh it finds in the USD.
    Authoring a single selected mesh keeps multi-prim material scopes from
    collapsing to only the first bound mesh at the external runtime boundary.
    """
    if not source_meshes:
        raise RuntimeError(
            "STEP1X_SCOPE_INVALID: no mesh prims were found for selected scope."
        )

    from pxr import Gf, Sdf, Usd

    points: list[Any] = []
    face_vertex_counts: list[int] = []
    face_vertex_indices: list[int] = []
    st_values: list[Any] = []
    has_any_st = False
    all_have_st = True
    point_offset = 0
    time_code = Usd.TimeCode.Default()

    for source_mesh in source_meshes:
        mesh_points = list(source_mesh.GetPointsAttr().Get() or [])
        mesh_counts = list(source_mesh.GetFaceVertexCountsAttr().Get() or [])
        mesh_indices = list(source_mesh.GetFaceVertexIndicesAttr().Get() or [])
        if not mesh_points or not mesh_counts or not mesh_indices:
            continue

        local_to_world = usd_geom.Xformable(
            source_mesh.GetPrim()
        ).ComputeLocalToWorldTransform(time_code)
        points.extend(
            local_to_world.Transform(Gf.Vec3d(point)) for point in mesh_points
        )
        face_vertex_counts.extend(int(count) for count in mesh_counts)
        face_vertex_indices.extend(int(index) + point_offset for index in mesh_indices)

        source_st = usd_geom.PrimvarsAPI(source_mesh.GetPrim()).GetPrimvar("st")
        resolved_st = _resolve_face_varying_st_values(
            source_st,
            point_count=len(mesh_points),
            face_vertex_indices=mesh_indices,
        )
        if resolved_st is None:
            all_have_st = False
        else:
            has_any_st = True
            st_values.extend(resolved_st)

        point_offset += len(mesh_points)

    if not points or not face_vertex_counts or not face_vertex_indices:
        raise RuntimeError(
            "STEP1X_SCOPE_INVALID: selected scope meshes have no polygon data."
        )
    if has_any_st and not all_have_st:
        raise RuntimeError(
            "STEP1X_UV_INVALID: selected scope mixes meshes with and without "
            "st primvars."
        )
    if has_any_st and len(st_values) != len(face_vertex_indices):
        raise RuntimeError(
            "STEP1X_UV_INVALID: selected scope st primvars do not match "
            "face-vertex topology."
        )

    # Step1X rescales against absolute coordinates without first centering.
    # Use the bounds midpoint so uneven vertex density cannot bias framing.
    bounds_min = [math.inf] * 3
    bounds_max = [-math.inf] * 3
    for point in points:
        for axis in range(3):
            coordinate = float(point[axis])
            if not math.isfinite(coordinate):
                raise RuntimeError(
                    "STEP1X_GEOMETRY_INVALID: selected scope contains non-finite "
                    "point coordinates."
                )
            bounds_min[axis] = min(bounds_min[axis], coordinate)
            bounds_max[axis] = max(bounds_max[axis], coordinate)
    center = Gf.Vec3d(
        *(
            _finite_bounds_midpoint(bounds_min[axis], bounds_max[axis])
            for axis in range(3)
        )
    )
    points = [point - center for point in points]

    dest_mesh.CreatePointsAttr(points)
    dest_mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    dest_mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    dest_mesh.CreateExtentAttr(usd_geom.PointBased.ComputeExtent(points))
    if has_any_st:
        dest_st = usd_geom.PrimvarsAPI(dest_mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            usd_geom.Tokens.faceVarying,
        )
        dest_st.Set(st_values)


def _finite_bounds_midpoint(lower: float, upper: float) -> float:
    """Return the finite binary64 midpoint of two finite bounds."""
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("bounds midpoint inputs must be finite")
    half_max = sys.float_info.max * 0.5
    twice_min_normal = sys.float_info.min * 2.0
    if abs(lower) <= half_max and abs(upper) <= half_max:
        return (lower + upper) * 0.5
    if abs(lower) < twice_min_normal:
        return lower + upper * 0.5
    if abs(upper) < twice_min_normal:
        return lower * 0.5 + upper
    return lower * 0.5 + upper * 0.5


def _resolve_face_varying_st_values(
    source_st: Any,
    *,
    point_count: int,
    face_vertex_indices: list[int],
) -> list[Any] | None:
    if not source_st or not source_st.HasValue():
        return None
    values = list(source_st.Get() or [])
    if not values:
        return None

    indices = list(source_st.GetIndices() or [])
    if indices:
        if len(indices) != len(face_vertex_indices):
            raise RuntimeError(
                "STEP1X_UV_INVALID: indexed st primvar length does not match "
                "face-vertex topology."
            )
        max_index = max(indices)
        if max_index >= len(values) or min(indices) < 0:
            raise RuntimeError(
                "STEP1X_UV_INVALID: indexed st primvar references missing values."
            )
        return _validate_finite_st_values([values[int(index)] for index in indices])

    if len(values) == len(face_vertex_indices):
        return _validate_finite_st_values(values)
    if len(values) == point_count:
        return _validate_finite_st_values(
            [values[int(index)] for index in face_vertex_indices]
        )

    interpolation = source_st.GetInterpolation()
    raise RuntimeError(
        "STEP1X_UV_INVALID: st primvar interpolation "
        f"{interpolation!r} with {len(values)} values cannot be converted to "
        f"{len(face_vertex_indices)} face-varying values."
    )


def _validate_finite_st_values(values: list[Any]) -> list[Any]:
    for value in values:
        try:
            u = float(value[0])
            v = float(value[1])
        except (IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "STEP1X_UV_INVALID: st primvars contain invalid coordinates."
            ) from exc
        if not math.isfinite(u) or not math.isfinite(v):
            raise RuntimeError(
                "STEP1X_UV_INVALID: st primvars contain non-finite coordinates."
            )
    return values


def _resolve_target_material_path(
    stage: Any, target: TextureTarget | None
) -> str | None:
    from pxr import UsdShade

    if target is None:
        material_paths = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Material)
        ]
        return material_paths[0] if len(material_paths) == 1 else None
    if target.material_path:
        prim = stage.GetPrimAtPath(target.material_path)
        if prim and prim.IsA(UsdShade.Material):
            return target.material_path
        raise RuntimeError(
            f"STEP1X_SCOPE_INVALID: material_path not found: {target.material_path}"
        )
    if target.material_name:
        matches = [
            str(prim.GetPath())
            for prim in stage.Traverse()
            if prim.IsA(UsdShade.Material) and prim.GetName() == target.material_name
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeError(
                "STEP1X_SCOPE_INVALID: material_name is ambiguous: "
                f"{target.material_name} -> {matches}"
            )
    return None


def _resolve_target_prim_paths(stage: Any, target: TextureTarget | None) -> list[str]:
    if target is None:
        return []
    resolved: list[str] = []
    for prim_path in target.prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            if target.strict_scope:
                raise RuntimeError(
                    f"STEP1X_SCOPE_INVALID: target prim not found: {prim_path}"
                )
            continue
        resolved.append(prim_path)
    return resolved


def _bound_material_paths(stage: Any, prim_paths: list[str]) -> set[str]:
    from pxr import UsdShade

    paths: set[str] = set()
    for prim_path in prim_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if not prim:
            continue
        for candidate in _iter_prim_subtree(prim):
            material, _ = UsdShade.MaterialBindingAPI(candidate).ComputeBoundMaterial()
            if material and material.GetPrim() and material.GetPrim().IsValid():
                paths.add(str(material.GetPath()))
    return paths


def _find_material_texture_paths(
    *,
    material_prim: Any,
    source_asset_path: Path,
    extraction_root: Path | None,
    sdf: Any,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    mdl_paths: list[Path] = []
    connected_channels = _connected_texture_channels(material_prim)
    for prim in _iter_prim_subtree(material_prim):
        for attr in prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, sdf.AssetPath):
                continue
            maybe_path = _resolve_asset_path(
                value,
                source_asset_path.parent,
                extraction_root=extraction_root,
            )
            if maybe_path is not None and maybe_path.suffix.lower() == ".mdl":
                mdl_paths.append(maybe_path)
            channel = connected_channels.get(str(prim.GetPath()))
            if channel is None:
                channel = _channel_from_attr_name(prim.GetName())
            if channel is None:
                channel = _channel_from_attr_name(attr.GetName())
            if channel is None:
                continue
            path = maybe_path
            if path is not None and path.suffix.lower() in _IMAGE_SUFFIXES:
                paths.setdefault(channel, path)
    for mdl_path in mdl_paths:
        for channel, path in _find_mdl_texture_paths(mdl_path).items():
            paths.setdefault(channel, path)
    return paths


_BASE_COLOR_INPUTS = {
    "base_color",
    "basecolor",
    "base_color_constant",
    "diffuse",
    "diffuse_color",
    "diffusecolor",
    "diffuse_color_constant",
    "diffuse_tint",
    "diffusecolorconstant",
    "diffusetint",
}


def _synthesize_source_albedo_texture(
    *,
    material_prim: Any,
    material_name: str | None,
    output_dir: Path | None,
    texture_size: int | None,
) -> Path | None:
    if output_dir is None:
        return None

    from PIL import Image

    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", material_name or "material").strip(
        "._-"
    )
    safe_name = safe_name or "material"
    size = max(1, int(texture_size or 1024))
    color = _material_base_color(material_prim) or (0.5, 0.5, 0.5)
    srgb = tuple(_linear_channel_to_srgb_byte(channel) for channel in color)

    albedo_dir = output_dir / "source_albedo"
    albedo_dir.mkdir(parents=True, exist_ok=True)
    albedo_path = albedo_dir / f"source_albedo_{safe_name}.png"
    Image.new("RGB", (size, size), srgb).save(albedo_path)
    return albedo_path.resolve()


def _material_base_color(material_prim: Any) -> tuple[float, float, float] | None:
    for prim in _iter_prim_subtree(material_prim):
        for attr in prim.GetAttributes():
            name = attr.GetName().split(":")[-1].replace("-", "_").lower()
            compact = name.replace("_", "")
            if name not in _BASE_COLOR_INPUTS and compact not in _BASE_COLOR_INPUTS:
                continue
            color = _coerce_color3(attr.Get())
            if color is not None:
                return color
    return None


def _coerce_color3(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        channels = (float(value[0]), float(value[1]), float(value[2]))
    except (IndexError, TypeError, ValueError):
        return None
    if not all(math.isfinite(channel) for channel in channels):
        return None
    return channels


def _linear_channel_to_srgb_byte(value: float) -> int:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.5
    if not math.isfinite(value):
        value = 0.5
    value = max(0.0, min(1.0, value))
    if value <= 0.0031308:
        srgb = 12.92 * value
    else:
        srgb = 1.055 * (value ** (1.0 / 2.4)) - 0.055
    return max(0, min(255, int(round(srgb * 255))))


def _connected_texture_channels(material_prim: Any) -> dict[str, str]:
    """Map texture shader prims to material channels from downstream connections."""
    channels: dict[str, str] = {}
    for prim in _iter_prim_subtree(material_prim):
        for attr in prim.GetAttributes():
            channel = _channel_from_attr_name(attr.GetName())
            if channel is None:
                continue
            for connection in attr.GetConnections():
                source_prim_path = connection.GetPrimPath()
                if not source_prim_path.isEmpty:
                    channels.setdefault(str(source_prim_path), channel)
    return channels


def _find_mdl_texture_paths(mdl_path: Path) -> dict[str, Path]:
    """Parse simple local MDL texture_2d assignments used by SimReady assets."""
    if not mdl_path.exists():
        return {}
    text = mdl_path.read_text(encoding="utf-8", errors="replace")
    paths: dict[str, Path] = {}
    for match in re.finditer(
        r"(?P<input>[A-Za-z0-9_:]+)\s*:\s*texture_2d\(\s*\"(?P<path>[^\"]+)\"",
        text,
    ):
        channel = _channel_from_attr_name(match.group("input"))
        if channel is None:
            continue
        raw_path = match.group("path")
        parsed = urlparse(raw_path)
        if parsed.scheme:
            continue
        path = Path(raw_path)
        if not path.is_absolute():
            path = mdl_path.parent / path
        if path.suffix.lower() in _IMAGE_SUFFIXES:
            paths.setdefault(channel, path.expanduser().resolve())
    return paths


def _iter_prim_subtree(prim: Any) -> list[Any]:
    prims = [prim]
    for child in prim.GetChildren():
        prims.extend(_iter_prim_subtree(child))
    return prims


def _channel_from_attr_name(attr_name: str) -> str | None:
    normalized = attr_name.split(":")[-1].replace("-", "_").lower()
    normalized = normalized.replace("__", "_")
    compact = normalized.replace("_", "")
    if normalized in _ALBEDO_INPUTS or compact in _ALBEDO_INPUTS:
        return "albedo"
    if normalized in _NORMAL_INPUTS or compact in _NORMAL_INPUTS:
        return "normal"
    if normalized in _ORM_INPUTS or compact in _ORM_INPUTS:
        return "orm"
    return None


def _resolve_asset_path(
    value: Any,
    source_dir: Path,
    *,
    extraction_root: Path | None,
) -> Path | None:
    raw = value.resolvedPath or value.path
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
    elif parsed.scheme:
        return None
    else:
        raw_path = raw
    package_member_path = _resolve_package_asset_path(
        raw_path,
        source_dir,
        extraction_root=extraction_root,
    )
    if package_member_path is not None:
        return package_member_path

    path = Path(raw_path)
    if not path.is_absolute():
        path = source_dir / path
    return path.expanduser().resolve() if path.exists() else path.expanduser()


def _resolve_package_asset_path(
    raw_path: str,
    source_dir: Path,
    *,
    extraction_root: Path | None,
) -> Path | None:
    package_member = parse_package_member_asset_path(raw_path, base_dir=source_dir)
    if package_member is None:
        return None

    if extraction_root is None:
        raise Step1XExtractionError(
            "STEP1X_PACKAGE_EXTRACTION_FAILED: package member extraction "
            "requires a writable output directory."
        )

    package_path, member_name = package_member
    extracted_root = (
        extraction_root
        / ".step1x_package_assets"
        / package_member_cache_name(package_path)
    )
    extracted_path = extracted_root / member_name
    try:
        package_mtime = package_path.stat().st_mtime
    except OSError as exc:
        raise Step1XExtractionError(
            f"STEP1X_PACKAGE_EXTRACTION_FAILED: failed to stat package "
            f"{package_path}: {exc}"
        ) from exc
    try:
        if extracted_path.exists() and extracted_path.stat().st_mtime >= package_mtime:
            return extracted_path.resolve()
    except OSError:
        pass

    try:
        member_path = extract_usdz_member_to_dir(
            package_path,
            member_name,
            extracted_root,
            max_bytes=_MAX_PACKAGE_ASSET_BYTES,
        )
        if member_path is None:
            return None
    except ArchiveSizeLimitExceeded as exc:
        with suppress(OSError):
            extracted_path.unlink()
        raise Step1XExtractionError(
            "STEP1X_PACKAGE_EXTRACTION_FAILED: package member exceeded "
            f"{_MAX_PACKAGE_ASSET_BYTES} bytes while extracting."
        ) from exc
    except ValueError:
        return None
    except OSError as exc:
        raise Step1XExtractionError(
            f"STEP1X_PACKAGE_EXTRACTION_FAILED: failed to extract package "
            f"member {member_name!r} from {package_path}: {exc}"
        ) from exc
    return extracted_path.resolve() if extracted_path.exists() else extracted_path


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _first_matching(root: Path, pattern: str) -> Path | None:
    matches = sorted(root.glob(pattern))
    return matches[0] if matches else None


def _validate_albedo(path: Path) -> tuple[int, int]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        if max(stat.var or [0.0]) <= 0.0:
            raise RuntimeError(
                f"STEP1X_OUTPUT_BLANK: generated albedo is blank: {path}"
            )
        return rgb.size


def _terminate_process(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _tail_text(path: Path, *, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:]


def _redact_command(cmd: list[str]) -> list[str]:
    redacted: list[str] = []
    hide_next = False
    for token in cmd:
        if hide_next:
            redacted.append("<redacted>")
            hide_next = False
            continue
        key = token.split("=", 1)[0].lower().lstrip("-").replace("-", "_")
        if _is_secret_key(key):
            redacted.append("<redacted>")
            if token.startswith("--") and "=" not in token:
                hide_next = True
            continue
        redacted.append(token)
    return redacted


def _is_secret_key(key: str) -> bool:
    return key in _SECRET_KEY_NAMES or any(
        key.endswith(suffix) for suffix in _SECRET_KEY_SUFFIXES
    )
