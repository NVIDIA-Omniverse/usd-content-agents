# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in VideoMatGen backend boundary for material creation.

WP8 keeps VideoMatGen behind the WP0 material-creation protocol without making
it a default backend.  No approved runtime, checkpoint, or usage-rights bundle is
present in this repository, so the adapter reports structured blockers instead
of fabricating material artifacts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from material_agent.material_library_generation.creation_contract import (
    BackendMaterialResult,
    CreateMaterialRequest,
    MaterialConditioningKind,
    MaterialCreationDiagnostic,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialDiagnosticSeverity,
    PreparedMaterialConditioning,
)

VIDEOMATGEN_BACKEND_NAME = "videomatgen"
VIDEOMATGEN_ADAPTER_REVISION = "material-agent-videomatgen-adapter.v1"
VIDEOMATGEN_UNAVAILABLE_REVISION = "videomatgen.runtime-unavailable"

_REQUIRED_CONDITIONING_KINDS = (
    MaterialConditioningKind.SCOPED_USD,
    MaterialConditioningKind.UV_LAYOUT,
    MaterialConditioningKind.UV_MASK,
    MaterialConditioningKind.NORMAL,
    MaterialConditioningKind.DEPTH,
    MaterialConditioningKind.RENDER,
)
_CHECKPOINT_DIAGNOSTIC_CODES = frozenset(
    {
        "VIDEOMATGEN_CHECKPOINT_UNCONFIGURED",
        "VIDEOMATGEN_CHECKPOINT_MISSING",
    }
)


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return None
        return Path(normalized).expanduser()
    return Path(value).expanduser()


def _diagnostic(
    code: str,
    message: str,
    *,
    phase: str,
    details: dict[str, str] | None = None,
) -> MaterialCreationDiagnostic:
    return MaterialCreationDiagnostic(
        code=code,
        message=message,
        severity=MaterialDiagnosticSeverity.ERROR,
        phase=phase,
        retryable=False,
        details=details or {},
    )


@dataclass(frozen=True)
class VideoMatGenBackendConfig:
    """Runtime coordinates required before WP8 can run VideoMatGen."""

    source_root: str | Path | None = None
    entrypoint: str | Path | None = None
    checkpoint_path: str | Path | None = None
    source_revision: str | None = None
    rights_evidence_uri: str | None = None
    model_revision: str | None = None
    min_gpu_count: int = 2

    def __post_init__(self) -> None:
        if self.min_gpu_count < 1:
            raise ValueError("min_gpu_count must be at least 1")
        object.__setattr__(self, "source_root", _optional_path(self.source_root))
        object.__setattr__(self, "entrypoint", _optional_path(self.entrypoint))
        object.__setattr__(
            self, "checkpoint_path", _optional_path(self.checkpoint_path)
        )
        object.__setattr__(
            self, "source_revision", _optional_text(self.source_revision)
        )
        object.__setattr__(
            self, "rights_evidence_uri", _optional_text(self.rights_evidence_uri)
        )
        object.__setattr__(self, "model_revision", _optional_text(self.model_revision))

    @property
    def effective_model_revision(self) -> str:
        return self.model_revision or VIDEOMATGEN_UNAVAILABLE_REVISION


class VideoMatGenMaterialCreationBackend:
    """WP0-compatible adapter that reports WP8 runtime blockers explicitly."""

    def __init__(self, config: VideoMatGenBackendConfig | None = None) -> None:
        self.config = config or VideoMatGenBackendConfig()

    @property
    def name(self) -> str:
        return VIDEOMATGEN_BACKEND_NAME

    @property
    def revision(self) -> str:
        return VIDEOMATGEN_ADAPTER_REVISION

    def create(
        self,
        request: CreateMaterialRequest,
        *,
        output_dir: Path,
        conditioning: PreparedMaterialConditioning | None = None,
        cancel_event: threading.Event | None = None,
    ) -> BackendMaterialResult:
        del output_dir
        if cancel_event is not None and cancel_event.is_set():
            raise MaterialCreationError(
                MaterialCreationErrorCode.CANCELLED,
                "VideoMatGen creation was cancelled before execution.",
                backend=self.name,
                retryable=False,
            )
        self._validate_request(request, conditioning)
        diagnostics = self._runtime_diagnostics()
        if not diagnostics:
            diagnostics = (
                _diagnostic(
                    "VIDEOMATGEN_RUNTIME_CONTRACT_UNAVAILABLE",
                    "VideoMatGen runtime assets are present, but the approved "
                    "multi-GPU inference command and normalized output mapping "
                    "are not checked into this repository.",
                    phase="runtime_probe",
                    details={
                        "source_revision": self.config.source_revision or "",
                        "model_revision": self.config.effective_model_revision,
                        "min_gpu_count": str(self.config.min_gpu_count),
                    },
                ),
            )
        error_code = MaterialCreationErrorCode.MISSING_CHECKPOINT
        if any(
            diagnostic.code not in _CHECKPOINT_DIAGNOSTIC_CODES
            for diagnostic in diagnostics
        ):
            error_code = MaterialCreationErrorCode.BACKEND_UNAVAILABLE
        raise MaterialCreationError(
            error_code,
            "VideoMatGen backend is not available for material creation.",
            backend=self.name,
            retryable=False,
            diagnostics=diagnostics,
        )

    def _validate_request(
        self,
        request: CreateMaterialRequest,
        conditioning: PreparedMaterialConditioning | None,
    ) -> None:
        if request.creation_mode is not MaterialCreationMode.ASSET_UV:
            diagnostic = _diagnostic(
                "VIDEOMATGEN_TILEABLE_UNSUPPORTED",
                "VideoMatGen WP8 only supports asset-UV material creation.",
                phase="capability_check",
                details={"creation_mode": request.creation_mode.value},
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.UNSUPPORTED_MATERIAL,
                diagnostic.message,
                backend=self.name,
                retryable=False,
                diagnostics=(diagnostic,),
            )
        if conditioning is None:
            diagnostic = _diagnostic(
                "VIDEOMATGEN_CONDITIONING_REQUIRED",
                "VideoMatGen requires WP4 prepared geometry conditioning.",
                phase="input_validation",
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                diagnostic.message,
                backend=self.name,
                retryable=False,
                diagnostics=(diagnostic,),
            )
        try:
            conditioning.validate_request(request)
        except ValueError as exc:
            diagnostic = _diagnostic(
                "VIDEOMATGEN_CONDITIONING_MISMATCH",
                str(exc),
                phase="input_validation",
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                str(exc),
                backend=self.name,
                retryable=False,
                diagnostics=(diagnostic,),
            ) from exc

        present = {artifact.kind for artifact in conditioning.artifacts}
        missing = tuple(
            kind.value for kind in _REQUIRED_CONDITIONING_KINDS if kind not in present
        )
        if missing:
            diagnostic = _diagnostic(
                "VIDEOMATGEN_CONDITIONING_INCOMPLETE",
                "VideoMatGen requires scoped geometry, UV, normal, depth, and "
                "render conditioning artifacts.",
                phase="input_validation",
                details={"missing_kinds": ",".join(missing)},
            )
            raise MaterialCreationError(
                MaterialCreationErrorCode.INVALID_REQUEST,
                diagnostic.message,
                backend=self.name,
                retryable=False,
                diagnostics=(diagnostic,),
            )

    def _runtime_diagnostics(self) -> tuple[MaterialCreationDiagnostic, ...]:
        diagnostics: list[MaterialCreationDiagnostic] = []
        source_root = self.config.source_root
        if source_root is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_SOURCE_UNCONFIGURED",
                    "VideoMatGen source root is not configured.",
                    phase="runtime_probe",
                )
            )
        else:
            source_root = Path(source_root)
        if source_root is not None and not source_root.exists():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_SOURCE_MISSING",
                    "VideoMatGen source root does not exist or is not a directory.",
                    phase="runtime_probe",
                    details={"source_root": source_root.as_posix()},
                )
            )
        elif source_root is not None and not source_root.is_dir():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_SOURCE_MISSING",
                    "VideoMatGen source root does not exist or is not a directory.",
                    phase="runtime_probe",
                    details={"source_root": source_root.as_posix()},
                )
            )
        entrypoint = self.config.entrypoint
        if entrypoint is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_ENTRYPOINT_UNCONFIGURED",
                    "VideoMatGen inference entrypoint is not configured.",
                    phase="runtime_probe",
                )
            )
        else:
            entrypoint = Path(entrypoint)
        if entrypoint is not None and not entrypoint.exists():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_ENTRYPOINT_MISSING",
                    "VideoMatGen inference entrypoint does not exist or is not a file.",
                    phase="runtime_probe",
                    details={"entrypoint": entrypoint.as_posix()},
                )
            )
        elif entrypoint is not None and not entrypoint.is_file():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_ENTRYPOINT_MISSING",
                    "VideoMatGen inference entrypoint does not exist or is not a file.",
                    phase="runtime_probe",
                    details={"entrypoint": entrypoint.as_posix()},
                )
            )
        if self.config.source_revision is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_SOURCE_REVISION_UNPINNED",
                    "VideoMatGen source revision is not pinned.",
                    phase="runtime_probe",
                )
            )
        if self.config.model_revision is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_MODEL_REVISION_UNPINNED",
                    "VideoMatGen model revision is not pinned.",
                    phase="runtime_probe",
                )
            )
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_CHECKPOINT_UNCONFIGURED",
                    "VideoMatGen checkpoint path is not configured.",
                    phase="runtime_probe",
                )
            )
        else:
            checkpoint_path = Path(checkpoint_path)
        if checkpoint_path is not None and not checkpoint_path.exists():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_CHECKPOINT_MISSING",
                    "VideoMatGen checkpoint path does not exist or is not a file.",
                    phase="runtime_probe",
                    details={"checkpoint_path": checkpoint_path.as_posix()},
                )
            )
        elif checkpoint_path is not None and not checkpoint_path.is_file():
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_CHECKPOINT_MISSING",
                    "VideoMatGen checkpoint path does not exist or is not a file.",
                    phase="runtime_probe",
                    details={"checkpoint_path": checkpoint_path.as_posix()},
                )
            )
        if self.config.rights_evidence_uri is None:
            diagnostics.append(
                _diagnostic(
                    "VIDEOMATGEN_RIGHTS_UNVERIFIED",
                    "VideoMatGen source/checkpoint usage rights are not documented.",
                    phase="runtime_probe",
                )
            )
        return tuple(diagnostics)
