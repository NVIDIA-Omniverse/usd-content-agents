# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner/executor client contract plus real and mock implementations."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import time
import zipfile
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Literal, Protocol
from uuid import uuid4

import requests
from pydantic import BaseModel, ConfigDict, Field, model_validator
from world_understanding.utils.usd.package import (
    DEFAULT_MAX_USDZ_EXTRACTED_BYTES,
    DEFAULT_MAX_USDZ_MEMBER_BYTES,
    DEFAULT_MAX_USDZ_MEMBERS,
)

from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding

from .models import (
    TextureExecutionResult,
    TextureGeneratorInputs,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureWorkflowRequest,
)

_REMOTE_EXECUTION_STATE_SCHEMA_VERSION: Literal[
    "content-agent-workflows.texture-remote-execution.v1"
] = "content-agent-workflows.texture-remote-execution.v1"
_PACKAGE_HASH_READ_BYTES = 1024 * 1024


def _verified_upload_snapshot(
    path: Path,
    *,
    expected_sha256: str,
    expected_size_bytes: int,
    label: str,
) -> BytesIO:
    """Capture exact verified bytes so multipart upload cannot reread a mutable inode."""

    expanded = path.expanduser()
    if expanded.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = expanded.resolve()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as error:
        raise ValueError(f"{label} bytes changed before upload") from error
    digest = hashlib.sha256()
    snapshot = BytesIO()
    with os.fdopen(descriptor, "rb") as source:
        source_stat = os.fstat(source.fileno())
        for chunk in iter(lambda: source.read(_PACKAGE_HASH_READ_BYTES), b""):
            digest.update(chunk)
            snapshot.write(chunk)
    if (
        not stat.S_ISREG(source_stat.st_mode)
        or source_stat.st_size != expected_size_bytes
        or snapshot.tell() != expected_size_bytes
        or digest.hexdigest() != expected_sha256
    ):
        raise ValueError(f"{label} bytes changed before upload")
    snapshot.seek(0)
    return snapshot


class _TextureRemoteExecutionState(BaseModel):
    """Durable intent for one exact-unit Texture workflow execution."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.texture-remote-execution.v1"] = (
        _REMOTE_EXECUTION_STATE_SCHEMA_VERSION
    )
    execution_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    session_id: str = Field(min_length=1)
    plan_key: str = Field(min_length=64, max_length=64)
    unit_ids: tuple[str, ...] = Field(min_length=1)
    preserved_unit_ids: tuple[str, ...] = ()
    generation_by_unit_before: dict[str, int]
    material_textures: dict[str, dict[str, Any]] = Field(default_factory=dict)
    full_plan_regeneration: bool
    phase: Literal["generation", "apply", "collect"] = "generation"
    baseline_status_digest: str | None = None
    baseline_execution_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{32}$",
    )
    dispatched: bool = False

    @model_validator(mode="after")
    def _validate_state(self) -> _TextureRemoteExecutionState:
        if len(self.unit_ids) != len(set(self.unit_ids)):
            raise ValueError("remote execution unit_ids must be unique")
        if len(self.preserved_unit_ids) != len(set(self.preserved_unit_ids)):
            raise ValueError("remote execution preserved_unit_ids must be unique")
        if set(self.unit_ids) & set(self.preserved_unit_ids):
            raise ValueError(
                "remote execution unit_ids and preserved_unit_ids must be disjoint"
            )
        if set(self.generation_by_unit_before) != set(self.unit_ids):
            raise ValueError(
                "remote execution generation state must cover requested unit_ids"
            )
        if any(value < 0 for value in self.generation_by_unit_before.values()):
            raise ValueError("remote execution generations must be non-negative")
        if self.phase in {"generation", "apply"} and not self.baseline_status_digest:
            raise ValueError(
                "remote execution dispatch phase requires a status baseline"
            )
        if self.phase == "apply" and not self.full_plan_regeneration:
            raise ValueError("only full-plan execution may use a separate apply phase")
        if self.phase == "collect" and (
            self.baseline_status_digest is not None
            or self.baseline_execution_id is not None
            or self.dispatched
        ):
            raise ValueError("remote execution collect phase cannot be dispatched")
        return self


class TexturePlannerExecutorClient(Protocol):
    """Workflow-facing boundary that WP7 may adapt to the real service.

    ``execute_resumable`` is an optional extension. Adapters that accept its
    ``cancellation_check`` keyword opt in by exposing
    ``supports_resumable_cancellation = True``. The workflow omits that keyword
    for adapters without the marker so existing resumable adapters remain
    compatible. Embedded authorization reconciliation is also optional and is
    enabled only by an adapter-provided ``can_reconcile_authorized_execution``
    proof for one exact checkpointed execution.
    """

    def plan(self, request: TextureWorkflowRequest) -> TexturePlanDocument:
        """Produce the immutable plan before any generation work."""

    def execute(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> TextureExecutionResult:
        """Execute exactly ``unit_ids`` while preserving accepted artifacts."""

    def export_resume_state(self, plan: TexturePlanDocument) -> dict[str, Any]:
        """Return JSON-safe adapter state needed to resume this plan."""

    def restore_resume_state(
        self,
        plan: TexturePlanDocument,
        state: Mapping[str, Any],
    ) -> None:
        """Restore adapter state before resuming an existing plan."""


class TextureAgentServiceCancellationRequested(RuntimeError):
    """Raised when cooperative cancellation interrupts a remote execution."""


class MockTextureExecutionCall(BaseModel):
    """Recorded invocation for assertions in wrapper and workflow tests."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    unit_ids: tuple[str, ...]
    preserved_unit_ids: tuple[str, ...]


def _mock_unit(index: int) -> dict[str, object]:
    material_path = f"/World/Looks/Material_{index:03d}"
    identity = {
        "material_prim_paths": [material_path],
        "unit_mode": "per_material",
    }
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    unit_id = f"tu_{digest[:20]}"
    return {
        "unit_id": unit_id,
        "unit_mode": "per_material",
        "material_prim_paths": [material_path],
        "member_prim_paths": [f"/World/Geometry/Mesh_{index:03d}"],
        "member_subset_paths": [],
        "group_key": None,
        "display_name": f"Material_{index:03d}",
        "selection_reason_code": "effectively_bound",
        "selection_reason": "Selected by deterministic mock effective binding.",
        "detail_policy": "surface_only",
    }


def _mock_plan(source_asset: str, unit_count: int) -> TexturePlanDocument:
    units = [_mock_unit(index) for index in range(unit_count)]
    payload = {
        "schema_version": "texture-agent-plan.v1",
        "generated_at": datetime(2026, 6, 29, tzinfo=UTC).isoformat(),
        "request": {
            "schema_version": "texture-agent-plan-request.v1",
            "source": {
                "source_asset": source_asset,
                "upstream_assignment_artifact": None,
                "source_asset_sha256": None,
            },
            "discovery_mode": "effective_bound",
            "unit_mode": "per_material",
            "explicit_material_paths": [],
            "explicit_prim_paths": [],
            "detail_policy": "surface_only",
            "texture_size": 1024,
            "backend": "mock",
            "backend_default_cap": 32,
            "operator_override_cap": None,
            "max_concurrency": 4,
            "unit_timeout_seconds": 600,
        },
        "limits": {
            "global_default_cap": 32,
            "backend_default_cap": 32,
            "operator_override_cap": None,
            "effective_cap": 32,
            "hard_cap": 64,
        },
        "execution": {
            "backend": "mock",
            "texture_size": 1024,
            "max_concurrency": 4,
            "unit_timeout_seconds": 600,
        },
        "counts": {
            "authored_material_count": unit_count,
            "renderable_prim_count": unit_count,
            "renderable_subset_count": 0,
            "effective_bound_material_count": unit_count,
            "selected_material_count": unit_count,
            "selected_unit_count": unit_count,
            "skipped_item_count": 0,
            "planned_generation_job_count": unit_count,
        },
        "selected_units": units,
        "skipped_items": [],
        "decision": {
            "state": "ready",
            "execution_allowed": True,
            "consolidation_required": False,
            "explicit_narrowing_required": False,
            "reasons": [],
            "recommended_actions": [],
        },
    }
    return TexturePlanDocument.model_validate(payload)


class MockTexturePlannerExecutorClient:
    """File-backed mock with no Texture Agent or model backend dependency."""

    def __init__(
        self,
        *,
        unit_count: int = 2,
        plan_document: TexturePlanDocument | None = None,
    ) -> None:
        if not 1 <= unit_count <= 32:
            raise ValueError("mock unit_count must be between 1 and 32")
        self._unit_count = unit_count
        self._plan_document = plan_document
        self.plan_calls: list[TextureWorkflowRequest] = []
        self.execution_calls: list[MockTextureExecutionCall] = []
        self._generation_by_unit: dict[str, int] = {}

    @property
    def unit_ids(self) -> tuple[str, ...]:
        """Expose deterministic IDs for mock validator fixtures."""

        plan = self._plan_document or _mock_plan("mock.usda", self._unit_count)
        return plan.selected_unit_ids

    def plan(self, request: TextureWorkflowRequest) -> TexturePlanDocument:
        self.plan_calls.append(request)
        return self._plan_document or _mock_plan(request.source_asset, self._unit_count)

    def execute(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> TextureExecutionResult:
        unknown_ids = set(unit_ids) - set(plan.selected_unit_ids)
        if unknown_ids:
            raise ValueError(f"mock executor received unknown unit IDs: {unknown_ids}")
        if set(unit_ids) & set(preserved_artifacts):
            raise ValueError("mock executor cannot regenerate preserved unit artifacts")
        self.execution_calls.append(
            MockTextureExecutionCall(
                unit_ids=unit_ids,
                preserved_unit_ids=tuple(
                    unit_id
                    for unit_id in plan.selected_unit_ids
                    if unit_id in preserved_artifacts
                ),
            )
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_asset = output_dir / "textured_asset.usda"
        if not output_asset.exists():
            output_asset.write_text(
                '#usda 1.0\n\ndef Xform "TexturedAsset" {\n}\n',
                encoding="utf-8",
            )

        artifacts: list[TextureUnitArtifact] = []
        for unit_id in unit_ids:
            generation = self._generation_by_unit.get(unit_id, 0) + 1
            self._generation_by_unit[unit_id] = generation
            artifact_path = (
                output_dir
                / "textures"
                / unit_id
                / f"generation-{generation}.mock-texture"
            )
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                f"mock texture for {unit_id}, generation {generation}\n",
                encoding="utf-8",
            )
            artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=(str(artifact_path.resolve()),),
                    generation=generation,
                    metadata={"backend": "mock"},
                )
            )

        return TextureExecutionResult(
            requested_unit_ids=unit_ids,
            unit_artifacts=tuple(artifacts),
            output_asset_path=str(output_asset.resolve()),
            metadata={"backend": "mock", "live_backend_invoked": False},
        )

    def export_resume_state(self, plan: TexturePlanDocument) -> dict[str, Any]:
        """Persist deterministic mock generations for interruption tests."""

        if set(self._generation_by_unit) - set(plan.selected_unit_ids):
            raise ValueError("mock generation state contains IDs outside the plan")
        return {"generation_by_unit": dict(self._generation_by_unit)}

    def restore_resume_state(
        self,
        plan: TexturePlanDocument,
        state: Mapping[str, Any],
    ) -> None:
        """Restore mock generation counters from a workflow checkpoint."""

        generations = state.get("generation_by_unit") or {}
        if not isinstance(generations, Mapping):
            raise ValueError("mock resume generations must be an object")
        restored = {str(key): int(value) for key, value in generations.items()}
        if set(restored) - set(plan.selected_unit_ids):
            raise ValueError("mock resume state contains IDs outside the plan")
        if any(value < 0 for value in restored.values()):
            raise ValueError("mock resume generations must be non-negative")
        self._generation_by_unit = restored


class TextureAgentServiceClient:
    """Adapter for the real Texture Agent plan/regenerate REST contract.

    ``max_status_poll_failures`` tolerates that many consecutive network,
    408, 429, or 5xx failures and raises the original error on the next one.
    """

    _TERMINAL_STATUSES = {"completed", "failed", "cancelled"}
    supports_resumable_cancellation = True

    def __init__(
        self,
        base_url: str = "http://localhost:8001",
        *,
        timeout_seconds: float = 1800,
        poll_interval_seconds: float = 1,
        max_status_poll_failures: int = 5,
        token: str | None = None,
        session: Any | None = None,
    ) -> None:
        if not timeout_seconds > 0:
            raise ValueError("timeout_seconds must be positive")
        if not poll_interval_seconds >= 0:
            raise ValueError("poll_interval_seconds must be non-negative")
        if max_status_poll_failures < 0:
            raise ValueError("max_status_poll_failures must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.max_status_poll_failures = max_status_poll_failures
        self._http = session or requests.Session()
        self._http.headers.update({"User-Agent": "content-workflow-texture/1.0"})
        if token:
            self._http.headers.update({"Authorization": f"Bearer {token}"})
        self._session_by_plan: dict[str, str] = {}
        self._generation_by_unit: dict[str, int] = {}
        self._pending_execution: _TextureRemoteExecutionState | None = None

    @staticmethod
    def _plan_key(plan: TexturePlanDocument) -> str:
        payload = json.dumps(
            plan.model_dump(mode="json", exclude_none=False),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def material_textures_for_outer_plan(
        unit_ids: tuple[str, ...],
        *,
        unit_scope_by_id: Mapping[str, Mapping[str, Any]],
        generator_inputs_by_unit: Mapping[str, TextureGeneratorInputs],
    ) -> dict[str, dict[str, Any]]:
        """Translate exact outer unit semantics into service regeneration inputs."""

        if tuple(unit_scope_by_id) != unit_ids:
            raise ValueError("Texture service scope does not match outer unit order")
        if tuple(generator_inputs_by_unit) != unit_ids:
            raise ValueError(
                "Texture service generator inputs do not match outer unit order"
            )
        material_textures: dict[str, dict[str, Any]] = {}
        common_reference_artifacts = generator_inputs_by_unit[
            unit_ids[0]
        ].reference_artifacts
        for unit_id in unit_ids:
            scope = unit_scope_by_id[unit_id]
            material_paths = tuple(
                str(path) for path in scope.get("material_prim_paths") or ()
            )
            if not material_paths:
                raise ValueError(
                    f"Texture service outer scope has no material: {unit_id}"
                )
            member_paths = tuple(
                str(path) for path in scope.get("member_prim_paths") or ()
            )
            inputs = generator_inputs_by_unit[unit_id]
            if inputs.execution_mode != "provider_generate" or inputs.provided_images:
                raise ValueError(
                    "Texture service client cannot apply outer-provided candidate images"
                )
            if inputs.reference_artifacts != common_reference_artifacts:
                raise ValueError(
                    "Texture service generator requires one common ordered "
                    "reference artifact set"
                )
            for material_path in material_paths:
                spec: dict[str, Any] = {
                    "prompt": inputs.prompt,
                    "detail_policy": inputs.detail_policy,
                    "material_path": material_path,
                }
                if member_paths:
                    spec["prim_paths"] = list(member_paths)
                previous = material_textures.get(material_path)
                if previous is not None and previous != spec:
                    raise ValueError(
                        "Texture outer plan assigns incompatible generator inputs "
                        f"to material {material_path}"
                    )
                material_textures[material_path] = spec
        return material_textures

    def _get_status(
        self, session_id: str, *, timeout: float | None = None
    ) -> dict[str, Any]:
        response = self._http.get(
            f"{self.base_url}/pipeline/{session_id}/status",
            timeout=self.timeout_seconds if timeout is None else timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Texture pipeline status response must be an object")
        return payload

    @staticmethod
    def _status_digest(status: Mapping[str, Any]) -> str:
        """Hash transition markers while excluding elapsed/remaining estimates."""

        payload = {
            key: status.get(key)
            for key in (
                "status",
                "current_step",
                "completed_steps",
                "updated_at",
                "failed_step",
                "execution_id",
            )
        }
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @staticmethod
    def _require_execution_protocol(status: Mapping[str, Any]) -> None:
        if "execution_id" not in status:
            raise RuntimeError(
                "Texture service does not advertise execution-id reconciliation; "
                "upgrade the service before resumable execution"
            )

    @staticmethod
    def _raise_for_terminal_status(status: Mapping[str, Any]) -> bool:
        state = status.get("status")
        if state not in TextureAgentServiceClient._TERMINAL_STATUSES:
            return False
        if state != "completed":
            message = status.get("error") or f"Texture pipeline {state}"
            raise RuntimeError(str(message))
        return True

    @staticmethod
    def _phase_execution_id(
        workflow_execution_id: str,
        phase: Literal["generation", "apply"],
    ) -> str:
        """Derive a stable service idempotency token for one remote phase."""

        canonical = (
            "content-agent-workflows.texture-service-phase.v1"
            f"\0{workflow_execution_id}\0{phase}"
        )
        # RegenerateRequest intentionally requires exactly 32 lowercase hex
        # characters. The truncated digest still provides a 128-bit token.
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

    @classmethod
    def _state_phase_execution_id(
        cls,
        state: _TextureRemoteExecutionState,
    ) -> str:
        phase: Literal["generation", "apply"]
        if state.phase == "generation":
            phase = "generation"
        elif state.phase == "apply":
            phase = "apply"
        else:
            raise RuntimeError("Texture remote execution is not in a dispatch phase")
        return cls._phase_execution_id(state.execution_id, phase)

    @classmethod
    def _completed_phase_execution_id(
        cls,
        state: _TextureRemoteExecutionState,
    ) -> str:
        completed_phase: Literal["generation", "apply"] = (
            "apply" if state.full_plan_regeneration else "generation"
        )
        return cls._phase_execution_id(state.execution_id, completed_phase)

    @staticmethod
    def _require_execution_status_owner(
        status: Mapping[str, Any],
        *,
        execution_id: str,
    ) -> None:
        observed_execution_id = status.get("execution_id")
        if observed_execution_id != execution_id:
            raise RuntimeError(
                "Texture service status belongs to a different execution: "
                f"expected {execution_id}, observed "
                f"{observed_execution_id!r}"
            )

    @staticmethod
    def _raise_if_cancelled(
        cancellation_check: Callable[[], bool] | None,
    ) -> None:
        if cancellation_check is not None and cancellation_check():
            raise TextureAgentServiceCancellationRequested(
                "Texture Agent service execution cancellation requested"
            )

    def _wait_for_terminal(
        self,
        session_id: str,
        *,
        baseline_status_digest: str | None = None,
        execution_id: str | None = None,
        orphan_replay_request: Mapping[str, Any] | None = None,
        cancellation_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        consecutive_failures = 0
        while True:
            self._raise_if_cancelled(cancellation_check)
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"Texture pipeline {session_id} did not complete within "
                    f"{self.timeout_seconds:g}s"
                )
            try:
                status = self._get_status(
                    session_id,
                    timeout=remaining_seconds,
                )
            except requests.RequestException as exc:
                status_code = getattr(exc.response, "status_code", None)
                retryable = (
                    status_code is None
                    or status_code in {408, 429}
                    or status_code >= 500
                )
                consecutive_failures += 1
                if (
                    not retryable
                    or consecutive_failures > self.max_status_poll_failures
                ):
                    raise
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise TimeoutError(
                        f"Texture pipeline {session_id} status polling did not "
                        f"recover within {self.timeout_seconds:g}s"
                    ) from exc
                self._raise_if_cancelled(cancellation_check)
                time.sleep(min(self.poll_interval_seconds, remaining_seconds))
                continue
            consecutive_failures = 0
            self._raise_if_cancelled(cancellation_check)
            if (
                baseline_status_digest is not None
                and self._status_digest(status) == baseline_status_digest
            ):
                remaining_seconds = deadline - time.monotonic()
                if remaining_seconds <= 0:
                    raise TimeoutError(
                        f"Texture pipeline {session_id} did not publish a new "
                        f"status within {self.timeout_seconds:g}s"
                    )
                self._raise_if_cancelled(cancellation_check)
                time.sleep(min(self.poll_interval_seconds, remaining_seconds))
                continue
            if execution_id is not None:
                self._require_execution_status_owner(
                    status,
                    execution_id=execution_id,
                )
            if self._raise_for_terminal_status(status):
                return status
            if orphan_replay_request is not None:
                self._post_regeneration_request(
                    session_id=session_id,
                    request=orphan_replay_request,
                )
                orphan_replay_request = None
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"Texture pipeline {session_id} did not complete within "
                    f"{self.timeout_seconds:g}s"
                )
            self._raise_if_cancelled(cancellation_check)
            time.sleep(min(self.poll_interval_seconds, remaining_seconds))

    def plan(self, request: TextureWorkflowRequest) -> TexturePlanDocument:
        metadata = request.metadata
        if metadata.get("source_dependencies"):
            raise ValueError(
                "Texture service requires a byte-self-contained source; external USD "
                "dependencies cannot be uploaded"
            )
        form: dict[str, str] = {
            "plan_only": "true",
            "user_prompt": request.intent,
            "auto_prompt_enabled": str(
                bool(metadata.get("auto_prompt_enabled", True))
            ).lower(),
            "detail_policy": str(metadata.get("detail_policy", "surface_only")),
            "discovery_mode": str(metadata.get("discovery_mode", "effective_bound")),
            "unit_mode": str(metadata.get("unit_mode", "per_material")),
        }
        for key in (
            "texture_backend",
            "texture_endpoint",
            "backend_engine",
            "texture_size",
            "seed",
            "uv_policy",
            "uv_scope",
            "operator_override_cap",
        ):
            value = metadata.get(key)
            if value is not None:
                form[key] = str(value)
        material_textures = metadata.get("material_textures")
        if material_textures is not None:
            form["material_textures_json"] = json.dumps(material_textures)
        backend_custom_parameters = metadata.get("backend_custom_parameters")
        if backend_custom_parameters is not None:
            if not isinstance(backend_custom_parameters, Mapping):
                raise ValueError("backend_custom_parameters must be an object")
            form["backend_custom_parameters_json"] = json.dumps(
                dict(backend_custom_parameters)
            )
        explicit_material_paths = metadata.get("explicit_material_paths")
        if explicit_material_paths is not None:
            form["explicit_material_paths_json"] = json.dumps(
                [str(path) for path in explicit_material_paths]
            )
        explicit_prim_paths = metadata.get("explicit_prim_paths")
        if explicit_prim_paths is not None:
            form["explicit_prim_paths_json"] = json.dumps(
                [str(path) for path in explicit_prim_paths]
            )

        if len(request.reference_artifacts) > 1:
            raise ValueError(
                "Texture service supports one uploaded reference image per session"
            )
        if request.reference_artifacts and (
            request.reference_artifacts[0].role != "appearance_reference"
        ):
            raise ValueError(
                "Texture service supports only the appearance_reference role"
            )
        source_asset = request.source_asset
        with ExitStack() as stack:
            files: dict[str, tuple[str, Any, str]] = {}
            if source_asset.startswith("s3://"):
                form["s3_uri"] = source_asset
            else:
                source_path = Path(source_asset)
                source_binding_payload = metadata.get("source_asset_binding")
                source_file: Any
                if source_binding_payload is None:
                    source_file = stack.enter_context(source_path.open("rb"))
                else:
                    source_binding = ExecutionArtifactBinding.model_validate(
                        source_binding_payload
                    )
                    if source_binding.path != str(source_path.expanduser().resolve()):
                        raise ValueError(
                            "Texture service source binding differs from upload path"
                        )
                    source_file = _verified_upload_snapshot(
                        source_path,
                        expected_sha256=source_binding.sha256,
                        expected_size_bytes=source_binding.size_bytes,
                        label="Texture service source asset",
                    )
                    stack.callback(source_file.close)
                files["usd_file"] = (
                    source_path.name,
                    source_file,
                    "application/octet-stream",
                )
            if request.reference_artifacts:
                reference = request.reference_artifacts[0].artifact
                reference_path = Path(reference.path).expanduser()
                reference_file = stack.enter_context(
                    _verified_upload_snapshot(
                        reference_path,
                        expected_sha256=reference.sha256,
                        expected_size_bytes=reference.size_bytes,
                        label="Texture service reference image",
                    )
                )
                files["reference_image_file"] = (
                    reference_path.resolve().name,
                    reference_file,
                    "application/octet-stream",
                )
            response = self._http.post(
                f"{self.base_url}/pipeline",
                data=form,
                files=files or None,
                timeout=self.timeout_seconds,
            )
        response.raise_for_status()
        session_id = str(response.json()["session_id"])
        self._wait_for_terminal(session_id)
        plan_response = self._http.get(
            f"{self.base_url}/pipeline/{session_id}/plan",
            timeout=self.timeout_seconds,
        )
        plan_response.raise_for_status()
        plan = TexturePlanDocument.model_validate(plan_response.json())
        self._session_by_plan[self._plan_key(plan)] = session_id
        return plan

    @staticmethod
    def _extract_texture_archive(content: bytes, output_dir: Path) -> tuple[Path, ...]:
        output_dir.mkdir(parents=True, exist_ok=True)
        extracted: list[Path] = []
        root = output_dir.resolve()
        with zipfile.ZipFile(BytesIO(content)) as archive:
            for info in archive.infolist():
                member = Path(info.filename)
                if info.is_dir() or member.is_absolute() or ".." in member.parts:
                    continue
                destination = (root / member).resolve()
                if not destination.is_relative_to(root):
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(info))
                extracted.append(destination)
        return tuple(extracted)

    @staticmethod
    def _path_belongs_to_unit(path: Path, unit_id: str) -> bool:
        return unit_id in path.parts or path.name.startswith(f"{unit_id}_")

    @staticmethod
    def _logical_unit_artifact_key(path: Path, unit_id: str) -> str:
        prefix = f"{unit_id}_"
        if path.name.startswith(prefix):
            return path.name[len(prefix) :]
        if unit_id in path.parts:
            unit_index = path.parts.index(unit_id)
            suffix = path.parts[unit_index + 1 :]
            if suffix:
                return "/".join(suffix)
        return path.name

    @classmethod
    def _unit_artifact_hashes(
        cls,
        paths: tuple[Path, ...],
        unit_id: str,
    ) -> dict[str, set[str]]:
        hashes: dict[str, set[str]] = {}
        for path in paths:
            key = cls._logical_unit_artifact_key(path, unit_id)
            digest = file_sha256(path)
            hashes.setdefault(key, set()).add(digest)
        return hashes

    @classmethod
    def _unit_package_hashes(
        cls,
        content: bytes,
        unit_id: str,
    ) -> dict[str, set[str]]:
        return cls._package_unit_hash_index(content, (unit_id,))[unit_id]

    @classmethod
    def _package_unit_hash_index(
        cls,
        package: bytes | Path,
        unit_ids: tuple[str, ...],
    ) -> dict[str, dict[str, set[str]]]:
        """Hash preserved-unit members in one bounded scan of one USDZ."""

        ordered_unit_ids = tuple(dict.fromkeys(unit_ids))
        hashes: dict[str, dict[str, set[str]]] = {
            unit_id: {} for unit_id in ordered_unit_ids
        }
        package_source = BytesIO(package) if isinstance(package, bytes) else package
        try:
            with zipfile.ZipFile(package_source) as archive:
                infos = archive.infolist()
                if len(infos) > DEFAULT_MAX_USDZ_MEMBERS:
                    raise RuntimeError(
                        "Texture service output USDZ exceeds the preserved-artifact "
                        "member limit"
                    )
                hashed_bytes = 0
                for info in infos:
                    path = Path(info.filename)
                    if info.is_dir():
                        continue
                    matching_unit_ids = tuple(
                        unit_id
                        for unit_id in ordered_unit_ids
                        if cls._path_belongs_to_unit(path, unit_id)
                    )
                    if not matching_unit_ids:
                        continue
                    if (
                        info.file_size < 0
                        or info.file_size > DEFAULT_MAX_USDZ_MEMBER_BYTES
                        or hashed_bytes + info.file_size
                        > DEFAULT_MAX_USDZ_EXTRACTED_BYTES
                    ):
                        raise RuntimeError(
                            "Texture service output USDZ exceeds the preserved-artifact "
                            "content limit"
                        )

                    digest = hashlib.sha256()
                    member_bytes = 0
                    with archive.open(info) as stream:
                        for chunk in iter(
                            lambda: stream.read(_PACKAGE_HASH_READ_BYTES),
                            b"",
                        ):
                            member_bytes += len(chunk)
                            if (
                                member_bytes > info.file_size
                                or member_bytes > DEFAULT_MAX_USDZ_MEMBER_BYTES
                                or hashed_bytes + member_bytes
                                > DEFAULT_MAX_USDZ_EXTRACTED_BYTES
                            ):
                                raise RuntimeError(
                                    "Texture service output USDZ exceeds the "
                                    "preserved-artifact content limit"
                                )
                            digest.update(chunk)
                    if member_bytes != info.file_size:
                        raise RuntimeError(
                            "Texture service output USDZ contains a truncated "
                            "preserved artifact"
                        )
                    hashed_bytes += member_bytes
                    member_digest = digest.hexdigest()
                    for unit_id in matching_unit_ids:
                        key = cls._logical_unit_artifact_key(path, unit_id)
                        hashes[unit_id].setdefault(key, set()).add(member_digest)
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                "Texture service output is not a readable USDZ archive"
            ) from exc
        return hashes

    @classmethod
    def _verify_preserved_texture_artifacts(
        cls,
        texture_paths: tuple[Path, ...],
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> None:
        for unit_id, artifact in preserved_artifacts.items():
            candidate_paths = tuple(
                path
                for path in texture_paths
                if cls._path_belongs_to_unit(path, unit_id)
            )
            if not candidate_paths:
                raise RuntimeError(
                    f"Texture service omitted preserved artifacts for {unit_id}"
                )
            preserved_paths = tuple(
                Path(path).resolve() for path in artifact.artifact_paths
            )
            if cls._unit_artifact_hashes(
                candidate_paths,
                unit_id,
            ) != cls._unit_artifact_hashes(preserved_paths, unit_id):
                raise RuntimeError(
                    f"Texture service changed preserved artifacts for {unit_id}"
                )

    @classmethod
    def _verify_preserved_package_artifacts(
        cls,
        package_content: bytes,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> None:
        if not preserved_artifacts:
            return
        previous_paths: dict[str, Path] = {}
        unit_ids_by_previous_path: dict[Path, list[str]] = {}
        for unit_id, artifact in preserved_artifacts.items():
            previous_output_path = artifact.metadata.get("output_asset_path")
            if not isinstance(previous_output_path, str) or not previous_output_path:
                raise RuntimeError(
                    "Preserved Texture artifact is missing its prior output package "
                    f"for {unit_id}"
                )
            try:
                resolved_previous_path = (
                    Path(previous_output_path).expanduser().resolve(strict=True)
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    "Preserved Texture artifact has an unreadable prior output "
                    f"package for {unit_id}"
                ) from exc
            previous_paths[unit_id] = resolved_previous_path
            unit_ids_by_previous_path.setdefault(resolved_previous_path, []).append(
                unit_id
            )

        preserved_unit_ids = tuple(preserved_artifacts)
        candidate_index = cls._package_unit_hash_index(
            package_content,
            preserved_unit_ids,
        )
        previous_indexes = {
            path: cls._package_unit_hash_index(path, tuple(unit_ids))
            for path, unit_ids in unit_ids_by_previous_path.items()
        }

        for unit_id in preserved_unit_ids:
            previous_hashes = previous_indexes[previous_paths[unit_id]][unit_id]
            candidate_hashes = candidate_index[unit_id]
            if not previous_hashes or not candidate_hashes:
                raise RuntimeError(
                    f"Texture output package omitted preserved artifacts for {unit_id}"
                )
            if candidate_hashes != previous_hashes:
                raise RuntimeError(
                    f"Texture output package changed preserved artifacts for {unit_id}"
                )

    @staticmethod
    def _generation_request(
        *,
        unit_ids: tuple[str, ...],
        full_plan_regeneration: bool,
        material_textures: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "steps": [
                "prepare_uvs",
                "generate_textures",
                "blend_textures",
            ],
        }
        if not full_plan_regeneration:
            request["steps"].append("apply_textures")
            request["texture_unit_ids"] = list(unit_ids)
        if material_textures:
            request["material_textures"] = {
                str(material): dict(spec)
                for material, spec in material_textures.items()
            }
        return request

    @classmethod
    def _completed_phase_request(
        cls,
        state: _TextureRemoteExecutionState,
    ) -> dict[str, Any]:
        if state.full_plan_regeneration:
            return {"steps": ["apply_textures"]}
        return cls._generation_request(
            unit_ids=state.unit_ids,
            full_plan_regeneration=False,
            material_textures=state.material_textures,
        )

    @staticmethod
    def _ordered_preserved_ids(
        plan: TexturePlanDocument,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> tuple[str, ...]:
        return tuple(
            unit_id
            for unit_id in plan.selected_unit_ids
            if unit_id in preserved_artifacts
        )

    def _validate_pending_execution(
        self,
        state: _TextureRemoteExecutionState,
        *,
        plan: TexturePlanDocument,
        session_id: str,
        unit_ids: tuple[str, ...],
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
        material_textures: Mapping[str, Mapping[str, Any]] | None,
    ) -> None:
        expected_preserved_ids = self._ordered_preserved_ids(
            plan,
            preserved_artifacts,
        )
        expected_generations = {
            unit_id: self._generation_by_unit.get(unit_id, 0) for unit_id in unit_ids
        }
        if state.session_id != session_id or state.plan_key != self._plan_key(plan):
            raise RuntimeError(
                "Checkpointed Texture execution does not match the service session"
            )
        if state.unit_ids != unit_ids:
            raise RuntimeError(
                "Checkpointed Texture execution unit IDs differ from workflow scope"
            )
        if state.preserved_unit_ids != expected_preserved_ids:
            raise RuntimeError(
                "Checkpointed Texture preserved unit IDs differ from workflow scope"
            )
        if state.generation_by_unit_before != expected_generations:
            raise RuntimeError(
                "Checkpointed Texture execution generation state has diverged"
            )
        if state.full_plan_regeneration != (unit_ids == plan.selected_unit_ids):
            raise RuntimeError(
                "Checkpointed Texture execution mode differs from workflow scope"
            )
        if material_textures is not None:
            expected_material_textures = {
                str(material): dict(spec)
                for material, spec in material_textures.items()
            }
            if state.material_textures != expected_material_textures:
                raise RuntimeError(
                    "Checkpointed Texture generator inputs differ from outer semantics"
                )

    def _persist_pending_execution(
        self,
        state: _TextureRemoteExecutionState,
        persist_resume_state: Callable[[], None],
    ) -> _TextureRemoteExecutionState:
        self._pending_execution = state
        persist_resume_state()
        return state

    def _start_or_restore_execution(
        self,
        *,
        plan: TexturePlanDocument,
        session_id: str,
        unit_ids: tuple[str, ...],
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
        material_textures: Mapping[str, Mapping[str, Any]],
        persist_resume_state: Callable[[], None],
        cancellation_check: Callable[[], bool] | None,
    ) -> _TextureRemoteExecutionState:
        if self._pending_execution is not None:
            self._validate_pending_execution(
                self._pending_execution,
                plan=plan,
                session_id=session_id,
                unit_ids=unit_ids,
                preserved_artifacts=preserved_artifacts,
                material_textures=material_textures,
            )
            return self._pending_execution

        self._raise_if_cancelled(cancellation_check)
        baseline_status = self._get_status(session_id)
        self._raise_if_cancelled(cancellation_check)
        self._require_execution_protocol(baseline_status)
        if baseline_status.get("status") not in self._TERMINAL_STATUSES:
            raise RuntimeError(
                "Texture service has untracked active work; refusing to dispatch "
                "another execution"
            )
        state = _TextureRemoteExecutionState(
            execution_id=uuid4().hex,
            session_id=session_id,
            plan_key=self._plan_key(plan),
            unit_ids=unit_ids,
            preserved_unit_ids=self._ordered_preserved_ids(
                plan,
                preserved_artifacts,
            ),
            generation_by_unit_before={
                unit_id: self._generation_by_unit.get(unit_id, 0)
                for unit_id in unit_ids
            },
            material_textures={
                str(material): dict(spec)
                for material, spec in material_textures.items()
            },
            full_plan_regeneration=unit_ids == plan.selected_unit_ids,
            phase="generation",
            baseline_status_digest=self._status_digest(baseline_status),
            baseline_execution_id=baseline_status.get("execution_id"),
        )
        return self._persist_pending_execution(state, persist_resume_state)

    def _wait_from_status(
        self,
        session_id: str,
        status: dict[str, Any],
        *,
        baseline_status_digest: str,
        execution_id: str,
        cancellation_check: Callable[[], bool] | None,
    ) -> dict[str, Any]:
        self._raise_if_cancelled(cancellation_check)
        if self._status_digest(status) != baseline_status_digest:
            self._require_execution_status_owner(
                status,
                execution_id=execution_id,
            )
            if self._raise_for_terminal_status(status):
                return status
        return self._wait_for_terminal(
            session_id,
            baseline_status_digest=baseline_status_digest,
            execution_id=execution_id,
            cancellation_check=cancellation_check,
        )

    def _post_regeneration_request(
        self,
        *,
        session_id: str,
        request: Mapping[str, Any],
    ) -> None:
        response = self._http.post(
            f"{self.base_url}/pipeline/{session_id}/regenerate",
            json=dict(request),
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

    def _run_pending_phase(
        self,
        state: _TextureRemoteExecutionState,
        *,
        request: dict[str, Any],
        persist_resume_state: Callable[[], None],
        cancellation_check: Callable[[], bool] | None,
    ) -> tuple[_TextureRemoteExecutionState, dict[str, Any]]:
        if state.phase not in {"generation", "apply"}:
            raise RuntimeError("Texture remote execution is not in a dispatch phase")
        if state.baseline_status_digest is None:
            raise RuntimeError("Texture remote execution has no status baseline")

        self._raise_if_cancelled(cancellation_check)
        current_status = self._get_status(state.session_id)
        self._raise_if_cancelled(cancellation_check)
        self._require_execution_protocol(current_status)
        remote_changed = (
            self._status_digest(current_status) != state.baseline_status_digest
        )
        if (
            not state.dispatched
            and remote_changed
            and current_status.get("status") == "completed"
            and current_status.get("execution_id") == state.baseline_execution_id
        ):
            state = self._persist_pending_execution(
                state.model_copy(
                    update={
                        "baseline_status_digest": self._status_digest(current_status),
                        "baseline_execution_id": current_status.get("execution_id"),
                    }
                ),
                persist_resume_state,
            )
            remote_changed = False
        baseline_status_digest = state.baseline_status_digest
        if baseline_status_digest is None:
            raise RuntimeError("Texture remote execution lost its status baseline")
        phase_execution_id = self._state_phase_execution_id(state)
        owned_request = {**request, "execution_id": phase_execution_id}
        if state.dispatched or remote_changed:
            if remote_changed:
                self._require_execution_status_owner(
                    current_status,
                    execution_id=phase_execution_id,
                )
            # Replaying every resumed phase validates the execution token's payload
            # digest. It also lets a restarted service atomically requeue an owned
            # pending/running/cancelling execution whose in-memory job disappeared.
            self._post_regeneration_request(
                session_id=state.session_id,
                request=owned_request,
            )
            if not state.dispatched:
                state = self._persist_pending_execution(
                    state.model_copy(update={"dispatched": True}),
                    persist_resume_state,
                )
            return state, self._wait_from_status(
                state.session_id,
                current_status,
                baseline_status_digest=baseline_status_digest,
                execution_id=phase_execution_id,
                cancellation_check=cancellation_check,
            )

        self._raise_if_cancelled(cancellation_check)
        try:
            self._post_regeneration_request(
                session_id=state.session_id,
                request=owned_request,
            )
        except requests.RequestException as exc:
            status_code = getattr(exc.response, "status_code", None)
            retryable = (
                status_code is None or status_code in {408, 429} or status_code >= 500
            )
            if not retryable:
                raise
            reconciled_status = self._get_status(state.session_id)
            if self._status_digest(reconciled_status) == state.baseline_status_digest:
                raise
            return state, self._wait_from_status(
                state.session_id,
                reconciled_status,
                baseline_status_digest=baseline_status_digest,
                execution_id=phase_execution_id,
                cancellation_check=cancellation_check,
            )

        state = self._persist_pending_execution(
            state.model_copy(update={"dispatched": True}),
            persist_resume_state,
        )
        return state, self._wait_for_terminal(
            state.session_id,
            baseline_status_digest=state.baseline_status_digest,
            execution_id=phase_execution_id,
            cancellation_check=cancellation_check,
        )

    def _transition_execution_phase(
        self,
        state: _TextureRemoteExecutionState,
        *,
        phase: Literal["apply", "collect"],
        completed_status: Mapping[str, Any],
        persist_resume_state: Callable[[], None],
    ) -> _TextureRemoteExecutionState:
        state = state.model_copy(
            update={
                "phase": phase,
                "baseline_status_digest": (
                    self._status_digest(completed_status) if phase == "apply" else None
                ),
                "baseline_execution_id": (
                    completed_status.get("execution_id") if phase == "apply" else None
                ),
                "dispatched": False,
            }
        )
        return self._persist_pending_execution(state, persist_resume_state)

    def _collect_execution_result(
        self,
        state: _TextureRemoteExecutionState,
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
        cancellation_check: Callable[[], bool] | None,
    ) -> TextureExecutionResult:
        session_id = state.session_id
        unit_ids = state.unit_ids
        completed_phase_execution_id = self._completed_phase_execution_id(state)
        completed_phase_request = {
            **self._completed_phase_request(state),
            "execution_id": completed_phase_execution_id,
        }
        self._wait_for_terminal(
            session_id,
            execution_id=completed_phase_execution_id,
            orphan_replay_request=completed_phase_request,
            cancellation_check=cancellation_check,
        )
        self._raise_if_cancelled(cancellation_check)
        results_response = self._http.get(
            f"{self.base_url}/pipeline/{session_id}/results",
            timeout=self.timeout_seconds,
        )
        results_response.raise_for_status()
        results = results_response.json()

        self._raise_if_cancelled(cancellation_check)
        # Reuse one deterministic collect directory across crash/retry attempts
        # instead of leaving a new UUID-named partial directory on every resume.
        service_dir = (
            output_dir
            / "texture_agent_service"
            / "executions"
            / completed_phase_execution_id
        )
        texture_response = self._http.get(
            f"{self.base_url}/artifacts/{session_id}/textures",
            timeout=self.timeout_seconds,
        )
        texture_response.raise_for_status()
        texture_paths = self._extract_texture_archive(
            texture_response.content,
            service_dir / "textures",
        )
        self._verify_preserved_texture_artifacts(
            texture_paths,
            preserved_artifacts,
        )
        self._raise_if_cancelled(cancellation_check)
        output_response = self._http.get(
            f"{self.base_url}/artifacts/{session_id}/output",
            timeout=self.timeout_seconds,
        )
        output_response.raise_for_status()
        self._verify_preserved_package_artifacts(
            output_response.content,
            preserved_artifacts,
        )
        output_asset = service_dir / "textured_output.usdz"
        output_asset.parent.mkdir(parents=True, exist_ok=True)
        output_asset.write_bytes(output_response.content)

        self._raise_if_cancelled(cancellation_check)
        final_status = self._get_status(session_id)
        self._raise_if_cancelled(cancellation_check)
        self._require_execution_status_owner(
            final_status,
            execution_id=completed_phase_execution_id,
        )
        if not self._raise_for_terminal_status(final_status):
            raise RuntimeError(
                "Texture service execution changed while collecting artifacts"
            )

        artifacts: list[TextureUnitArtifact] = []
        completed_generations: dict[str, int] = {}
        for unit_id in unit_ids:
            matching_paths = tuple(
                str(path)
                for path in texture_paths
                if self._path_belongs_to_unit(path, unit_id)
            )
            if not matching_paths:
                raise RuntimeError(
                    f"Texture service returned no downloadable artifact for {unit_id}"
                )
            generation = state.generation_by_unit_before[unit_id] + 1
            completed_generations[unit_id] = generation
            artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=matching_paths,
                    generation=generation,
                    metadata={
                        "backend": "texture-agent-service",
                        "output_asset_path": str(output_asset.resolve()),
                    },
                )
            )

        # Commit generation counters only after every requested artifact has
        # been validated. A partial or interrupted collect must remain
        # resumable from the durable pre-execution generation snapshot.
        self._generation_by_unit.update(completed_generations)
        stats = results.get("stats") or {}
        return TextureExecutionResult(
            requested_unit_ids=unit_ids,
            unit_artifacts=tuple(artifacts),
            output_asset_path=str(output_asset.resolve()),
            cache_hit_unit_ids=tuple(stats.get("cache_hit_unit_ids") or ()),
            retry_count=int(stats.get("retry_count") or 0),
            metadata={
                "backend": "texture-agent-service",
                "live_backend_invoked": True,
                "session_id": session_id,
                "execution_id": completed_phase_execution_id,
                "workflow_execution_id": state.execution_id,
                "stats": stats,
            },
        )

    def execute_resumable(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
        persist_resume_state: Callable[[], None],
        cancellation_check: Callable[[], bool] | None = None,
        material_textures: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> TextureExecutionResult:
        """Execute or adopt an exact-unit remote run from durable client state."""

        if not unit_ids or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_ids must be a non-empty unique sequence")
        unknown_ids = set(unit_ids) - set(plan.selected_unit_ids)
        if unknown_ids:
            raise ValueError(
                f"executor received unit IDs outside the plan: {unknown_ids}"
            )
        if set(unit_ids) & set(preserved_artifacts):
            raise ValueError("requested regeneration IDs overlap preserved artifacts")
        if set(unit_ids) | set(preserved_artifacts) != set(plan.selected_unit_ids):
            raise ValueError(
                "requested and preserved unit IDs must partition the Texture plan"
            )
        resolved_material_textures = {
            str(material): dict(spec)
            for material, spec in (material_textures or {}).items()
        }
        session_id = self._session_by_plan.get(self._plan_key(plan))
        if session_id is None:
            raise ValueError("plan was not created by this service client")

        state = self._start_or_restore_execution(
            plan=plan,
            session_id=session_id,
            unit_ids=unit_ids,
            preserved_artifacts=preserved_artifacts,
            material_textures=resolved_material_textures,
            persist_resume_state=persist_resume_state,
            cancellation_check=cancellation_check,
        )
        if state.phase == "generation":
            state, completed_status = self._run_pending_phase(
                state,
                request=self._generation_request(
                    unit_ids=unit_ids,
                    full_plan_regeneration=state.full_plan_regeneration,
                    material_textures=state.material_textures,
                ),
                persist_resume_state=persist_resume_state,
                cancellation_check=cancellation_check,
            )
            if state.full_plan_regeneration:
                state = self._transition_execution_phase(
                    state,
                    phase="apply",
                    completed_status=completed_status,
                    persist_resume_state=persist_resume_state,
                )
            else:
                state = self._transition_execution_phase(
                    state,
                    phase="collect",
                    completed_status=completed_status,
                    persist_resume_state=persist_resume_state,
                )

        if state.phase == "apply":
            state, completed_status = self._run_pending_phase(
                state,
                request={"steps": ["apply_textures"]},
                persist_resume_state=persist_resume_state,
                cancellation_check=cancellation_check,
            )
            state = self._transition_execution_phase(
                state,
                phase="collect",
                completed_status=completed_status,
                persist_resume_state=persist_resume_state,
            )

        result = self._collect_execution_result(
            state,
            output_dir=output_dir,
            preserved_artifacts=preserved_artifacts,
            cancellation_check=cancellation_check,
        )
        # Do not checkpoint the clear here. If the process stops before the
        # workflow records ``result``, the durable collect phase lets resume
        # reconstruct the same result without another remote submission.
        self._pending_execution = None
        return result

    def execute_outer_plan(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        unit_scope_by_id: Mapping[str, Mapping[str, Any]],
        generator_inputs_by_unit: Mapping[str, TextureGeneratorInputs],
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
        persist_resume_state: Callable[[], None],
    ) -> TextureExecutionResult:
        """Execute a service session with exact outer-authored material semantics."""

        material_textures = self.material_textures_for_outer_plan(
            unit_ids,
            unit_scope_by_id=unit_scope_by_id,
            generator_inputs_by_unit=generator_inputs_by_unit,
        )
        return self.execute_resumable(
            plan,
            unit_ids,
            output_dir=output_dir,
            preserved_artifacts=preserved_artifacts,
            persist_resume_state=persist_resume_state,
            material_textures=material_textures,
        )

    def can_reconcile_authorized_execution(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> bool:
        """Prove that resume can adopt one exact pending service execution."""

        state = self._pending_execution
        if state is None:
            return False
        session_id = self._session_by_plan.get(self._plan_key(plan))
        if session_id is None:
            return False
        self._validate_pending_execution(
            state,
            plan=plan,
            session_id=session_id,
            unit_ids=unit_ids,
            preserved_artifacts=preserved_artifacts,
            material_textures=None,
        )
        return True

    def execute(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> TextureExecutionResult:
        """Execute directly; workflows use ``execute_resumable`` to checkpoint."""

        return self.execute_resumable(
            plan,
            unit_ids,
            output_dir=output_dir,
            preserved_artifacts=preserved_artifacts,
            persist_resume_state=lambda: None,
        )

    def export_resume_state(self, plan: TexturePlanDocument) -> dict[str, Any]:
        """Persist the service session and per-unit generation counters."""

        session_id = self._session_by_plan.get(self._plan_key(plan))
        if session_id is None:
            raise ValueError("plan was not created or restored by this service client")
        return {
            "session_id": session_id,
            "generation_by_unit": dict(self._generation_by_unit),
            "pending_execution": (
                self._pending_execution.model_dump(mode="json")
                if self._pending_execution is not None
                else None
            ),
        }

    def restore_resume_state(
        self,
        plan: TexturePlanDocument,
        state: Mapping[str, Any],
    ) -> None:
        """Reconnect a new client instance to a durable Texture service session."""

        session_id = state.get("session_id")
        if not isinstance(session_id, str) or not session_id:
            raise ValueError("Texture service resume state requires session_id")
        generations = state.get("generation_by_unit") or {}
        if not isinstance(generations, Mapping):
            raise ValueError("Texture service resume generations must be an object")
        restored = {str(key): int(value) for key, value in generations.items()}
        if set(restored) - set(plan.selected_unit_ids):
            raise ValueError(
                "Texture service resume state contains IDs outside the plan"
            )
        if any(value < 0 for value in restored.values()):
            raise ValueError("Texture service resume generations must be non-negative")
        pending_payload = state.get("pending_execution")
        if pending_payload is not None and not isinstance(pending_payload, Mapping):
            raise ValueError("Texture service pending execution must be an object")
        pending_execution = (
            _TextureRemoteExecutionState.model_validate(pending_payload)
            if pending_payload is not None
            else None
        )
        if pending_execution is not None:
            if (
                pending_execution.session_id != session_id
                or pending_execution.plan_key != self._plan_key(plan)
            ):
                raise ValueError(
                    "Texture service pending execution does not match the session plan"
                )
            selected_ids = set(plan.selected_unit_ids)
            if (
                set(pending_execution.unit_ids)
                | set(pending_execution.preserved_unit_ids)
                != selected_ids
            ):
                raise ValueError(
                    "Texture service pending execution does not partition the plan"
                )
            if pending_execution.generation_by_unit_before != {
                unit_id: restored.get(unit_id, 0)
                for unit_id in pending_execution.unit_ids
            }:
                raise ValueError(
                    "Texture service pending execution generation state has diverged"
                )

        plan_response = self._http.get(
            f"{self.base_url}/pipeline/{session_id}/plan",
            timeout=self.timeout_seconds,
        )
        plan_response.raise_for_status()
        service_plan = TexturePlanDocument.model_validate(plan_response.json())
        if service_plan.model_dump(
            mode="json",
            exclude_none=False,
        ) != plan.model_dump(mode="json", exclude_none=False):
            raise ValueError(
                "Texture service session plan does not match the checkpointed plan"
            )

        self._session_by_plan[self._plan_key(plan)] = session_id
        self._generation_by_unit = restored
        self._pending_execution = pending_execution
