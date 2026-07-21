# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner/executor client contract plus real and mock implementations."""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
from collections.abc import Mapping
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

import requests
from pydantic import BaseModel, ConfigDict

from .models import (
    TextureExecutionResult,
    TexturePlanDocument,
    TextureUnitArtifact,
    TextureWorkflowRequest,
)


class TexturePlannerExecutorClient(Protocol):
    """Workflow-facing boundary that WP7 may adapt to the real service."""

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


class TextureAgentServiceClient:
    """Adapter for the real Texture Agent plan/regenerate REST contract.

    ``max_status_poll_failures`` tolerates that many consecutive network,
    408, 429, or 5xx failures and raises the original error on the next one.
    """

    _TERMINAL_STATUSES = {"completed", "failed", "cancelled"}

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

    @staticmethod
    def _plan_key(plan: TexturePlanDocument) -> str:
        payload = plan.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _wait_for_terminal(self, session_id: str) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        consecutive_failures = 0
        while True:
            remaining_seconds = deadline - time.monotonic()
            if remaining_seconds <= 0:
                raise TimeoutError(
                    f"Texture pipeline {session_id} did not complete within "
                    f"{self.timeout_seconds:g}s"
                )
            try:
                response = self._http.get(
                    f"{self.base_url}/pipeline/{session_id}/status",
                    timeout=remaining_seconds,
                )
                response.raise_for_status()
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
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Texture pipeline {session_id} status polling did not "
                        f"recover within {self.timeout_seconds:g}s"
                    ) from exc
                time.sleep(self.poll_interval_seconds)
                continue
            consecutive_failures = 0
            status: dict[str, Any] = response.json()
            state = status.get("status")
            if state in self._TERMINAL_STATUSES:
                if state != "completed":
                    message = status.get("error") or f"Texture pipeline {state}"
                    raise RuntimeError(str(message))
                return status
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Texture pipeline {session_id} did not complete within "
                    f"{self.timeout_seconds:g}s"
                )
            time.sleep(self.poll_interval_seconds)

    def plan(self, request: TextureWorkflowRequest) -> TexturePlanDocument:
        metadata = request.metadata
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
            "operator_override_cap",
        ):
            value = metadata.get(key)
            if value is not None:
                form[key] = str(value)
        material_textures = metadata.get("material_textures")
        if material_textures is not None:
            form["material_textures_json"] = json.dumps(material_textures)
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

        source_asset = request.source_asset
        if source_asset.startswith("s3://"):
            form["s3_uri"] = source_asset
            response = self._http.post(
                f"{self.base_url}/pipeline",
                data=form,
                timeout=self.timeout_seconds,
            )
        else:
            source_path = Path(source_asset)
            with source_path.open("rb") as source_file:
                response = self._http.post(
                    f"{self.base_url}/pipeline",
                    data=form,
                    files={
                        "usd_file": (
                            source_path.name,
                            source_file,
                            "application/octet-stream",
                        )
                    },
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

    def execute(
        self,
        plan: TexturePlanDocument,
        unit_ids: tuple[str, ...],
        *,
        output_dir: Path,
        preserved_artifacts: Mapping[str, TextureUnitArtifact],
    ) -> TextureExecutionResult:
        if not unit_ids or len(unit_ids) != len(set(unit_ids)):
            raise ValueError("unit_ids must be a non-empty unique sequence")
        unknown_ids = set(unit_ids) - set(plan.selected_unit_ids)
        if unknown_ids:
            raise ValueError(
                f"executor received unit IDs outside the plan: {unknown_ids}"
            )
        if set(unit_ids) & set(preserved_artifacts):
            raise ValueError("requested regeneration IDs overlap preserved artifacts")
        session_id = self._session_by_plan.get(self._plan_key(plan))
        if session_id is None:
            raise ValueError("plan was not created by this service client")

        response = self._http.post(
            f"{self.base_url}/pipeline/{session_id}/regenerate",
            json={
                "steps": [
                    "prepare_uvs",
                    "generate_textures",
                    "blend_textures",
                    "apply_textures",
                ],
                "texture_unit_ids": list(unit_ids),
            },
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        self._wait_for_terminal(session_id)
        results_response = self._http.get(
            f"{self.base_url}/pipeline/{session_id}/results",
            timeout=self.timeout_seconds,
        )
        results_response.raise_for_status()
        results = results_response.json()

        service_dir = output_dir / "texture_agent_service"
        texture_response = self._http.get(
            f"{self.base_url}/artifacts/{session_id}/textures",
            timeout=self.timeout_seconds,
        )
        texture_response.raise_for_status()
        texture_paths = self._extract_texture_archive(
            texture_response.content,
            service_dir / "textures",
        )
        output_response = self._http.get(
            f"{self.base_url}/artifacts/{session_id}/output",
            timeout=self.timeout_seconds,
        )
        output_response.raise_for_status()
        output_asset = service_dir / "textured_output.usdz"
        output_asset.parent.mkdir(parents=True, exist_ok=True)
        output_asset.write_bytes(output_response.content)

        artifacts: list[TextureUnitArtifact] = []
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
            generation = self._generation_by_unit.get(unit_id, 0) + 1
            self._generation_by_unit[unit_id] = generation
            artifacts.append(
                TextureUnitArtifact(
                    unit_id=unit_id,
                    artifact_paths=matching_paths,
                    generation=generation,
                    metadata={"backend": "texture-agent-service"},
                )
            )

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
                "stats": stats,
            },
        )
