# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import asyncio
import hashlib
import io
import json
import threading
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringCapabilityReport,
    GeometryAuthoringExportRequest,
    GeometryAuthoringParameterValue,
    GeometryAuthoringProviderIdentity,
    GeometryAuthoringProviderReceipt,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometryCoordinateSystem,
    GeometryPartBinding,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySemanticParameter,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    geometry_authoring_request_digest,
    geometry_source_bundle_id,
    resolve_semantic_parameter_overrides,
    validate_geometry_source_bundle_identity,
)
from pydantic import SecretStr, ValidationError

from geometry_agent_service import main as service_main
from geometry_agent_service.config import Settings, settings
from geometry_agent_service.main import app
from geometry_agent_service.models import ArtifactRecord, GeometryGenerationRequest
from geometry_agent_service.storage import JobStore, WorkspaceStorage

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _canonical_bundle(bundle: GeometrySourceBundle) -> GeometrySourceBundle:
    return bundle.model_copy(
        update={
            "bundle_id": geometry_source_bundle_id(
                provider=bundle.producer,
                source_revision=bundle.source_revision,
                coordinate_system=bundle.coordinate_system,
                representations=bundle.representations,
                parts=bundle.parts,
                parameters=bundle.parameters,
                verification_assertions=bundle.verification_assertions,
                provenance=bundle.provenance,
                rights=bundle.rights,
            )
        }
    )


def _single_artifact_bundle(
    path: Path,
    *,
    sha256: str | None = None,
) -> GeometrySourceBundle:
    content = path.read_bytes()
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    representation = GeometryRepresentationBinding(
        representation_id="render",
        role="render_geometry",
        format="usda",
        media_type="model/vnd.usda",
        artifact=GeometryArtifactBinding(
            path=str(path),
            sha256=sha256 or hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
        ),
    )
    bundle = GeometrySourceBundle(
        bundle_id="a" * 64,
        producer=provider,
        source_revision="fixture-revision",
        coordinate_system=GeometryCoordinateSystem(meters_per_unit=1.0),
        representations=(representation,),
        provenance=GeometrySourceProvenance(request_digest="b" * 64),
        rights=GeometryRightsAssertion(assertion="Authorized fixture output."),
    )
    return _canonical_bundle(bundle)


def _artifact_record(
    filename: str,
    marker: str,
    *,
    bundle_member: str | None = None,
) -> ArtifactRecord:
    digest = marker * 64
    return ArtifactRecord(
        artifact_id=f"sha256:{digest}:{digest}",
        sha256=digest,
        filename=filename,
        bundle_member=bundle_member or filename,
        media_type="application/octet-stream",
        size_bytes=1,
    )


def test_source_identity_includes_complete_bundle_artifacts(tmp_path: Path) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    geometry = _artifact_record("geometry.usda", "a")
    manifest = _artifact_record("geometry.source.json", "b")
    provenance = _artifact_record(
        "input.png",
        "c",
        bundle_member="provenance/input.png",
    )

    legacy = storage._create_source_record(
        geometry,
        role="geometry_source",
        archive_entrypoint=None,
        source_bundle_manifest=manifest,
        source_bundle_artifacts=[geometry],
        source_representation_id="render",
    )
    complete = storage._create_source_record(
        geometry,
        role="geometry_source",
        archive_entrypoint=None,
        source_bundle_manifest=manifest,
        source_bundle_artifacts=[geometry, provenance],
        source_representation_id="render",
    )

    assert complete.source_id != legacy.source_id
    assert complete.source_bundle_artifacts == [geometry, provenance]


def test_settings_require_an_explicit_remote_render_endpoint() -> None:
    with pytest.raises(ValidationError, match="render_remote_base_url"):
        Settings(render_backend="remote")

    with pytest.raises(ValidationError, match="render_remote_api_key"):
        Settings(
            render_backend="remote",
            render_remote_base_url="https://render.example/v1",
        )
    with pytest.raises(ValidationError, match="render_remote_api_key"):
        Settings(
            render_backend="remote",
            render_remote_base_url="https://render.example/v1",
            render_remote_api_key=SecretStr(""),
        )

    configured = Settings(
        render_backend="remote",
        render_remote_base_url="https://render.example/v1",
        render_remote_api_key=SecretStr("render-key"),
    )
    assert configured.render_remote_base_url == "https://render.example/v1"
    assert configured.render_remote_allow_unauthenticated_identity is False

    trusted_sidecar = Settings(
        render_backend="remote",
        render_remote_base_url="http://render-sidecar:8011",
        render_remote_allow_unauthenticated_identity=True,
    )
    assert trusted_sidecar.render_remote_api_key is None
    assert trusted_sidecar.render_remote_allow_unauthenticated_identity is True

    with pytest.raises(ValidationError, match="render_backend=remote"):
        Settings(render_remote_base_url="https://render.example/v1")
    with pytest.raises(ValidationError, match="render_backend=remote"):
        Settings(render_remote_allow_unauthenticated_identity=True)


def test_settings_load_trusted_unauthenticated_render_sidecar_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GEOMETRY_AGENT_SERVICE_RENDER_BACKEND", "remote")
    monkeypatch.setenv(
        "GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_BASE_URL",
        "http://trusted-render-sidecar:8011",
    )
    monkeypatch.setenv(
        "GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_ALLOW_UNAUTHENTICATED_IDENTITY",
        "true",
    )
    monkeypatch.delenv("GEOMETRY_AGENT_SERVICE_RENDER_REMOTE_API_KEY", raising=False)

    configured = Settings()

    assert configured.render_remote_api_key is None
    assert configured.render_remote_allow_unauthenticated_identity is True


def test_job_recovery_is_scoped_to_the_owning_service_instance(tmp_path: Path) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    first_instance = JobStore(storage, owner_instance_id="pod-a")
    job = first_instance.running(first_instance.create("generation"))

    second_instance = JobStore(storage, owner_instance_id="pod-b")
    assert second_instance.recover_interrupted() == 0
    assert second_instance.get(job.job_id).status == "running"

    restarted_first_instance = JobStore(storage, owner_instance_id="pod-a")
    assert restarted_first_instance.recover_interrupted() == 1
    recovered = restarted_first_instance.get(job.job_id)
    assert recovered.status == "failed"
    assert recovered.error is not None
    assert recovered.error.code == "service_restarted"


def test_job_recovery_reclaims_a_stale_foreign_owner(tmp_path: Path) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    started = datetime(2026, 1, 1, tzinfo=UTC)
    first_instance = JobStore(
        storage,
        owner_instance_id="pod-a",
        owner_lease_timeout_seconds=60,
    )
    first_instance.heartbeat(observed_at=started)
    job = first_instance.running(first_instance.create("generation"))
    second_instance = JobStore(
        storage,
        owner_instance_id="pod-b",
        owner_lease_timeout_seconds=60,
    )

    assert second_instance.recover_interrupted(now=started + timedelta(seconds=59)) == 0
    assert second_instance.recover_interrupted(now=started + timedelta(seconds=61)) == 1
    assert second_instance.get(job.job_id).status == "failed"


def test_settings_require_complete_generic_delegated_worker_configuration() -> None:
    with pytest.raises(ValidationError, match="configured together"):
        Settings(
            delegated_authoring_provider_id="private-authoring-worker",
        )
    with pytest.raises(ValidationError, match="configured together"):
        Settings(
            delegated_authoring_endpoint_url="https://workers.example/private",
            delegated_authoring_rights_assertion="Authorized worker output.",
        )

    configured = Settings(
        delegated_authoring_provider_id="private-authoring-worker",
        delegated_authoring_endpoint_url="https://workers.example/private",
        delegated_authoring_bearer_token=SecretStr("worker-token"),
        delegated_authoring_rights_assertion="Authorized worker output.",
    )
    assert configured.delegated_authoring_provider_id == "private-authoring-worker"


@pytest.mark.parametrize(
    "provider_id",
    (
        "build123d-http",
        "forgecad-http",
    ),
)
def test_settings_reject_generic_workers_using_built_in_provider_ids(
    provider_id: str,
) -> None:
    with pytest.raises(ValidationError, match="must not reuse a built-in provider ID"):
        Settings(
            delegated_authoring_provider_id=provider_id,
            delegated_authoring_endpoint_url="https://workers.example/generic",
            delegated_authoring_rights_assertion="Authorized generic worker output.",
        )


def test_settings_require_explicit_forgecad_automated_use_authorization() -> None:
    with pytest.raises(ValidationError, match="automated_use_authorized"):
        Settings(
            forgecad_authoring_endpoint_url="https://workers.example/forgecad",
            forgecad_authoring_rights_assertion="Licensed worker output.",
        )

    configured = Settings(
        forgecad_authoring_endpoint_url="https://workers.example/forgecad",
        forgecad_authoring_rights_assertion="Licensed worker output.",
        forgecad_automated_use_authorized=True,
    )
    assert configured.forgecad_automated_use_authorized is True
    assert configured.forgecad_authoring_supported_formats == ("step",)
    assert configured.forgecad_authoring_supports_revision is False
    assert configured.forgecad_authoring_supports_export is False
    assert configured.forgecad_authoring_max_family_variants == 0


def test_settings_reject_incoherent_forgecad_capability_claims() -> None:
    with pytest.raises(ValidationError, match="text or image"):
        Settings(
            forgecad_authoring_supports_text=False,
            forgecad_authoring_supports_image=False,
        )
    with pytest.raises(ValidationError, match="semantic parameter support"):
        Settings(forgecad_authoring_supports_parameter_definitions=True)
    with pytest.raises(ValidationError, match="revision and parameter definitions"):
        Settings(forgecad_authoring_max_family_variants=1)
    with pytest.raises(ValidationError, match="connect timeout"):
        Settings(
            forgecad_authoring_connect_timeout_seconds=60.0,
            forgecad_authoring_read_timeout_seconds=30.0,
        )
    with pytest.raises(ValidationError, match="less than or equal to 900"):
        Settings(forgecad_authoring_read_timeout_seconds=901.0)

    configured = Settings(forgecad_authoring_read_timeout_seconds=900.0)
    assert configured.forgecad_authoring_read_timeout_seconds == 900.0


@pytest.mark.parametrize(
    "parameters",
    (
        {"unsafe parameter": 10.0},
        {"label": "x" * 4_097},
        {f"parameter-{index}": index for index in range(257)},
        {"width": float("nan")},
    ),
)
def test_generation_request_rejects_unbounded_or_unsafe_parameters(
    parameters: dict[str, object],
) -> None:
    with pytest.raises(ValidationError):
        GeometryGenerationRequest.model_validate(
            {
                "provider_id": "provider.v1",
                "prompt": "Create a body.",
                "parameters": parameters,
            }
        )


def test_generation_request_rejects_duplicate_image_sources() -> None:
    source_id = "src_" + "a" * 64
    with pytest.raises(ValidationError, match="image_source_ids must be unique"):
        GeometryGenerationRequest(
            provider_id="provider.v1",
            image_source_ids=[source_id, source_id],
        )


class FakeConnector:
    def __init__(self, output_root: Path) -> None:
        self.output_root = output_root

    @property
    def provider_id(self) -> str:
        return "fixture-provider"

    def capabilities(self) -> GeometryAuthoringCapabilityReport:
        return GeometryAuthoringCapabilityReport(
            provider=GeometryAuthoringProviderIdentity(
                provider_id=self.provider_id,
                provider_version="1.0",
            ),
            operations=("generate", "revise", "export"),
            input_modalities=("text",),
            output_formats=("usda",),
            supports_semantic_parameters=True,
            returns_native_source=False,
            features=(
                "immutable_revisions",
                "semantic_parameters",
                "parameter_definitions",
                "parameter_families",
            ),
            max_family_variants=64,
            max_reference_artifacts=0,
            max_prompt_characters=32_768,
        )

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        self.output_root.mkdir(parents=True, exist_ok=False)
        output = self.output_root / "fixture.usda"
        content = b"#usda 1.0\n"
        output.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        request_digest = geometry_authoring_request_digest(request)
        provider = self.capabilities().provider
        bundle = _canonical_bundle(
            GeometrySourceBundle(
                bundle_id="a" * 64,
                producer=provider,
                source_revision="fixture-revision",
                coordinate_system=GeometryCoordinateSystem(meters_per_unit=1.0),
                representations=(
                    GeometryRepresentationBinding(
                        representation_id="render",
                        role="render_geometry",
                        format="usda",
                        media_type="model/vnd.usda",
                        artifact=GeometryArtifactBinding(
                            path=str(output),
                            sha256=digest,
                            size_bytes=len(content),
                        ),
                    ),
                ),
                parameters=(
                    GeometrySemanticParameter(
                        name="width_mm",
                        value=next(
                            (
                                item.value
                                for item in request.parameters
                                if item.name == "width_mm"
                            ),
                            40.0,
                        ),
                        unit="mm",
                        minimum=20.0,
                        maximum=100.0,
                        step=1.0,
                        semantic_role="overall_width",
                        effects=("exact_geometry",),
                    ),
                ),
                provenance=GeometrySourceProvenance(request_digest=request_digest),
                rights=GeometryRightsAssertion(
                    assertion="Fixture output is authorized."
                ),
            )
        )
        return GeometryAuthoringProviderReceipt(
            provider=provider,
            operation="generate",
            request_digest=request_digest,
            disposition="succeeded",
            source_bundle=bundle,
        )

    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        safe_request_id = request.request_id.replace(":", "_")
        output_root = self.output_root.parent / f"provider-revision-{safe_request_id}"
        output_root.mkdir(parents=True, exist_ok=False)
        output = output_root / "fixture-revised.usda"
        content = b'#usda 1.0\ndef Xform "Revised" {}\n'
        output.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        request_digest = geometry_authoring_request_digest(request)
        provider = self.capabilities().provider
        bundle = _canonical_bundle(
            GeometrySourceBundle(
                bundle_id="b" * 64,
                producer=provider,
                source_revision="fixture-revision-2",
                coordinate_system=GeometryCoordinateSystem(meters_per_unit=1.0),
                representations=(
                    GeometryRepresentationBinding(
                        representation_id="render",
                        role="render_geometry",
                        format="usda",
                        media_type="model/vnd.usda",
                        artifact=GeometryArtifactBinding(
                            path=str(output),
                            sha256=digest,
                            size_bytes=len(content),
                        ),
                    ),
                ),
                parameters=resolve_semantic_parameter_overrides(
                    request.source_bundle.parameters,
                    request.parameter_overrides,
                ),
                provenance=GeometrySourceProvenance(
                    request_digest=request_digest,
                    parent_bundle_id=request.source_bundle.bundle_id,
                ),
                rights=GeometryRightsAssertion(
                    assertion="Fixture output is authorized."
                ),
            )
        )
        return GeometryAuthoringProviderReceipt(
            provider=provider,
            operation="revise",
            request_digest=request_digest,
            disposition="succeeded",
            source_bundle=bundle,
        )

    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        output_root = self.output_root.parent / f"provider-export-{request.request_id}"
        output_root.mkdir(parents=True, exist_ok=False)
        output = output_root / "fixture-exported.usda"
        content = b'#usda 1.0\ndef Xform "Exported" {}\n'
        output.write_bytes(content)
        digest = hashlib.sha256(content).hexdigest()
        request_digest = geometry_authoring_request_digest(request)
        provider = self.capabilities().provider
        bundle = _canonical_bundle(
            GeometrySourceBundle(
                bundle_id="c" * 64,
                producer=provider,
                source_revision=request.source_bundle.source_revision,
                coordinate_system=request.source_bundle.coordinate_system,
                representations=(
                    GeometryRepresentationBinding(
                        representation_id="render",
                        role="render_geometry",
                        format="usda",
                        media_type="model/vnd.usda",
                        artifact=GeometryArtifactBinding(
                            path=str(output),
                            sha256=digest,
                            size_bytes=len(content),
                        ),
                    ),
                ),
                parts=request.source_bundle.parts,
                parameters=request.source_bundle.parameters,
                provenance=GeometrySourceProvenance(
                    request_digest=request_digest,
                    parent_bundle_id=request.source_bundle.bundle_id,
                ),
                rights=request.source_bundle.rights,
            )
        )
        return GeometryAuthoringProviderReceipt(
            provider=provider,
            operation="export",
            request_digest=request_digest,
            disposition="succeeded",
            source_bundle=bundle,
        )


class BroadIdConnector(FakeConnector):
    @property
    def provider_id(self) -> str:
        return "Provider.v1:CAD"


class InvalidReceiptConnector(FakeConnector):
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root

    def generate(
        self,
        request: GeometryAuthoringRequest,
    ) -> GeometryAuthoringProviderReceipt:
        output_root = self.artifact_root / (
            f"{request.request_id}-{'a' * 12}-{'b' * 32}"
        )
        receipt = FakeConnector(output_root).generate(request)
        return receipt.model_copy(update={"request_digest": "0" * 64})


class ForeignConnector(FakeConnector):
    @property
    def provider_id(self) -> str:
        return "foreign-provider"


class ParameterDefinitionDriftConnector(FakeConnector):
    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        receipt = super().revise(request)
        assert receipt.source_bundle is not None
        parameter = receipt.source_bundle.parameters[0].model_copy(
            update={"minimum": 10.0}
        )
        bundle = _canonical_bundle(
            receipt.source_bundle.model_copy(update={"parameters": (parameter,)})
        )
        return receipt.model_copy(update={"source_bundle": bundle})


class SameRevisionConnector(FakeConnector):
    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        receipt = super().revise(request)
        assert receipt.source_bundle is not None
        bundle = _canonical_bundle(
            receipt.source_bundle.model_copy(
                update={"source_revision": request.source_bundle.source_revision}
            )
        )
        return receipt.model_copy(update={"source_bundle": bundle})


class UndeclaredParameterRevisionConnector(FakeConnector):
    def revise(
        self,
        request: GeometryAuthoringRevisionRequest,
    ) -> GeometryAuthoringProviderReceipt:
        declared_parameters = tuple(
            GeometrySemanticParameter(
                name=item.name,
                value=item.value,
                unit=item.unit,
            )
            for item in request.parameter_overrides
        )
        declared_source = _canonical_bundle(
            request.source_bundle.model_copy(update={"parameters": declared_parameters})
        )
        delegated_request = request.model_copy(
            update={"source_bundle": declared_source}
        )
        receipt = super().revise(delegated_request)
        assert receipt.source_bundle is not None
        request_digest = geometry_authoring_request_digest(request)
        bundle = _canonical_bundle(
            receipt.source_bundle.model_copy(
                update={
                    "parameters": declared_parameters,
                    "provenance": GeometrySourceProvenance(
                        request_digest=request_digest,
                        parent_bundle_id=request.source_bundle.bundle_id,
                    ),
                }
            )
        )
        return receipt.model_copy(
            update={"request_digest": request_digest, "source_bundle": bundle}
        )


class ExportParameterDefinitionDriftConnector(FakeConnector):
    def export(
        self,
        request: GeometryAuthoringExportRequest,
    ) -> GeometryAuthoringProviderReceipt:
        receipt = super().export(request)
        assert receipt.source_bundle is not None
        parameter = receipt.source_bundle.parameters[0].model_copy(
            update={"minimum": 10.0}
        )
        bundle = _canonical_bundle(
            receipt.source_bundle.model_copy(update={"parameters": (parameter,)})
        )
        return receipt.model_copy(update={"source_bundle": bundle})


@pytest.fixture(autouse=True)
def configured_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(settings, "host", "127.0.0.1")
    monkeypatch.setattr(settings, "api_key", "test-geometry-key")
    monkeypatch.setattr(settings, "workspace_root", str(tmp_path / "workspace"))
    monkeypatch.setattr(settings, "default_runtime_engine", "none")
    monkeypatch.setattr(settings, "render_backend", "ovrtx")
    monkeypatch.setattr(settings, "render_remote_base_url", None)
    monkeypatch.setattr(settings, "render_remote_api_key", None)
    monkeypatch.setattr(
        settings,
        "render_remote_allow_unauthenticated_identity",
        False,
    )
    monkeypatch.setattr(settings, "delegated_authoring_provider_id", None)
    monkeypatch.setattr(
        settings,
        "delegated_authoring_provider_label",
        "Delegated authoring worker",
    )
    monkeypatch.setattr(settings, "delegated_authoring_endpoint_url", None)
    monkeypatch.setattr(settings, "delegated_authoring_bearer_token", None)
    monkeypatch.setattr(settings, "delegated_authoring_rights_assertion", None)
    monkeypatch.setattr(
        settings,
        "delegated_authoring_supported_formats",
        ("step", "stl", "usd", "usda"),
    )
    monkeypatch.setattr(settings, "delegated_authoring_supports_image", True)
    monkeypatch.setattr(settings, "delegated_authoring_supports_revision", True)
    monkeypatch.setattr(settings, "delegated_authoring_supports_export", True)
    monkeypatch.setattr(settings, "delegated_authoring_returns_native_source", True)
    monkeypatch.setattr(settings, "build123d_endpoint_url", None)
    monkeypatch.setattr(settings, "build123d_bearer_token", None)
    monkeypatch.setattr(settings, "build123d_rights_assertion", None)
    monkeypatch.setattr(
        settings,
        "build123d_supported_formats",
        ("step",),
    )
    monkeypatch.setattr(settings, "build123d_supports_text", True)
    monkeypatch.setattr(settings, "build123d_supports_image", False)
    monkeypatch.setattr(settings, "build123d_supports_revision", False)
    monkeypatch.setattr(settings, "build123d_supports_export", False)
    monkeypatch.setattr(settings, "build123d_supports_semantic_parameters", False)
    monkeypatch.setattr(settings, "build123d_supports_parameter_definitions", False)
    monkeypatch.setattr(settings, "build123d_supports_semantic_parts", False)
    monkeypatch.setattr(settings, "build123d_supports_provider_assertions", False)
    monkeypatch.setattr(settings, "build123d_max_family_variants", 0)
    monkeypatch.setattr(settings, "forgecad_authoring_endpoint_url", None)
    monkeypatch.setattr(settings, "forgecad_authoring_bearer_token", None)
    monkeypatch.setattr(settings, "forgecad_authoring_rights_assertion", None)
    monkeypatch.setattr(settings, "forgecad_automated_use_authorized", False)
    monkeypatch.setattr(settings, "forgecad_authoring_supported_formats", ("step",))
    monkeypatch.setattr(settings, "forgecad_authoring_supports_text", True)
    monkeypatch.setattr(settings, "forgecad_authoring_supports_image", False)
    monkeypatch.setattr(settings, "forgecad_authoring_supports_revision", False)
    monkeypatch.setattr(settings, "forgecad_authoring_supports_export", False)
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_semantic_parameters", False
    )
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_parameter_definitions", False
    )
    monkeypatch.setattr(settings, "forgecad_authoring_supports_semantic_parts", False)
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_provider_assertions", False
    )
    monkeypatch.setattr(settings, "forgecad_authoring_max_family_variants", 0)
    monkeypatch.setattr(settings, "forgecad_authoring_returns_native_source", False)
    service_main.authoring_providers.clear()
    service_main.reset_runtime_state()
    yield
    service_main.authoring_providers.clear()
    service_main.reset_runtime_state()


def test_configured_delegated_workers_are_separate_explicit_providers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "forgecad_authoring_endpoint_url",
        "https://workers.example/forgecad",
    )
    monkeypatch.setattr(
        settings,
        "forgecad_authoring_rights_assertion",
        "Operator has authorized automated ForgeCAD use.",
    )
    monkeypatch.setattr(settings, "forgecad_automated_use_authorized", True)
    monkeypatch.setattr(settings, "forgecad_authoring_supports_revision", True)
    monkeypatch.setattr(settings, "forgecad_authoring_supports_export", True)
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_semantic_parameters", True
    )
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_parameter_definitions", True
    )
    monkeypatch.setattr(settings, "forgecad_authoring_supports_semantic_parts", True)
    monkeypatch.setattr(
        settings, "forgecad_authoring_supports_provider_assertions", True
    )
    monkeypatch.setattr(settings, "forgecad_authoring_max_family_variants", 16)
    monkeypatch.setattr(settings, "forgecad_authoring_returns_native_source", True)
    storage = service_main.WorkspaceStorage(
        Settings(workspace_root=str(tmp_path / "delegated-workspace"))
    )
    storage.initialize()

    service_main._register_configured_providers(storage)

    providers = {
        item.provider_id: item for item in service_main.authoring_providers.list()
    }
    assert set(providers) == {"forgecad-http"}
    assert providers["forgecad-http"].capabilities["returns_native_source"] is True
    forgecad_capabilities = providers["forgecad-http"].capabilities
    assert "parameter_definitions" in forgecad_capabilities["features"]
    assert "parameter_families" in forgecad_capabilities["features"]
    assert "semantic_parts" in forgecad_capabilities["features"]
    assert "provider_assertions" in forgecad_capabilities["features"]
    assert forgecad_capabilities["max_family_variants"] == 16
    assert "export" in forgecad_capabilities["operations"]


def test_generic_delegated_worker_registers_without_private_imports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "delegated_authoring_provider_id",
        "private-authoring-worker",
    )
    monkeypatch.setattr(
        settings,
        "delegated_authoring_provider_label",
        "Private authoring worker",
    )
    monkeypatch.setattr(
        settings,
        "delegated_authoring_endpoint_url",
        "https://workers.example/private-authoring",
    )
    monkeypatch.setattr(
        settings,
        "delegated_authoring_bearer_token",
        SecretStr("private-worker-token"),
    )
    monkeypatch.setattr(
        settings,
        "delegated_authoring_rights_assertion",
        "Operator authorizes private worker output.",
    )
    storage = service_main.WorkspaceStorage(
        Settings(workspace_root=str(tmp_path / "generic-delegated-workspace"))
    )
    storage.initialize()

    service_main._register_configured_providers(storage)

    providers = service_main.authoring_providers.list()
    assert [item.provider_id for item in providers] == ["private-authoring-worker"]
    assert "image" in providers[0].capabilities["input_modalities"]
    assert "revise" in providers[0].capabilities["operations"]
    assert providers[0].capabilities["returns_native_source"] is True


def test_configured_build123d_capabilities_match_operator_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        settings,
        "build123d_endpoint_url",
        "http://127.0.0.1:8765/v1/author",
    )
    monkeypatch.setattr(
        settings,
        "build123d_rights_assertion",
        "Authorized external Build123d worker output.",
    )
    monkeypatch.setattr(settings, "build123d_supported_formats", ("step", "stl"))
    monkeypatch.setattr(settings, "build123d_supports_text", True)
    monkeypatch.setattr(settings, "build123d_supports_image", False)
    monkeypatch.setattr(settings, "build123d_supports_revision", False)
    monkeypatch.setattr(settings, "build123d_supports_export", False)
    monkeypatch.setattr(settings, "build123d_supports_semantic_parameters", False)
    monkeypatch.setattr(settings, "build123d_supports_parameter_definitions", False)
    monkeypatch.setattr(settings, "build123d_supports_semantic_parts", False)
    monkeypatch.setattr(settings, "build123d_supports_provider_assertions", False)
    monkeypatch.setattr(settings, "build123d_max_family_variants", 0)
    storage = service_main.WorkspaceStorage(
        Settings(workspace_root=str(tmp_path / "build123d-workspace"))
    )
    storage.initialize()

    service_main._register_configured_providers(storage)

    providers = service_main.authoring_providers.list()
    assert [item.provider_id for item in providers] == ["build123d-http"]
    capabilities = providers[0].capabilities
    assert capabilities["output_formats"] == ["step", "stl"]
    assert capabilities["input_modalities"] == ["text"]
    assert capabilities["operations"] == ["generate"]
    assert capabilities["supports_semantic_parameters"] is False
    assert capabilities["features"] == ["native_source", "multi_format_output"]
    assert capabilities["max_family_variants"] == 0


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"X-Geometry-Agent-Service-Key": "test-geometry-key"}


def _upload_geometry(
    client: TestClient,
    auth_headers: dict[str, str],
    *,
    filename: str = "asset.usda",
    payload: bytes = b"#usda 1.0\n",
    extra_headers: dict[str, str] | None = None,
) -> Any:
    headers = {**auth_headers, **(extra_headers or {})}
    return client.post(
        "/api/geometry/sources",
        headers=headers,
        files={"file": (filename, payload, "application/octet-stream")},
        data={"role": "geometry_source"},
    )


def test_health_and_info_do_not_require_authentication(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json() == {"ok": True, "service": "geometry-agent-service"}

    info = client.get("/api/info")
    assert info.status_code == 200
    assert "/api/geometry/sources" in info.json()["routes"]


def test_geometry_routes_require_authentication(client: TestClient) -> None:
    response = client.get("/api/geometry/providers")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "authentication_required"

    response = client.get(
        "/api/geometry/providers",
        headers={"Authorization": "Bearer test-geometry-key"},
    )
    assert response.status_code == 200
    assert response.json() == {"providers": []}


def test_run_defaults_to_geometry_checks_without_a_simulation_runtime(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded = _upload_geometry(client, auth_headers)
    assert uploaded.status_code == 201
    observed: dict[str, Any] = {}

    def fake_run(params: Any) -> dict[str, Any]:
        observed["run_runtime_validation"] = params.run_runtime_validation
        observed["runtime_engine"] = params.runtime_engine
        return {"success": False, "error": "captured runtime defaults"}

    monkeypatch.setattr(service_main, "run_geometry_workflow", fake_run)
    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={"source_id": uploaded.json()["source_id"]},
    )

    assert response.status_code == 202
    assert observed == {
        "run_runtime_validation": False,
        "runtime_engine": "none",
    }


def test_runtime_validation_requires_an_available_engine(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={
            "source_id": "src_" + "0" * 64,
            "run_runtime_validation": True,
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == (
        "runtime_validation_engine_unavailable"
    )


def test_startup_guard_requires_auth_for_every_bind_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "api_key", None)
    with pytest.raises(RuntimeError, match="requires GEOMETRY_AGENT_SERVICE_API_KEY"):
        service_main._validate_service_security("127.0.0.1")
    with pytest.raises(RuntimeError, match="requires GEOMETRY_AGENT_SERVICE_API_KEY"):
        service_main._validate_service_security("0.0.0.0")


def test_upload_is_content_addressed_and_verifies_digest(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    payload = b'#usda 1.0\ndef Xform "Asset" {}\n'
    digest = hashlib.sha256(payload).hexdigest()
    response = _upload_geometry(
        client,
        auth_headers,
        payload=payload,
        extra_headers={"X-Content-SHA256": digest},
    )
    assert response.status_code == 201
    source = response.json()
    assert source["source_id"].startswith("src_")
    assert source["artifact"]["artifact_id"].startswith(f"sha256:{digest}:")
    assert source["artifact"]["sha256"] == digest
    assert "path" not in source["artifact"]

    mismatch = _upload_geometry(
        client,
        auth_headers,
        payload=payload,
        extra_headers={"X-Content-SHA256": "0" * 64},
    )
    assert mismatch.status_code == 422
    assert mismatch.json()["detail"]["code"] == "digest_mismatch"


def test_identical_artifact_bytes_retain_per_reference_metadata(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    payload = b'#usda 1.0\ndef Xform "Asset" {}\n'
    first = _upload_geometry(
        client,
        auth_headers,
        filename="first.usda",
        payload=payload,
    ).json()["artifact"]
    second = _upload_geometry(
        client,
        auth_headers,
        filename="second.usda",
        payload=payload,
    ).json()["artifact"]

    assert first["sha256"] == second["sha256"]
    assert first["artifact_id"] != second["artifact_id"]
    for record in (first, second):
        response = client.get(
            f"/api/geometry/artifacts/{record['artifact_id']}",
            headers=auth_headers,
        )
        assert response.status_code == 200
        assert record["filename"] in response.headers["content-disposition"]
        assert response.content == payload


def test_upload_rejects_filename_and_archive_traversal(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    filename_escape = _upload_geometry(
        client,
        auth_headers,
        filename="../../escape.usda",
    )
    assert filename_escape.status_code == 422
    assert filename_escape.json()["detail"]["code"] == "unsafe_filename"

    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("../escape.usda", "#usda 1.0\n")
    archive_escape = _upload_geometry(
        client,
        auth_headers,
        filename="bundle.zip",
        payload=archive_bytes.getvalue(),
    )
    assert archive_escape.status_code == 422
    assert archive_escape.json()["detail"]["code"] == "archive_traversal_forbidden"


@pytest.mark.parametrize(
    ("filename", "entrypoint"),
    (("asset.usdz", "scene.usda"), ("asset.3mf", "geometry/mesh.stl")),
)
def test_zip_based_source_containers_honor_explicit_geometry_entrypoints(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
    filename: str,
    entrypoint: str,
) -> None:
    geometry = (
        b'#usda 1.0\ndef Xform "Asset" {}\n'
        if entrypoint.endswith(".usda")
        else b"explicit geometry entrypoint"
    )
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(entrypoint, geometry)
        archive.writestr("metadata/manifest.json", "{}")

    response = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={
            "file": (filename, archive_bytes.getvalue(), "application/octet-stream")
        },
        data={"role": "geometry_source", "archive_entrypoint": entrypoint},
    )

    assert response.status_code == 201
    record = response.json()
    assert record["archive_entrypoint"] == entrypoint
    storage, _ = service_main._runtime()
    destination = (
        storage.executions_dir
        / f"archive-entrypoint-{tmp_path.name}-{Path(filename).suffix[1:]}"
    )
    materialized = storage.materialize_source(
        storage.get_source(record["source_id"]),
        destination,
    )
    assert materialized.relative_to(destination).as_posix() == entrypoint
    assert materialized.read_bytes() == geometry


@pytest.mark.parametrize(
    ("filename", "payload"),
    (
        (
            "remote.gltf",
            b'{"asset":{"version":"2.0"},"buffers":'
            b'[{"uri":"https://assets.example/model.bin"}]}',
        ),
        (
            "traversal.gltf",
            b'{"asset":{"version":"2.0"},"buffers":[{"uri":"../../outside.bin"}]}',
        ),
        (
            "external.usda",
            b'#usda 1.0\ndef Xform "Asset" (references = @../../outside.usda@) {}\n',
        ),
        (
            "external.obj",
            b"mtllib ../../outside.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        ),
        (
            "command.obj",
            b"call outside.script\nv 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n",
        ),
    ),
)
def test_upload_rejects_unbound_or_executable_geometry_dependencies(
    client: TestClient,
    auth_headers: dict[str, str],
    filename: str,
    payload: bytes,
) -> None:
    response = _upload_geometry(
        client,
        auth_headers,
        filename=filename,
        payload=payload,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "unbound_source_dependency"
    assert "outside" not in response.text
    assert "assets.example" not in response.text


def test_archive_upload_accepts_a_complete_bound_gltf_dependency_closure(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr(
            "scene.gltf",
            json.dumps(
                {
                    "asset": {"version": "2.0"},
                    "buffers": [{"uri": "buffer.bin", "byteLength": 4}],
                }
            ),
        )
        archive.writestr("buffer.bin", b"data")

    response = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={"file": ("scene.zip", archive_bytes.getvalue(), "application/zip")},
        data={"role": "geometry_source", "archive_entrypoint": "scene.gltf"},
    )

    assert response.status_code == 201
    assert response.json()["archive_entrypoint"] == "scene.gltf"


def test_upload_accepts_a_standard_3mf_relationship_manifest(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(
            "_rels/.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rel0" Target="/3D/3dmodel.model" Type="model"/>'
            "</Relationships>",
        )
        archive.writestr(
            "3D/3dmodel.model",
            '<model xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02" '
            'unit="millimeter"><resources/><build/></model>',
        )
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>',
        )

    response = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={
            "file": ("scene.3mf", archive_bytes.getvalue(), "model/3mf"),
        },
        data={"role": "geometry_source"},
    )

    assert response.status_code == 201
    assert response.json()["archive_entrypoint"] is None


def test_archive_run_binds_the_extracted_entrypoint_digest(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = b'#usda 1.0\ndef Xform "Asset" {}\n'
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("scene.usda", geometry)
    uploaded = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={"file": ("scene.zip", archive_bytes.getvalue(), "application/zip")},
        data={"role": "geometry_source", "archive_entrypoint": "scene.usda"},
    )
    assert uploaded.status_code == 201
    observed: dict[str, str] = {}

    def fake_run(params: Any) -> dict[str, Any]:
        observed["expected"] = params.expected_source_sha256
        observed["actual"] = hashlib.sha256(params.source_path.read_bytes()).hexdigest()
        return {"success": False, "error": "captured archive identity"}

    monkeypatch.setattr(service_main, "run_geometry_workflow", fake_run)
    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={"source_id": uploaded.json()["source_id"]},
    )

    assert response.status_code == 202
    assert observed["expected"] == hashlib.sha256(geometry).hexdigest()
    assert observed["actual"] == observed["expected"]
    assert observed["expected"] != hashlib.sha256(archive_bytes.getvalue()).hexdigest()


def test_archive_materialization_reverifies_the_registered_blob(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    archive_bytes = io.BytesIO()
    with zipfile.ZipFile(archive_bytes, "w") as archive:
        archive.writestr("scene.usda", b'#usda 1.0\ndef Xform "Asset" {}\n')
    uploaded = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={"file": ("scene.zip", archive_bytes.getvalue(), "application/zip")},
        data={"role": "geometry_source", "archive_entrypoint": "scene.usda"},
    )
    assert uploaded.status_code == 201
    storage, _ = service_main._runtime()
    source = storage.get_source(uploaded.json()["source_id"])
    _record, blob = storage.get_artifact(source.artifact.artifact_id)
    blob.chmod(0o600)
    blob.write_bytes(b"corrupt archive bytes")

    with pytest.raises(RuntimeError, match="Corrupt content-addressed artifact"):
        storage.materialize_source(
            source,
            storage.executions_dir / "corrupt-archive-materialization",
        )


@pytest.mark.parametrize("filename", ["author.py", "author.forge.js", "script.sh"])
def test_upload_rejects_executable_source_types(
    client: TestClient,
    auth_headers: dict[str, str],
    filename: str,
) -> None:
    response = _upload_geometry(
        client,
        auth_headers,
        filename=filename,
        payload=b"raise RuntimeError('must not execute')\n",
    )
    assert response.status_code == 415
    assert response.json()["detail"]["code"] == "source_type_forbidden"


def test_provider_unavailable_is_a_durable_typed_failure(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={"provider_id": "missing", "prompt": "Create a rigid bracket."},
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "authoring_provider_unavailable"

    fetched = client.get(
        f"/api/geometry/jobs/{job['job_id']}",
        headers=auth_headers,
    )
    assert fetched.status_code == 200
    assert fetched.json() == job


def test_provider_registry_offloads_synchronous_provider_calls(tmp_path: Path) -> None:
    class ThreadRecordingConnector(FakeConnector):
        provider_thread_id: int | None = None

        def generate(
            self,
            request: GeometryAuthoringRequest,
        ) -> GeometryAuthoringProviderReceipt:
            self.provider_thread_id = threading.get_ident()
            return super().generate(request)

    connector = ThreadRecordingConnector(tmp_path / "threaded-provider")
    registry = service_main.ProviderRegistry()
    registry.register(connector)
    request = GeometryAuthoringRequest(
        request_id="thread-offload",
        prompt="Create a rigid bracket.",
        target_profile="rigid_pick_place",
        requested_formats=("usda",),
    )
    caller_thread_id = threading.get_ident()

    receipt = asyncio.run(registry.generate(connector.provider_id, request))

    assert receipt.disposition == "succeeded"
    assert connector.provider_thread_id is not None
    assert connector.provider_thread_id != caller_thread_id


def test_provider_registry_awaits_native_async_provider(tmp_path: Path) -> None:
    class AsyncConnector(FakeConnector):
        provider_thread_id: int | None = None

        async def generate(
            self,
            request: GeometryAuthoringRequest,
        ) -> GeometryAuthoringProviderReceipt:
            self.provider_thread_id = threading.get_ident()
            await asyncio.sleep(0)
            return super().generate(request)

    connector = AsyncConnector(tmp_path / "async-provider")
    registry = service_main.ProviderRegistry()
    registry.register(connector)
    request = GeometryAuthoringRequest(
        request_id="async-provider",
        prompt="Create a rigid bracket.",
        target_profile="rigid_pick_place",
        requested_formats=("usda",),
    )
    caller_thread_id = threading.get_ident()

    receipt = asyncio.run(registry.generate(connector.provider_id, request))

    assert receipt.disposition == "succeeded"
    assert connector.provider_thread_id == caller_thread_id


def test_provider_registry_requires_a_deliverable_requested_format(
    tmp_path: Path,
) -> None:
    class NativeOnlyConnector(FakeConnector):
        def generate(
            self,
            request: GeometryAuthoringRequest,
        ) -> GeometryAuthoringProviderReceipt:
            receipt = super().generate(request)
            assert receipt.source_bundle is not None
            source = receipt.source_bundle
            representations = tuple(
                item.model_copy(update={"role": "native_source"})
                for item in source.representations
            )
            source = _canonical_bundle(
                source.model_copy(update={"representations": representations})
            )
            return receipt.model_copy(update={"source_bundle": source})

    connector = NativeOnlyConnector(tmp_path / "native-only-provider")
    registry = service_main.ProviderRegistry()
    registry.register(connector)
    request = GeometryAuthoringRequest(
        request_id="native-only-provider",
        prompt="Create a rigid bracket.",
        target_profile="rigid_pick_place",
        requested_formats=("usda",),
    )

    with pytest.raises(
        service_main.GeometryAuthoringInvalidResponse,
        match="requested exports",
    ):
        asyncio.run(registry.generate(connector.provider_id, request))


def test_registered_typed_provider_publishes_a_runnable_source(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    providers = client.get("/api/geometry/providers", headers=auth_headers)
    assert providers.status_code == 200
    assert providers.json()["providers"][0]["provider_id"] == "fixture-provider"

    response = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "succeeded"
    source = job["result"]["source"]
    assert source["schema_version"] == "geometry.source.v1"
    assert source["source_id"].startswith("src_")
    assert source["representations"][0]["artifact"]["artifact_id"].startswith("sha256:")
    assert "path" not in response.text
    assert str(tmp_path) not in response.text


def test_registry_rejection_discards_service_owned_provider_staging(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    storage, _jobs = service_main._runtime()
    artifact_root = storage.root / "provider-artifacts" / "fixture-provider"
    service_main.authoring_providers.register(InvalidReceiptConnector(artifact_root))

    response = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "failed"
    assert job["error"]["code"] == "invalid_response"
    request_dir = artifact_root / f"{job['job_id']}-{'a' * 12}-{'b' * 32}"
    assert not request_dir.exists()
    assert request_dir.parent.is_dir()


def test_service_accepts_every_provider_id_allowed_by_the_shared_contract(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    provider = BroadIdConnector(tmp_path / "broad-id-provider")
    service_main.authoring_providers.register(provider)

    response = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": provider.provider_id,
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "succeeded"


def test_provider_revision_requires_and_preserves_an_immutable_source_bundle(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()
    source_id = generated["result"]["source"]["source_id"]
    generated_bundle_id = generated["result"]["source"]["bundle_id"]

    response = client.post(
        "/api/geometry/revisions",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": source_id,
            "instructions": "Make the bracket wider.",
            "requested_formats": ["usda"],
        },
    )

    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "revision"
    assert job["status"] == "succeeded"
    assert job["source_id"] == source_id
    revised = job["result"]["source"]
    assert revised["bundle_id"] != generated_bundle_id
    assert len(revised["bundle_id"]) == 64
    assert revised["source_revision"] == "fixture-revision-2"
    assert "path" not in response.text
    assert str(tmp_path) not in response.text

    plain_source = _upload_geometry(client, auth_headers).json()["source_id"]
    rejected = client.post(
        "/api/geometry/revisions",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": plain_source,
            "instructions": "Make it wider.",
            "requested_formats": ["usda"],
        },
    ).json()
    assert rejected["status"] == "failed"
    assert rejected["error"]["code"] == "source_bundle_required"


def test_image_conditioned_source_can_be_revised_after_persistence(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    class ImageProvenanceConnector(FakeConnector):
        def capabilities(self) -> GeometryAuthoringCapabilityReport:
            return (
                super()
                .capabilities()
                .model_copy(
                    update={
                        "input_modalities": ("text", "image"),
                        "max_reference_artifacts": 8,
                    }
                )
            )

        def generate(
            self,
            request: GeometryAuthoringRequest,
        ) -> GeometryAuthoringProviderReceipt:
            receipt = super().generate(request)
            assert receipt.source_bundle is not None
            provenance = receipt.source_bundle.provenance.model_copy(
                update={
                    "input_artifacts": tuple(
                        reference.artifact for reference in request.references
                    )
                }
            )
            source_bundle = _canonical_bundle(
                receipt.source_bundle.model_copy(update={"provenance": provenance})
            )
            return receipt.model_copy(update={"source_bundle": source_bundle})

        def revise(
            self,
            request: GeometryAuthoringRevisionRequest,
        ) -> GeometryAuthoringProviderReceipt:
            assert len(request.source_bundle.provenance.input_artifacts) == 1
            persisted_input = request.source_bundle.provenance.input_artifacts[0]
            path = Path(persisted_input.path)
            assert path.is_file()
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == persisted_input.sha256
            )
            return super().revise(request)

    service_main.authoring_providers.register(
        ImageProvenanceConnector(tmp_path / "image-provider")
    )
    uploaded = client.post(
        "/api/geometry/sources",
        headers=auth_headers,
        files={
            "file": (
                "reference.png",
                b"\x89PNG\r\n\x1a\nbounded-test-reference",
                "image/png",
            )
        },
        data={"role": "reference_image"},
    )
    assert uploaded.status_code == 201

    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create geometry using the reference image.",
            "image_source_ids": [uploaded.json()["source_id"]],
            "requested_formats": ["usda"],
        },
    ).json()
    assert generated["status"] == "succeeded"

    revised = client.post(
        "/api/geometry/revisions",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "instructions": "Make one bounded revision.",
            "requested_formats": ["usda"],
        },
    ).json()
    assert revised["status"] == "succeeded"


def test_provider_registry_allows_first_parameters_on_unparameterized_revision(
    tmp_path: Path,
) -> None:
    connector = UndeclaredParameterRevisionConnector(tmp_path / "provider")
    registry = service_main.ProviderRegistry()
    registry.register(connector)
    generated = connector.generate(
        GeometryAuthoringRequest(
            request_id="undeclared-parameter-source",
            prompt="Create a rigid bracket.",
            target_profile="rigid_pick_place",
            requested_formats=("usda",),
        )
    )
    assert generated.source_bundle is not None
    source_bundle = _canonical_bundle(
        generated.source_bundle.model_copy(update={"parameters": ()})
    )
    request = GeometryAuthoringRevisionRequest(
        request_id="undeclared-parameter-revision",
        source_bundle=source_bundle,
        instructions="Make the bracket wider.",
        parameter_overrides=(
            GeometryAuthoringParameterValue(
                name="width_mm",
                value=80.0,
                unit="mm",
            ),
        ),
        requested_formats=("usda",),
    )

    receipt = asyncio.run(registry.revise(connector.provider_id, request))

    assert receipt.source_bundle is not None
    assert receipt.source_bundle.parameters == (
        GeometrySemanticParameter(name="width_mm", value=80.0, unit="mm"),
    )


def test_provider_revision_rejects_a_source_owned_by_another_provider(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    service_main.authoring_providers.register(
        ForeignConnector(tmp_path / "foreign-provider")
    )
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()

    revised = client.post(
        "/api/geometry/revisions",
        headers=auth_headers,
        json={
            "provider_id": "foreign-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "instructions": "Make it wider.",
            "requested_formats": ["usda"],
        },
    ).json()

    assert revised["status"] == "failed"
    assert revised["error"]["code"] == "source_provider_mismatch"


def test_provider_revision_rejects_reused_immutable_revision_identifier(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(
        SameRevisionConnector(tmp_path / "same-revision-provider")
    )
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()

    revised = client.post(
        "/api/geometry/revisions",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "instructions": "Make it wider.",
            "requested_formats": ["usda"],
        },
    ).json()

    assert revised["status"] == "failed"
    assert "new immutable revision" in revised["error"]["message"]


def test_provider_registry_accepts_usda_for_a_generic_usd_request(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))

    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a rigid bracket.",
            "requested_formats": ["usd"],
        },
    ).json()

    assert generated["status"] == "succeeded"
    assert generated["result"]["source"]["representations"][0]["format"] == "usda"


def test_parameter_family_materializes_validated_variants_in_parallel(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a parameterized rigid bracket.",
            "parameters": {"width_mm": {"value": 40.0, "unit": "mm"}},
            "requested_formats": ["usda"],
        },
    ).json()
    source_id = generated["result"]["source"]["source_id"]

    response = client.post(
        "/api/geometry/families",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": source_id,
            "requested_formats": ["usda"],
            "variants": [
                {
                    "variant_id": "compact",
                    "parameter_overrides": {"width_mm": {"value": 30.0, "unit": "mm"}},
                },
                {"variant_id": "standard", "parameter_overrides": {"width_mm": 50.0}},
                {"variant_id": "wide", "parameter_overrides": {"width_mm": 80.0}},
            ],
        },
    )

    assert response.status_code == 202
    job = response.json()
    assert job["kind"] == "family"
    assert job["status"] == "succeeded"
    assert job["result"]["succeeded_count"] == 3
    assert job["result"]["failed_count"] == 0
    observed = {
        item["variant_id"]: item["source"]["parameters"][0]["value"]
        for item in job["result"]["variants"]
    }
    assert observed == {"compact": 30.0, "standard": 50.0, "wide": 80.0}
    assert job["result"]["variants"][0]["parameter_units"] == {"width_mm": "mm"}
    assert "path" not in response.text
    assert str(tmp_path) not in response.text


def test_parameter_family_accepts_a_maximum_length_variant_identifier(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a parameterized rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()
    variant_id = "v" * 128

    job = client.post(
        "/api/geometry/families",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "requested_formats": ["usda"],
            "variants": [
                {"variant_id": variant_id, "parameter_overrides": {"width_mm": 50.0}}
            ],
        },
    ).json()

    assert job["status"] == "succeeded"
    assert job["result"]["variants"][0]["variant_id"] == variant_id


def test_parameter_family_rejects_provider_parameter_definition_drift(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(
        ParameterDefinitionDriftConnector(tmp_path / "drift-provider")
    )
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a parameterized rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()

    job = client.post(
        "/api/geometry/families",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "requested_formats": ["usda"],
            "variants": [
                {"variant_id": "wide", "parameter_overrides": {"width_mm": 80.0}}
            ],
        },
    ).json()

    assert job["status"] == "failed"
    assert job["error"]["code"] == "parameter_family_incomplete"
    variant_error = job["result"]["variants"][0]["error"]
    assert variant_error["code"] == "invalid_response"
    assert variant_error["retryable"] is False
    assert "declared semantic parameter family" in variant_error["message"]


def test_parameter_family_rejects_undeclared_or_out_of_range_values(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create a parameterized rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()
    source_id = generated["result"]["source"]["source_id"]

    for variant in (
        {"variant_id": "unknown", "parameter_overrides": {"height_mm": 20.0}},
        {"variant_id": "too-wide", "parameter_overrides": {"width_mm": 101.0}},
    ):
        job = client.post(
            "/api/geometry/families",
            headers=auth_headers,
            json={
                "provider_id": "fixture-provider",
                "source_id": source_id,
                "requested_formats": ["usda"],
                "variants": [variant],
            },
        ).json()
        assert job["status"] == "failed"
        assert job["error"]["code"] == "parameter_family_invalid"


def test_explicit_provider_export_preserves_source_lineage(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create an exportable rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()
    source = generated["result"]["source"]

    job = client.post(
        "/api/geometry/exports",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": source["source_id"],
            "requested_formats": ["usda"],
        },
    ).json()

    assert job["kind"] == "export"
    assert job["status"] == "succeeded"
    assert job["result"]["source"]["bundle_id"] != source["bundle_id"]
    assert job["result"]["source"]["parameters"][0]["value"] == 40.0


def test_immutable_provider_export_admits_an_external_revision(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(FakeConnector(tmp_path / "provider"))

    job = client.post(
        "/api/geometry/provider-exports",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_revision": "fixture-external-revision",
            "coordinate_system": {
                "meters_per_unit": 0.001,
                "up_axis": "Z",
                "forward_axis": "+Y",
                "handedness": "right",
            },
            "rights_assertion": "The caller is authorized to export this revision.",
            "requested_formats": ["usda"],
        },
    ).json()

    assert job["status"] == "succeeded"
    assert job["result"]["source"]["source_revision"] == ("fixture-external-revision")


def test_provider_publication_preserves_relative_bundle_layout(tmp_path: Path) -> None:
    provider_root = tmp_path / "provider-output" / "scenes"
    layer_root = provider_root / "layers"
    layer_root.mkdir(parents=True)
    root = provider_root / "root.usda"
    child = layer_root / "child.usda"
    root.write_text(
        "#usda 1.0\n(\n    subLayers = [@layers/child.usda@]\n)\n",
        encoding="utf-8",
    )
    child.write_text('#usda 1.0\ndef Xform "Child" {}\n', encoding="utf-8")
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    representations = tuple(
        GeometryRepresentationBinding(
            representation_id=representation_id,
            role=role,
            format="usda",
            media_type="model/vnd.usda",
            artifact=GeometryArtifactBinding(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            ),
        )
        for representation_id, role, path in (
            ("root", "render_geometry", root),
            ("child", "supporting_asset", child),
        )
    )
    provenance = GeometrySourceProvenance(request_digest="a" * 64)
    rights = GeometryRightsAssertion(assertion="Authorized fixture output.")
    coordinate_system = GeometryCoordinateSystem(meters_per_unit=1.0)
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=provider,
            source_revision="nested-revision",
            coordinate_system=coordinate_system,
            representations=representations,
            provenance=provenance,
            rights=rights,
        ),
        producer=provider,
        source_revision="nested-revision",
        coordinate_system=coordinate_system,
        representations=representations,
        provenance=provenance,
        rights=rights,
    )
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    execution = storage.execution_dir("job_" + "a" * 32)

    published = service_main._publish_provider_bundle(
        bundle,
        storage=storage,
        provider_output_dir=execution / "provider",
        provider_request_id="nested-bundle-request",
    )
    source = storage.get_source(published["source_id"])
    selected, manifest = storage.materialize_source_handoff(
        source,
        storage.executions_dir / "materialized-nested-bundle",
    )

    assert selected.name == "root.usda"
    assert (selected.parent / "layers" / "child.usda").is_file()
    assert "@layers/child.usda@" in selected.read_text(encoding="utf-8")
    assert manifest is not None
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert [
        item["artifact"]["path"] for item in manifest_payload["representations"]
    ] == [
        "root.usda",
        "layers/child.usda",
    ]
    assert root.is_file()
    assert child.is_file()


def test_provider_publication_rejects_an_unbound_source_dependency(
    tmp_path: Path,
) -> None:
    provider_root = tmp_path / "provider-output"
    provider_root.mkdir()
    root = provider_root / "root.usda"
    root.write_text(
        '#usda 1.0\ndef Xform "Asset" (references = @../../outside.usda@) {}\n',
        encoding="utf-8",
    )
    bundle = _single_artifact_bundle(root)
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    execution = storage.execution_dir("job_" + "b" * 32)

    with pytest.raises(
        ValueError,
        match="Provider geometry dependencies must be self-contained",
    ):
        service_main._publish_provider_bundle(
            bundle,
            storage=storage,
            provider_output_dir=execution / "provider",
            provider_request_id="unbound-provider-bundle",
        )


@pytest.mark.parametrize("root_references_native", (False, True))
def test_provider_native_source_is_opaque_and_not_a_dependency_target(
    tmp_path: Path,
    root_references_native: bool,
) -> None:
    provider_root = tmp_path / "provider-output"
    provider_root.mkdir()
    root = provider_root / "root.usda"
    root.write_text(
        "#usda 1.0\n"
        + ("(\n    subLayers = [@native.usda@]\n)\n" if root_references_native else "")
        + 'def Xform "Asset" {}\n',
        encoding="utf-8",
    )
    native = provider_root / "native.usda"
    native.write_text(
        '#usda 1.0\ndef Xform "Native" (references = '
        "@https://provider.example/private.usda@) {}\n",
        encoding="utf-8",
    )
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    representations = tuple(
        GeometryRepresentationBinding(
            representation_id=representation_id,
            role=role,
            format="usda",
            media_type="model/vnd.usda",
            artifact=GeometryArtifactBinding(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                size_bytes=path.stat().st_size,
            ),
        )
        for representation_id, role, path in (
            ("root", "render_geometry", root),
            ("native", "native_source", native),
        )
    )
    provenance = GeometrySourceProvenance(request_digest="a" * 64)
    rights = GeometryRightsAssertion(assertion="Authorized fixture output.")
    coordinate_system = GeometryCoordinateSystem(meters_per_unit=1.0)
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=provider,
            source_revision="opaque-native-revision",
            coordinate_system=coordinate_system,
            representations=representations,
            provenance=provenance,
            rights=rights,
        ),
        producer=provider,
        source_revision="opaque-native-revision",
        coordinate_system=coordinate_system,
        representations=representations,
        provenance=provenance,
        rights=rights,
    )
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    execution = storage.execution_dir("job_" + "e" * 32)

    if root_references_native:
        with pytest.raises(
            ValueError,
            match="Provider geometry dependencies must be self-contained",
        ):
            service_main._publish_provider_bundle(
                bundle,
                storage=storage,
                provider_output_dir=execution / "provider",
                provider_request_id="native-dependency-bundle",
            )
        return

    published = service_main._publish_provider_bundle(
        bundle,
        storage=storage,
        provider_output_dir=execution / "provider",
        provider_request_id="opaque-native-bundle",
    )
    source = storage.get_source(published["source_id"])
    assert len(source.source_bundle_artifacts) == 2
    assert published["selected_representation_id"] == "root"


def test_provider_provenance_inputs_are_persisted_and_rematerialized(
    tmp_path: Path,
) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    publication_execution = storage.execution_dir("job_" + "1" * 32)
    reference = publication_execution / "references" / "reference.png"
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b"bounded-reference-image")
    provider_root = tmp_path / "provider-output"
    provider_root.mkdir()
    geometry = provider_root / "geometry.usda"
    geometry.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    bundle = _single_artifact_bundle(geometry)
    bundle = _canonical_bundle(
        bundle.model_copy(
            update={
                "provenance": GeometrySourceProvenance(
                    request_digest=bundle.provenance.request_digest,
                    input_artifacts=(
                        GeometryArtifactBinding(
                            path=str(reference),
                            sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
                            size_bytes=reference.stat().st_size,
                        ),
                    ),
                )
            }
        )
    )

    published = service_main._publish_provider_bundle(
        bundle,
        storage=storage,
        provider_output_dir=publication_execution / "provider",
        provider_request_id="provenance-persistence-request",
    )
    source = storage.get_source(published["source_id"])
    assert len(source.source_bundle_artifacts) == 2
    manifest_record = source.source_bundle_manifest
    assert manifest_record is not None
    _record, manifest_blob = storage.get_artifact(manifest_record.artifact_id)
    persisted_manifest = GeometrySourceBundle.model_validate_json(
        manifest_blob.read_bytes()
    )
    persisted_input = persisted_manifest.provenance.input_artifacts[0]
    assert persisted_input.path.startswith("provenance/input-001-")
    assert not persisted_input.path.startswith("sha256:")

    materialization_execution = storage.execution_dir("job_" + "2" * 32)
    materialized = service_main._materialize_source_bundle(
        storage,
        source,
        materialization_execution / "prior-source",
    )
    materialized_input = materialized.provenance.input_artifacts[0]
    materialized_path = Path(materialized_input.path)
    assert materialized_path.is_relative_to(materialization_execution)
    assert materialized_path.read_bytes() == reference.read_bytes()
    assert (
        materialized_input.sha256 == hashlib.sha256(reference.read_bytes()).hexdigest()
    )
    validate_geometry_source_bundle_identity(materialized)


def test_provider_provenance_input_must_come_from_service_workspace(
    tmp_path: Path,
) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    execution = storage.execution_dir("job_" + "3" * 32)
    provider_root = tmp_path / "provider-output"
    provider_root.mkdir()
    geometry = provider_root / "geometry.usda"
    geometry.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")
    outside_input = tmp_path / "outside-reference.png"
    outside_input.write_bytes(b"must-not-be-copied")
    bundle = _single_artifact_bundle(geometry)
    bundle = _canonical_bundle(
        bundle.model_copy(
            update={
                "provenance": GeometrySourceProvenance(
                    request_digest=bundle.provenance.request_digest,
                    input_artifacts=(
                        GeometryArtifactBinding(
                            path=str(outside_input),
                            sha256=hashlib.sha256(
                                outside_input.read_bytes()
                            ).hexdigest(),
                            size_bytes=outside_input.stat().st_size,
                        ),
                    ),
                )
            }
        )
    )

    with pytest.raises(
        TypeError,
        match="provenance inputs must be service-materialized",
    ):
        service_main._publish_provider_bundle(
            bundle,
            storage=storage,
            provider_output_dir=execution / "provider",
            provider_request_id="outside-provenance-request",
        )


@pytest.mark.parametrize("valid_digest", (True, False))
@pytest.mark.parametrize(
    "provider_request_id",
    ("request-001", f"job_{'d' * 32}:variant-01-{'e' * 16}"),
)
def test_provider_publication_discards_service_owned_staging(
    tmp_path: Path,
    valid_digest: bool,
    provider_request_id: str,
) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    safe_request_id = provider_request_id.replace(":", "_")
    request_dir = (
        storage.root
        / "provider-artifacts"
        / "fixture-provider"
        / f"{safe_request_id}-{'b' * 12}-{'c' * 32}"
    )
    request_dir.mkdir(parents=True)
    source_path = request_dir / "geometry.usda"
    source_path.write_bytes(b"#usda 1.0\n")
    bundle = _single_artifact_bundle(
        source_path,
        sha256=None if valid_digest else "0" * 64,
    )
    execution = storage.execution_dir("job_" + "c" * 32)

    if valid_digest:
        published = service_main._publish_provider_bundle(
            bundle,
            storage=storage,
            provider_output_dir=execution / "provider",
            provider_request_id=provider_request_id,
        )
        assert (
            storage.get_source(published["source_id"]).source_id
            == published["source_id"]
        )
    else:
        with pytest.raises(TypeError, match="declared identity"):
            service_main._publish_provider_bundle(
                bundle,
                storage=storage,
                provider_output_dir=execution / "provider",
                provider_request_id=provider_request_id,
            )

    assert not request_dir.exists()
    assert request_dir.parent.is_dir()


def test_provider_cleanup_refuses_another_request_staging_directory(
    tmp_path: Path,
) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    request_dir = (
        storage.root
        / "provider-artifacts"
        / "fixture-provider"
        / f"other-request-{'b' * 12}-{'c' * 32}"
    )
    request_dir.mkdir(parents=True)
    source_path = request_dir / "geometry.usda"
    source_path.write_bytes(b"#usda 1.0\n")
    bundle = _single_artifact_bundle(source_path)

    discarded = service_main._discard_provider_staging_bundle(
        bundle,
        storage=storage,
        provider_request_id="current-request",
    )

    assert discarded is False
    assert source_path.is_file()


def test_provider_cleanup_preserves_a_directory_swapped_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    provider_root = storage.root / "provider-artifacts" / "fixture-provider"
    request_id = "current-request"
    request_dir = provider_root / f"{request_id}-{'b' * 12}-{'c' * 32}"
    request_dir.mkdir(parents=True)
    source_path = request_dir / "geometry.usda"
    source_path.write_bytes(b"#usda 1.0\n")
    bundle = _single_artifact_bundle(source_path)
    replacement = provider_root / "replacement"
    replacement.mkdir()
    (replacement / "other-request.usda").write_bytes(b"other-request")
    parked = provider_root / "parked"
    original_rename = service_main.os.rename
    raced = False

    def race_before_quarantine(
        source: str | Path,
        destination: str | Path,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal raced
        if not raced and source == request_dir.name and src_dir_fd is not None:
            raced = True
            original_rename(request_dir, parked)
            original_rename(replacement, request_dir)
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(service_main.os, "rename", race_before_quarantine)

    discarded = service_main._discard_provider_staging_bundle(
        bundle,
        storage=storage,
        provider_request_id=request_id,
    )

    assert discarded is False
    assert (parked / "geometry.usda").read_bytes() == b"#usda 1.0\n"
    preserved = list(provider_root.rglob("other-request.usda"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"other-request"


def test_provider_publication_prefers_exact_design_over_render_mesh(
    tmp_path: Path,
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider_root = tmp_path / "provider-output"
    provider_root.mkdir()
    step_path = provider_root / "assembly.step"
    part_path = provider_root / "part.step"
    stl_path = provider_root / "assembly.stl"
    step_path.write_bytes(b"exact design fixture")
    part_path.write_bytes(b"exact part fixture")
    stl_path.write_bytes(b"lossy render fixture")
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    representations = (
        GeometryRepresentationBinding(
            representation_id="render",
            role="render_geometry",
            format="stl",
            media_type="model/stl",
            artifact=GeometryArtifactBinding(
                path=str(stl_path),
                sha256=hashlib.sha256(stl_path.read_bytes()).hexdigest(),
                size_bytes=stl_path.stat().st_size,
            ),
        ),
        GeometryRepresentationBinding(
            representation_id="part-design",
            role="design_exchange",
            format="step",
            media_type="model/step",
            artifact=GeometryArtifactBinding(
                path=str(part_path),
                sha256=hashlib.sha256(part_path.read_bytes()).hexdigest(),
                size_bytes=part_path.stat().st_size,
            ),
        ),
        GeometryRepresentationBinding(
            representation_id="design",
            role="design_exchange",
            format="step",
            media_type="model/step",
            artifact=GeometryArtifactBinding(
                path=str(step_path),
                sha256=hashlib.sha256(step_path.read_bytes()).hexdigest(),
                size_bytes=step_path.stat().st_size,
            ),
        ),
    )
    parts = (
        GeometryPartBinding(
            part_id="assembly",
            name="Assembly",
            representation_ids=("design",),
        ),
        GeometryPartBinding(
            part_id="part",
            name="Part",
            parent_part_id="assembly",
            representation_ids=("part-design",),
        ),
    )
    provenance = GeometrySourceProvenance(request_digest="a" * 64)
    rights = GeometryRightsAssertion(assertion="Authorized fixture output.")
    coordinate_system = GeometryCoordinateSystem(meters_per_unit=0.001)
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=provider,
            source_revision="mixed-revision",
            coordinate_system=coordinate_system,
            representations=representations,
            parts=parts,
            provenance=provenance,
            rights=rights,
        ),
        producer=provider,
        source_revision="mixed-revision",
        coordinate_system=coordinate_system,
        representations=representations,
        parts=parts,
        provenance=provenance,
        rights=rights,
    )
    storage, _jobs = service_main._runtime()
    execution = storage.execution_dir("job_" + "b" * 32)

    published = service_main._publish_provider_bundle(
        bundle,
        storage=storage,
        provider_output_dir=execution / "provider",
        provider_request_id="mixed-bundle-request",
    )
    source = storage.get_source(published["source_id"])
    selected, _manifest = storage.materialize_source_handoff(
        source,
        storage.executions_dir / "materialized-mixed-bundle",
    )

    assert published["selected_representation_id"] == "design"
    assert selected.name == "assembly.step"
    assert selected.read_bytes() == step_path.read_bytes()

    observed: dict[str, Any] = {}

    def fake_run(params: Any) -> dict[str, Any]:
        observed["source_representation_id"] = params.source_representation_id
        return {"success": False, "error": "captured selected representation"}

    monkeypatch.setattr(service_main, "run_geometry_workflow", fake_run)
    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={
            "source_id": published["source_id"],
            "runtime_engine": "none",
            "run_runtime_validation": False,
        },
    )

    assert response.status_code == 202
    assert observed["source_representation_id"] == "design"


def test_provider_export_rejects_parameter_definition_drift(
    client: TestClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    service_main.authoring_providers.register(
        ExportParameterDefinitionDriftConnector(tmp_path / "provider")
    )
    generated = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "prompt": "Create an exportable rigid bracket.",
            "requested_formats": ["usda"],
        },
    ).json()

    exported = client.post(
        "/api/geometry/exports",
        headers=auth_headers,
        json={
            "provider_id": "fixture-provider",
            "source_id": generated["result"]["source"]["source_id"],
            "requested_formats": ["usda"],
        },
    ).json()

    assert exported["status"] == "failed"
    assert exported["error"]["code"] == "invalid_response"
    assert "immutable semantic parameters" in exported["error"]["message"]


def test_source_to_run_happy_path_publishes_artifact_ids_only(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(settings, "render_backend", "remote")
    monkeypatch.setattr(
        settings,
        "render_remote_base_url",
        "https://render.example/v1",
    )
    monkeypatch.setattr(
        settings,
        "render_remote_api_key",
        SecretStr("render-endpoint-test-key"),
    )
    monkeypatch.setattr(
        settings,
        "render_remote_allow_unauthenticated_identity",
        True,
    )
    uploaded = _upload_geometry(client, auth_headers)
    assert uploaded.status_code == 201
    source_id = uploaded.json()["source_id"]
    observed: dict[str, Any] = {}

    class FakeWorkflowResult:
        def model_dump(self, *, mode: str) -> dict[str, Any]:
            assert mode == "json"
            return observed["result"]

    def fake_run(params: Any) -> FakeWorkflowResult:
        observed["params"] = params
        assert params.source_path.is_file()
        assert params.source_path.read_bytes() == b"#usda 1.0\n"
        params.output_dir.mkdir(parents=True, exist_ok=True)
        output = params.output_dir / "asset.geometry.usda"
        evidence = params.output_dir / "geometry_validation_evidence.json"
        render = params.output_dir / "renders" / "front.png"
        render.parent.mkdir()
        output.write_text("#usda 1.0\n", encoding="utf-8")
        render.write_bytes(b"render-image")
        evidence.write_text(
            json.dumps(
                {
                    "status": "pass",
                    "checks": [
                        {
                            "evidence_artifacts": [
                                {"kind": "ovrtx_render_view", "path": str(render)}
                            ]
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        observed["result"] = {
            "success": True,
            "route": {"route": "existing_source"},
            "output_dir": str(params.output_dir),
            "source_usd_path": str(params.source_path),
            "optimized_usd_path": str(output),
            "validation_evidence_path": str(evidence),
            "validation_status": "pass",
            "preview_renders": [{"view": "front", "path": str(render)}],
            "error": None,
        }
        return FakeWorkflowResult()

    monkeypatch.setattr(service_main, "run_geometry_workflow", fake_run)
    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={
            "source_id": source_id,
            "runtime_engine": "none",
            "run_runtime_validation": False,
        },
    )
    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "succeeded"
    assert observed["params"].render_backend == "remote"
    assert observed["params"].render_remote_base_url == "https://render.example/v1"
    assert observed["params"].render_remote_api_key.get_secret_value() == (
        "render-endpoint-test-key"
    )
    assert observed["params"].render_remote_allow_unauthenticated_identity is True
    assert job["result"]["workflow"]["optimized_usd_path"].startswith("sha256:")
    assert job["result"]["workflow"]["validation_evidence_path"].startswith("sha256:")
    assert job["result"]["artifacts"]["optimized_usd_path"]["filename"] == (
        "asset.geometry.usda"
    )
    assert str(tmp_path) not in response.text

    evidence_id = job["result"]["workflow"]["validation_evidence_path"]
    evidence_response = client.get(
        f"/api/geometry/artifacts/{evidence_id}",
        headers=auth_headers,
    )
    assert evidence_response.status_code == 200
    published_evidence = evidence_response.json()
    render_id = published_evidence["checks"][0]["evidence_artifacts"][0]["path"]
    assert render_id.startswith("sha256:")
    assert job["result"]["workflow"]["preview_renders"][0]["path"] == render_id
    render_response = client.get(
        f"/api/geometry/artifacts/{render_id}",
        headers=auth_headers,
    )
    assert render_response.status_code == 200
    assert render_response.content == b"render-image"
    assert str(tmp_path) not in evidence_response.text


def test_run_rejects_parameter_overrides_before_workflow_execution(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uploaded = _upload_geometry(client, auth_headers)
    assert uploaded.status_code == 201
    monkeypatch.setattr(
        service_main,
        "run_geometry_workflow",
        lambda _params: pytest.fail("unapplied overrides must not reach the workflow"),
    )

    response = client.post(
        "/api/geometry/runs",
        headers=auth_headers,
        json={
            "source_id": uploaded.json()["source_id"],
            "param_overrides": {"width": 35.0},
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == (
        "run_parameter_overrides_require_revision"
    )
    assert "/api/geometry/revisions" in response.json()["detail"]["message"]


def test_workflow_publication_omits_nested_symlink_references(tmp_path: Path) -> None:
    storage = WorkspaceStorage(Settings(workspace_root=str(tmp_path / "workspace")))
    storage.initialize()
    execution = storage.execution_dir("job_" + "c" * 32)
    render = execution / "renders" / "front.png"
    render.parent.mkdir()
    render.write_bytes(b"render-image")
    linked_render = execution / "linked-front.png"
    linked_render.symlink_to(render)
    evidence = execution / "geometry_validation_evidence.json"
    evidence.write_text(
        json.dumps({"render_path": linked_render.name}) + "\n",
        encoding="utf-8",
    )

    published = service_main._publish_workflow_result(
        {"validation_evidence_path": str(evidence)},
        storage=storage,
        execution_dir=execution,
    )

    _, public_evidence_path = storage.get_artifact(
        published["workflow"]["validation_evidence_path"]
    )
    public_evidence = json.loads(public_evidence_path.read_text(encoding="utf-8"))
    assert public_evidence["render_path"] is None
    assert "validation_evidence_path/render_path" in published["omitted_path_fields"]


def test_request_body_limit_applies_before_json_parsing(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_request_bytes", 128)
    response = client.post(
        "/api/geometry/generations",
        headers=auth_headers,
        json={"provider_id": "missing", "prompt": "x" * 512},
    )
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "request_body_too_large"


def test_service_has_no_private_authoring_imports() -> None:
    def private_root(*parts: str) -> str:
        return "_".join(parts)

    forbidden_roots = {
        private_root("cad", "agent"),
        private_root("cad", "agent", "server"),
        private_root("cad", "dsl"),
        private_root("cad", "sdk"),
        private_root("geometry", "authoring", "internal"),
    }
    for path in (PROJECT_ROOT / "geometry_agent_service").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        assert not {module.split(".", 1)[0] for module in imported}.intersection(
            forbidden_roots
        ), path

    project_text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert not any(name in project_text for name in forbidden_roots)
