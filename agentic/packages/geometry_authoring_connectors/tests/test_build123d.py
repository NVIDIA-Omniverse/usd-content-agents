# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import base64
import hashlib
import json
from pathlib import Path

import pytest
from geometry_authoring_contracts import (
    GeometryAuthoringExportRequest,
    GeometryAuthoringProviderIdentity,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
)
from pydantic import ValidationError

from geometry_authoring_connectors import (
    ArtifactLimitError,
    AuthoringRequest,
    Build123dGeometryAuthoringProvider,
    Build123dHttpConnector,
    Build123dWorkerResult,
    Build123dWorkerRunner,
    ConnectorConfigurationError,
    InlineInputArtifact,
    InvalidProviderResponseError,
    ProviderTransportError,
    UnsafeArtifactError,
    UnsupportedCapabilityError,
    WorkerArtifact,
    WorkerIsolationError,
    WorkerIsolationKind,
)

from .conftest import FakeResponse, QueueTransport
from .trusted_fixture_backend import TrustedFixtureBackend


def _request() -> AuthoringRequest:
    return AuthoringRequest(
        request_id="build123d-test-1",
        prompt="Create a 10 by 20 by 30 millimeter box.",
        target_formats=("step",),
    )


def _worker_payload() -> dict[str, object]:
    runner = Build123dWorkerRunner(
        TrustedFixtureBackend(),
        allow_trusted_test_fixture=True,
    )
    return runner.handle(_request().model_dump(mode="json"))


def test_trusted_fixture_backend_requires_explicit_test_opt_in() -> None:
    with pytest.raises(WorkerIsolationError, match="container or sandboxed-process"):
        Build123dWorkerRunner(TrustedFixtureBackend())


def test_worker_and_http_connector_round_trip(tmp_path: Path) -> None:
    payload = _worker_payload()
    transport = QueueTransport(FakeResponse.json(payload))
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        bearer_token="secret-token",
        supported_formats=("step",),
        transport=transport,
    )

    bundle = connector.generate(_request(), output_dir=tmp_path / "result")

    assert bundle.source_revision == "trusted-fixture-revision-1"
    assert {item.filename for item in bundle.artifacts} == {
        "trusted_fixture.py",
        "trusted_fixture.step",
    }
    assert all(item.path.is_file() for item in bundle.artifacts)
    assert bundle.parts[0].part_id == "body"
    assert bundle.parameters[0].semantic_role == "overall_width"
    assert bundle.parameters[0].effects == ("exact_geometry",)
    assert bundle.verification_assertions[0].status == "passed"
    recorded = transport.requests[0]
    assert recorded.url == "https://geometry-worker.example/v1/author"
    assert "secret-token" not in recorded.url
    assert recorded.kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert recorded.kwargs["allow_redirects"] is False
    assert json.loads(recorded.kwargs["data"])["prompt"].startswith("Create")


def test_canonical_provider_returns_a_geometry_source_receipt(tmp_path: Path) -> None:
    transport = QueueTransport(FakeResponse.json(_worker_payload()))
    provider = Build123dGeometryAuthoringProvider(
        connector=Build123dHttpConnector(
            endpoint_url="https://geometry-worker.example/v1/author",
            supported_formats=("step",),
            supports_revision=True,
            supports_export=True,
            supports_semantic_parameters=True,
            supports_parameter_definitions=True,
            supports_semantic_parts=True,
            supports_provider_assertions=True,
            max_family_variants=64,
            transport=transport,
        ),
        artifact_root=tmp_path / "provider",
        rights_assertion="Authorized test output.",
    )
    request = GeometryAuthoringRequest(
        request_id="canonical-generate-1",
        prompt="Create a 10 by 20 by 30 millimeter box.",
        target_profile="rigid_pick_place",
        requested_formats=("step",),
    )

    receipt = provider.generate(request)

    capabilities = provider.capabilities()
    assert "parameter_families" in capabilities.features
    assert "native_source" in capabilities.features
    assert "export" in capabilities.operations
    assert capabilities.max_family_variants == 64
    assert receipt.disposition == "succeeded"
    assert receipt.operation == "generate"
    assert receipt.source_bundle is not None
    assert receipt.source_bundle.schema_version == "geometry.source.v1"
    assert {item.role for item in receipt.source_bundle.representations} == {
        "native_source",
        "design_exchange",
    }
    assert all(Path(item.artifact.path).is_file() for item in receipt.source_bundle.representations)

    foreign_bundle = receipt.source_bundle.model_copy(
        update={
            "producer": GeometryAuthoringProviderIdentity(
                provider_id="another-provider",
                provider_version="1",
            )
        }
    )
    rejected = provider.revise(
        GeometryAuthoringRevisionRequest(
            request_id="cross-provider-revision",
            source_bundle=foreign_bundle,
            instructions="Make it wider.",
            requested_formats=("step",),
        )
    )
    assert rejected.disposition == "failed"
    assert rejected.failure is not None
    assert rejected.failure.code == "unsupported_operation"
    assert len(transport.requests) == 1


def test_build123d_provider_exports_from_its_immutable_revision(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse.json(_worker_payload()),
        FakeResponse.json(_worker_payload()),
    )
    provider = Build123dGeometryAuthoringProvider(
        connector=Build123dHttpConnector(
            endpoint_url="https://geometry-worker.example/v1/author",
            supported_formats=("step",),
            supports_export=True,
            transport=transport,
        ),
        artifact_root=tmp_path / "provider",
        rights_assertion="Authorized test output.",
    )
    generated = provider.generate(
        GeometryAuthoringRequest(
            request_id="build123d-generate-for-export",
            prompt="Create a 10 by 20 by 30 millimeter box.",
            target_profile="rigid_pick_place",
            requested_formats=("step",),
        )
    )
    assert generated.source_bundle is not None

    exported = provider.export(
        GeometryAuthoringExportRequest(
            request_id="build123d-export",
            source_bundle=generated.source_bundle,
            requested_formats=("step",),
        )
    )

    assert exported.disposition == "succeeded"
    assert exported.source_bundle is not None
    assert exported.source_bundle.provenance.parent_bundle_id == generated.source_bundle.bundle_id
    low_request = json.loads(transport.requests[1].kwargs["data"])
    assert low_request["operation"] == "export"
    assert low_request["prior_source_revision"] == "trusted-fixture-revision-1"


@pytest.mark.parametrize(
    "url",
    (
        "http://remote.example/v1/author",
        "https://user:password@remote.example/v1/author",
        "https://remote.example/v1/author?token=secret",
        "https://remote.example/v1/author#fragment",
    ),
)
def test_worker_endpoint_must_be_credential_free_and_https(url: str) -> None:
    with pytest.raises(ConnectorConfigurationError):
        Build123dHttpConnector(endpoint_url=url)


def test_loopback_http_worker_is_allowed() -> None:
    Build123dHttpConnector(endpoint_url="http://127.0.0.1:8765/v1/author")


def test_worker_capabilities_are_conservative_and_operator_configurable() -> None:
    defaults = Build123dHttpConnector(endpoint_url="http://127.0.0.1:8765/v1/author").capabilities()
    assert defaults.formats == ("step",)
    assert defaults.text is True
    assert defaults.image is False
    assert defaults.revision is False
    assert defaults.export is False
    assert defaults.parameter_definitions is False
    assert defaults.semantic_parts is False
    assert defaults.provider_assertions is False
    assert defaults.max_family_variants == 0

    configured = Build123dHttpConnector(
        endpoint_url="http://127.0.0.1:8765/v1/author",
        supported_formats=("step", "stl"),
        supports_text=False,
        supports_image=True,
        supports_revision=False,
        supports_export=False,
        supports_semantic_parameters=False,
        supports_parameter_definitions=False,
        supports_semantic_parts=False,
        supports_provider_assertions=False,
        max_family_variants=0,
    ).capabilities()
    assert configured.formats == ("step", "stl")
    assert configured.text is False
    assert configured.image is True
    assert configured.revision is False
    assert configured.export is False
    assert configured.parameter_definitions is False
    assert configured.semantic_parts is False
    assert configured.provider_assertions is False
    assert configured.max_family_variants == 0


def test_worker_redirect_is_rejected_without_fallback(tmp_path: Path) -> None:
    transport = QueueTransport(
        FakeResponse(b"", status_code=307, headers={"Location": "https://evil.example"})
    )
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=transport,
    )
    with pytest.raises(ProviderTransportError, match="redirects are not permitted"):
        connector.generate(_request(), output_dir=tmp_path)


def test_worker_oversized_response_is_rejected_before_streaming(tmp_path: Path) -> None:
    response = FakeResponse.json(_worker_payload())
    response.headers["Content-Length"] = str(769 * 1024 * 1024)
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=QueueTransport(response),
    )
    with pytest.raises(ArtifactLimitError, match="exceeds the byte limit"):
        connector.generate(_request(), output_dir=tmp_path)


def test_worker_http_failure_is_one_shot_and_secret_free(tmp_path: Path) -> None:
    transport = QueueTransport(
        RuntimeError("Bearer should-not-surface"),
        FakeResponse.json(_worker_payload()),
    )
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        bearer_token="should-not-surface",
        supported_formats=("step",),
        transport=transport,
    )
    with pytest.raises(ProviderTransportError) as captured:
        connector.generate(_request(), output_dir=tmp_path)
    assert len(transport.requests) == 1
    assert len(transport.responses) == 1
    assert "should-not-surface" not in str(captured.value)
    assert "should-not-surface" not in json.dumps(captured.value.as_failure())


def test_worker_timeout_bounds_are_validated() -> None:
    with pytest.raises(ConnectorConfigurationError, match="timeouts"):
        Build123dHttpConnector(
            endpoint_url="https://geometry-worker.example/v1/author",
            connect_timeout_seconds=0,
        )


def test_worker_response_digest_tamper_is_rejected(tmp_path: Path) -> None:
    payload = _worker_payload()
    payload["artifacts"][1]["sha256"] = "0" * 64  # type: ignore[index]
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=QueueTransport(FakeResponse.json(payload)),
    )
    with pytest.raises(InvalidProviderResponseError):
        connector.generate(_request(), output_dir=tmp_path)
    assert not list(tmp_path.iterdir())


def test_worker_must_return_every_requested_format(tmp_path: Path) -> None:
    payload = _worker_payload()
    obj = b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
    payload["artifacts"][1].update(  # type: ignore[index]
        {
            "filename": "trusted_fixture.obj",
            "role": "mesh_geometry",
            "media_type": "model/obj",
            "content_base64": base64.b64encode(obj).decode("ascii"),
            "size_bytes": len(obj),
            "sha256": hashlib.sha256(obj).hexdigest(),
        }
    )
    payload["parts"][0]["artifact_filenames"] = ["trusted_fixture.obj"]  # type: ignore[index]
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=QueueTransport(FakeResponse.json(payload)),
    )
    with pytest.raises(InvalidProviderResponseError, match="omitted requested"):
        connector.generate(_request(), output_dir=tmp_path)


def test_connector_rejects_unsupported_format_before_http(tmp_path: Path) -> None:
    transport = QueueTransport()
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=transport,
    )
    request = AuthoringRequest(
        request_id="unsupported-format",
        prompt="Create a box",
        target_formats=("usd",),
    )
    with pytest.raises(UnsupportedCapabilityError):
        connector.generate(request, output_dir=tmp_path)
    assert not transport.requests


def test_image_binding_is_verified_during_request_validation() -> None:
    image = InlineInputArtifact.from_bytes(
        filename="reference.png",
        media_type="image/png",
        content=b"\x89PNG\r\n\x1a\nfixture",
    )
    payload = image.model_dump(mode="json")
    payload["sha256"] = "0" * 64
    with pytest.raises(ValidationError, match="SHA-256"):
        InlineInputArtifact.model_validate(payload)


def test_non_image_bytes_cannot_cross_the_image_authoring_boundary() -> None:
    with pytest.raises(ValidationError, match="signature"):
        InlineInputArtifact.from_bytes(
            filename="uploaded.py.png",
            media_type="image/png",
            content=b"print('not an image')",
        )


def test_worker_rejects_output_path_escape(tmp_path: Path) -> None:
    escaped = tmp_path / "outside.step"
    escaped.write_bytes(b"ISO-10303-21;\nEND-ISO-10303-21;\n")

    class EscapeBackend:
        isolation_kind: WorkerIsolationKind = "container"

        def execute(self, request: AuthoringRequest, *, workspace: Path) -> Build123dWorkerResult:
            del request, workspace
            return Build123dWorkerResult(
                provider_version="escape-test",
                source_revision="escape-test-revision",
                units="millimeter",
                up_axis="Z",
                artifacts=(
                    WorkerArtifact(
                        path=escaped,
                        filename="outside.step",
                        role="cad_geometry",
                        media_type="model/step",
                    ),
                ),
            )

    runner = Build123dWorkerRunner(EscapeBackend())
    with pytest.raises(UnsafeArtifactError, match="escapes"):
        runner.handle(_request().model_dump(mode="json"))


def test_materialization_is_create_only(tmp_path: Path) -> None:
    payload = _worker_payload()
    destination = tmp_path / "result"
    destination.mkdir()
    existing = destination / "trusted_fixture.step"
    existing.write_text("do not replace", encoding="utf-8")
    connector = Build123dHttpConnector(
        endpoint_url="https://geometry-worker.example/v1/author",
        supported_formats=("step",),
        transport=QueueTransport(FakeResponse.json(payload)),
    )
    with pytest.raises(UnsafeArtifactError, match="already exists"):
        connector.generate(_request(), output_dir=destination)
    assert existing.read_text(encoding="utf-8") == "do not replace"
    assert not (destination / "trusted_fixture.py").exists()


def test_production_connector_modules_have_no_local_python_executor() -> None:
    package = Path(__file__).parents[1] / "geometry_authoring_connectors"
    trees = [
        ast.parse((package / name).read_text(encoding="utf-8"))
        for name in ("build123d.py", "build123d_worker.py")
    ]
    imported = {
        alias.name
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or ""
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    )
    called_names = {
        node.func.id
        for tree in trees
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "build123d" not in imported
    assert "subprocess" not in imported
    assert "exec" not in called_names
