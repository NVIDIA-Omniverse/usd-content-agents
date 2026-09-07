# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import json
import re
import tomllib
from pathlib import Path

import pytest
from geometry_authoring_contracts import (
    GeometryAuthoringExportRequest,
    GeometryAuthoringParameterValue,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    validate_geometry_source_bundle_identity,
)
from pydantic import ValidationError

from geometry_authoring_connectors import (
    FORGECAD_AUTHORING_PROVIDER_ID,
    AuthoringRequest,
    ConnectorConfigurationError,
    DelegatedAuthoringHttpConnector,
    DelegatedGeometryAuthoringProvider,
    InlineInputArtifact,
    InvalidProviderResponseError,
    ProviderTransportError,
    UnsupportedCapabilityError,
    WireArtifact,
    WirePart,
    WireSemanticParameter,
    WireSourceBundle,
    WireVerificationAssertion,
)

from .conftest import FakeResponse, QueueTransport


def test_default_http_transport_ignores_proxy_and_netrc_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geometry_authoring_connectors import _http as http_module

    class Session:
        trust_env = True

    session = Session()
    monkeypatch.setattr(http_module.requests, "Session", lambda: session)

    http_module.BoundedHttpClient(
        provider_id="provider.v1",
        endpoint_alias="provider worker",
        base_url="https://workers.example/provider",
        bearer_token=None,
        connect_timeout_seconds=10.0,
        read_timeout_seconds=30.0,
    )

    assert session.trust_env is False


def test_http_transport_rejects_header_unsafe_bearer_tokens() -> None:
    from geometry_authoring_connectors import _http as http_module

    with pytest.raises(ConnectorConfigurationError, match="printable ASCII"):
        http_module.BoundedHttpClient(
            provider_id="provider.v1",
            endpoint_alias="provider worker",
            base_url="https://workers.example/provider",
            bearer_token="secret\nInjected: true",
            connect_timeout_seconds=10.0,
            read_timeout_seconds=30.0,
        )


def test_http_transport_rejects_control_bytes_from_signing_callbacks() -> None:
    from geometry_authoring_connectors import _http as http_module

    client = http_module.BoundedHttpClient(
        provider_id="provider.v1",
        endpoint_alias="provider worker",
        base_url="https://workers.example/provider",
        bearer_token=None,
        connect_timeout_seconds=10.0,
        read_timeout_seconds=30.0,
        transport=QueueTransport(),
        request_header_provider=lambda _method, _url, _content_type: {
            "Authorization": "signed\x00value"
        },
    )

    with pytest.raises(ConnectorConfigurationError, match="invalid header"):
        client.request_json("GET")


def test_wire_geometry_models_reject_non_finite_values() -> None:
    with pytest.raises(ValidationError, match="finite number"):
        WirePart(
            part_id="body",
            name="Body",
            transform=(float("nan"),) + (0.0,) * 15,
        )


@pytest.mark.parametrize(
    ("status_code", "error_type"),
    (
        (400, InvalidProviderResponseError),
        (503, ProviderTransportError),
    ),
)
def test_http_status_failures_have_correct_retry_classification(
    status_code: int,
    error_type: type[Exception],
) -> None:
    from geometry_authoring_connectors import _http as http_module

    client = http_module.BoundedHttpClient(
        provider_id="provider.v1",
        endpoint_alias="provider worker",
        base_url="https://workers.example/provider",
        bearer_token=None,
        connect_timeout_seconds=10.0,
        read_timeout_seconds=30.0,
        transport=QueueTransport(FakeResponse.json({}, status_code=status_code)),
    )

    with pytest.raises(error_type):
        client.request_json("GET")


def _payload(*, provider_id: str, native_source: bool) -> dict[str, object]:
    artifacts = [
        WireArtifact.from_bytes(
            filename="generated.step",
            role="cad_geometry",
            media_type="model/step",
            content=b"ISO-10303-21;\nEND-ISO-10303-21;\n",
        )
    ]
    if native_source:
        artifacts.append(
            WireArtifact.from_bytes(
                filename="generated.source.txt",
                role="native_source",
                media_type="text/plain",
                content=b"opaque provider-native provenance\n",
            )
        )
    return WireSourceBundle(
        provider_id=provider_id,
        provider_version="fixture-1",
        source_revision="fixture-revision-1",
        units="millimeter",
        up_axis="Z",
        artifacts=tuple(artifacts),
    ).model_dump(mode="json")


@pytest.mark.parametrize(
    ("provider_id", "label", "returns_native_source"),
    ((FORGECAD_AUTHORING_PROVIDER_ID, "ForgeCAD worker", True),),
)
def test_delegated_provider_round_trip_is_explicit_and_dependency_free(
    tmp_path: Path,
    provider_id: str,
    label: str,
    returns_native_source: bool,
) -> None:
    transport = QueueTransport(
        FakeResponse.json(
            _payload(
                provider_id=provider_id,
                native_source=returns_native_source,
            )
        )
    )
    provider = DelegatedGeometryAuthoringProvider(
        connector=DelegatedAuthoringHttpConnector(
            provider_id=provider_id,
            provider_label=label,
            endpoint_alias=provider_id,
            endpoint_url=f"https://workers.example/{provider_id}",
            bearer_token="worker-secret",
            supported_formats=("step",),
            supports_text=True,
            supports_image=False,
            supports_revision=True,
            returns_native_source=returns_native_source,
            transport=transport,
        ),
        artifact_root=tmp_path / provider_id,
        rights_assertion="Operator attests that this worker output is authorized.",
    )

    receipt = provider.generate(
        GeometryAuthoringRequest(
            request_id=f"{provider_id}-generation",
            prompt="Create a 20 mm cube.",
            target_profile="rigid_pick_place",
            requested_formats=("step",),
        )
    )

    assert receipt.disposition == "succeeded"
    assert receipt.provider.provider_id == provider_id
    assert receipt.source_bundle is not None
    assert receipt.source_bundle.rights.authorized_for_requested_use is True
    assert all(Path(item.artifact.path).is_file() for item in receipt.source_bundle.representations)
    request = transport.requests[0]
    assert request.kwargs["headers"]["Authorization"] == "Bearer worker-secret"
    assert "worker-secret" not in request.url


def test_delegated_provider_rolls_back_staging_after_adapter_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geometry_authoring_connectors import providers as provider_module

    provider_id = FORGECAD_AUTHORING_PROVIDER_ID
    artifact_root = tmp_path / "provider-artifacts"
    provider = DelegatedGeometryAuthoringProvider(
        connector=DelegatedAuthoringHttpConnector(
            provider_id=provider_id,
            provider_label="ForgeCAD worker",
            endpoint_alias=provider_id,
            endpoint_url="https://workers.example/forgecad",
            supported_formats=("step",),
            transport=QueueTransport(
                FakeResponse.json(_payload(provider_id=provider_id, native_source=False))
            ),
        ),
        artifact_root=artifact_root,
        rights_assertion="Operator authorizes this provider output.",
    )

    def fail_canonicalization(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("forced adapter failure")

    monkeypatch.setattr(
        provider_module,
        "canonicalize_materialized_bundle",
        fail_canonicalization,
    )

    receipt = provider.generate(
        GeometryAuthoringRequest(
            request_id="rollback-generation",
            prompt="Create a rigid block.",
            target_profile="rigid_pick_place",
            requested_formats=("step",),
        )
    )

    assert receipt.disposition == "failed"
    assert list(artifact_root.iterdir()) == []


def test_adapter_rollback_preserves_a_directory_swapped_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from geometry_authoring_connectors import providers as provider_module

    artifact_root = tmp_path / "provider-artifacts"
    artifact_root.mkdir()
    candidate = artifact_root / "request"
    candidate.mkdir()
    (candidate / "provider.step").write_bytes(b"provider")
    replacement = artifact_root / "replacement"
    replacement.mkdir()
    (replacement / "other-request.step").write_bytes(b"other-request")
    parked = artifact_root / "parked"
    original_rename = provider_module.os.rename
    raced = False

    def race_before_quarantine(
        source: str | Path,
        destination: str | Path,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
    ) -> None:
        nonlocal raced
        if not raced and source == candidate.name and src_dir_fd is not None:
            raced = True
            original_rename(candidate, parked)
            original_rename(replacement, candidate)
        original_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(provider_module.os, "rename", race_before_quarantine)

    provider_module._discard_owned_output_directory(artifact_root, candidate)

    assert raced is True
    assert (parked / "provider.step").read_bytes() == b"provider"
    preserved = list(artifact_root.rglob("other-request.step"))
    assert len(preserved) == 1
    assert preserved[0].read_bytes() == b"other-request"


def test_delegated_provider_rejects_a_symlinked_artifact_root(tmp_path: Path) -> None:
    actual_root = tmp_path / "actual"
    actual_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(actual_root, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symlink"):
        DelegatedGeometryAuthoringProvider(
            connector=DelegatedAuthoringHttpConnector(
                provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
                provider_label="ForgeCAD worker",
                endpoint_alias="forgecad-http",
                endpoint_url="https://workers.example/forgecad",
                supported_formats=("step",),
                transport=QueueTransport(),
            ),
            artifact_root=linked_root / "provider",
            rights_assertion="Operator authorizes this provider output.",
        )


@pytest.mark.parametrize(
    ("provider_id", "label", "returns_native_source"),
    ((FORGECAD_AUTHORING_PROVIDER_ID, "ForgeCAD worker", True),),
)
def test_delegated_providers_export_from_their_immutable_revision(
    tmp_path: Path,
    provider_id: str,
    label: str,
    returns_native_source: bool,
) -> None:
    transport = QueueTransport(
        FakeResponse.json(_payload(provider_id=provider_id, native_source=returns_native_source)),
        FakeResponse.json(_payload(provider_id=provider_id, native_source=returns_native_source)),
    )
    provider = DelegatedGeometryAuthoringProvider(
        connector=DelegatedAuthoringHttpConnector(
            provider_id=provider_id,
            provider_label=label,
            endpoint_alias=provider_id,
            endpoint_url=f"https://workers.example/{provider_id}",
            supported_formats=("step",),
            supports_export=True,
            returns_native_source=returns_native_source,
            transport=transport,
        ),
        artifact_root=tmp_path / provider_id,
        rights_assertion="Operator authorizes this provider output.",
    )
    generated = provider.generate(
        GeometryAuthoringRequest(
            request_id=f"{provider_id}-generate",
            prompt="Create a parameterized mounting block.",
            target_profile="rigid_pick_place",
            requested_formats=("step",),
        )
    )
    assert generated.source_bundle is not None

    exported = provider.export(
        GeometryAuthoringExportRequest(
            request_id=f"{provider_id}-export",
            source_bundle=generated.source_bundle,
            requested_formats=("step",),
        )
    )

    assert "export" in provider.capabilities().operations
    assert exported.disposition == "succeeded"
    assert exported.source_bundle is not None
    assert exported.source_bundle.provenance.parent_bundle_id == generated.source_bundle.bundle_id
    low_request = json.loads(transport.requests[1].kwargs["data"])
    assert low_request["operation"] == "export"
    assert low_request["prior_source_revision"] == "fixture-revision-1"
    assert low_request["prompt"] is None
    assert low_request["images"] == []
    assert low_request["parameters"] == {}


def test_delegated_connector_enforces_advertised_modalities_before_http(
    tmp_path: Path,
) -> None:
    transport = QueueTransport()
    connector = DelegatedAuthoringHttpConnector(
        provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
        provider_label="ForgeCAD worker",
        endpoint_alias="forgecad-http",
        endpoint_url="https://workers.example/forgecad",
        supported_formats=("step",),
        supports_image=False,
        transport=transport,
    )
    image = InlineInputArtifact.from_bytes(
        filename="reference.png",
        media_type="image/png",
        content=b"\x89PNG\r\n\x1a\nfixture",
    )
    request = AuthoringRequest(
        request_id="unsupported-image",
        images=(image,),
        target_formats=("step",),
    )

    with pytest.raises(UnsupportedCapabilityError, match="image input"):
        connector.generate(request, output_dir=tmp_path)
    assert not transport.requests


def test_delegated_connector_defaults_fail_closed_for_rich_features() -> None:
    connector = DelegatedAuthoringHttpConnector(
        provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
        provider_label="ForgeCAD worker",
        endpoint_alias="forgecad-http",
        endpoint_url="https://workers.example/forgecad",
    )

    capabilities = connector.capabilities()
    assert capabilities.formats == ("step",)
    assert capabilities.revision is True
    assert capabilities.export is False
    assert capabilities.parameter_definitions is False
    assert capabilities.semantic_parts is False
    assert capabilities.provider_assertions is False
    assert capabilities.max_family_variants == 0


@pytest.mark.parametrize(
    "overrides",
    (
        {"supports_parameter_definitions": True},
        {"max_family_variants": 4},
        {
            "supports_semantic_parameters": True,
            "supports_parameter_definitions": True,
            "supports_revision": False,
            "max_family_variants": 4,
        },
        {"max_family_variants": -1},
        {"max_family_variants": 65},
        {"max_family_variants": True},
    ),
)
def test_delegated_connector_rejects_incompatible_capability_settings(
    overrides: dict[str, object],
) -> None:
    with pytest.raises(ConnectorConfigurationError):
        DelegatedAuthoringHttpConnector(
            provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
            provider_label="ForgeCAD worker",
            endpoint_alias="forgecad-http",
            endpoint_url="https://workers.example/forgecad",
            **overrides,  # type: ignore[arg-type]
        )


def test_delegated_connector_rejects_revision_without_a_new_revision(
    tmp_path: Path,
) -> None:
    provider_id = "provider.v1"
    connector = DelegatedAuthoringHttpConnector(
        provider_id=provider_id,
        provider_label="Provider worker",
        endpoint_alias="provider-worker",
        endpoint_url="https://workers.example/provider",
        supported_formats=("step",),
        transport=QueueTransport(
            FakeResponse.json(_payload(provider_id=provider_id, native_source=False))
        ),
    )

    with pytest.raises(InvalidProviderResponseError, match="new immutable revision"):
        connector.revise(
            AuthoringRequest(
                request_id="revision-request",
                operation="revise",
                prompt="Make the body wider.",
                target_formats=("step",),
                prior_source_revision="fixture-revision-1",
            ),
            output_dir=tmp_path,
        )


def test_delegated_connector_requires_promised_native_source(tmp_path: Path) -> None:
    connector = DelegatedAuthoringHttpConnector(
        provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
        provider_label="ForgeCAD worker",
        endpoint_alias="forgecad-http",
        endpoint_url="https://workers.example/forgecad",
        supported_formats=("step",),
        returns_native_source=True,
        transport=QueueTransport(
            FakeResponse.json(
                _payload(
                    provider_id=FORGECAD_AUTHORING_PROVIDER_ID,
                    native_source=False,
                )
            )
        ),
    )

    with pytest.raises(InvalidProviderResponseError, match="native source"):
        connector.generate(
            AuthoringRequest(
                request_id="missing-native-source",
                prompt="Create a cube.",
                target_formats=("step",),
            ),
            output_dir=tmp_path,
        )


def test_delegated_provider_preserves_profile_parameters_parts_and_assertions(
    tmp_path: Path,
) -> None:
    provider_id = "private-authoring-worker"
    geometry = WireArtifact.from_bytes(
        filename="assembly.usda",
        role="render_geometry",
        media_type="model/vnd.usda",
        content=b'#usda 1.0\ndef Xform "Asset" {}\n',
    )
    collision = WireArtifact.from_bytes(
        filename="collision.usda",
        role="collision_geometry",
        media_type="model/vnd.usda",
        content=b'#usda 1.0\ndef Xform "Collision" {}\n',
    )
    wire_bundle = WireSourceBundle(
        provider_id=provider_id,
        provider_version="2026.08.23",
        source_revision="revision-parameterized-1",
        units="millimeter",
        up_axis="Z",
        artifacts=(geometry, collision),
        parts=(
            WirePart(
                part_id="housing",
                name="Housing",
                artifact_filenames=("assembly.usda", "collision.usda"),
            ),
        ),
        parameters=(
            WireSemanticParameter(
                name="key_spacing",
                value=19.0,
                unit="mm",
                minimum=16.0,
                maximum=24.0,
            ),
            WireSemanticParameter(
                name="target_profile",
                value="compact",
            ),
        ),
        verification_assertions=(
            WireVerificationAssertion(
                assertion_id="provider-topology",
                status="passed",
                summary="Provider reports valid topology.",
            ),
        ),
    )
    transport = QueueTransport(FakeResponse.json(wire_bundle.model_dump(mode="json")))
    provider = DelegatedGeometryAuthoringProvider(
        connector=DelegatedAuthoringHttpConnector(
            provider_id=provider_id,
            provider_label="Private authoring worker",
            endpoint_alias="private-authoring",
            endpoint_url="https://workers.example/private-authoring",
            supported_formats=("usda",),
            supports_semantic_parameters=True,
            supports_parameter_definitions=True,
            supports_semantic_parts=True,
            supports_provider_assertions=True,
            max_family_variants=64,
            returns_native_source=False,
            transport=transport,
        ),
        artifact_root=tmp_path / "outputs",
        rights_assertion="Operator authorizes this worker output.",
    )

    receipt = provider.generate(
        GeometryAuthoringRequest(
            request_id="parameterized-keyboard",
            prompt="Create a compact keyboard.",
            target_profile="rigid-pick-place",
            requested_formats=("usda",),
            parameters=(GeometryAuthoringParameterValue(name="target_profile", value="compact"),),
        )
    )

    assert receipt.disposition == "succeeded"
    assert receipt.source_bundle is not None
    source = validate_geometry_source_bundle_identity(receipt.source_bundle)
    assert {item.role for item in source.representations} == {
        "render_geometry",
        "collision_candidate",
    }
    assert source.parts[0].representation_ids == (
        "representation-001",
        "representation-002",
    )
    assert source.parameters[0].name == "key_spacing"
    assert source.verification_assertions[0].assertion_id == "provider-topology"
    low_request = json.loads(transport.requests[0].kwargs["data"])
    assert low_request["target_profile"] == "rigid-pick-place"
    assert low_request["parameters"] == {"target_profile": "compact"}


def test_delegated_revision_prefers_returned_topology_and_preserves_parameter_schema(
    tmp_path: Path,
) -> None:
    provider_id = "parametric-worker"

    def payload(revision: str, width: float, *, revised: bool) -> dict[str, object]:
        geometry = WireArtifact.from_bytes(
            filename="keyboard.step",
            role="cad_geometry",
            media_type="model/step",
            content=b"ISO-10303-21;\nEND-ISO-10303-21;\n",
        )
        return WireSourceBundle(
            provider_id=provider_id,
            provider_version="fixture-1",
            source_revision=revision,
            units="millimeter",
            up_axis="Z",
            artifacts=(geometry,),
            parts=(
                WirePart(
                    part_id="body",
                    name="Body",
                    artifact_filenames=("keyboard.step",),
                ),
                *(
                    (
                        WirePart(
                            part_id="numpad",
                            name="Numpad",
                            parent_part_id="body",
                            artifact_filenames=("keyboard.step",),
                        ),
                    )
                    if revised
                    else ()
                ),
            ),
            parameters=(
                WireSemanticParameter(
                    name="width_mm",
                    value=width,
                    unit="mm",
                    minimum=200.0,
                    maximum=500.0,
                    step=10.0,
                    effects=("exact_geometry", "topology"),
                ),
            ),
        ).model_dump(mode="json")

    provider = DelegatedGeometryAuthoringProvider(
        connector=DelegatedAuthoringHttpConnector(
            provider_id=provider_id,
            provider_label="Parametric authoring worker",
            endpoint_alias=provider_id,
            endpoint_url="https://workers.example/parametric",
            supported_formats=("step",),
            supports_semantic_parameters=True,
            supports_parameter_definitions=True,
            transport=QueueTransport(
                FakeResponse.json(payload("revision-1", 300.0, revised=False)),
                FakeResponse.json(payload("revision-2", 400.0, revised=True)),
            ),
        ),
        artifact_root=tmp_path / "outputs",
        rights_assertion="Authorized parametric fixture output.",
    )
    generated = provider.generate(
        GeometryAuthoringRequest(
            request_id="keyboard-generate",
            prompt="Create a parameterized keyboard.",
            target_profile="rigid-pick-place",
            requested_formats=("step",),
            parameters=(GeometryAuthoringParameterValue(name="width_mm", value=300.0),),
        )
    )
    assert generated.source_bundle is not None

    revised = provider.revise(
        GeometryAuthoringRevisionRequest(
            request_id="keyboard-revise",
            source_bundle=generated.source_bundle,
            parameter_overrides=(GeometryAuthoringParameterValue(name="width_mm", value=400.0),),
            requested_formats=("step",),
        )
    )

    assert revised.disposition == "succeeded"
    assert revised.source_bundle is not None
    assert [item.part_id for item in revised.source_bundle.parts] == ["body", "numpad"]
    parameter = revised.source_bundle.parameters[0]
    assert parameter.value == 400.0
    assert parameter.step == 10.0
    assert parameter.effects == ("exact_geometry", "topology")


def test_public_connectors_do_not_import_or_execute_vendor_runtimes() -> None:
    package = Path(__file__).parents[1] / "geometry_authoring_connectors"
    module_names = ("build123d.py", "delegated.py", "forgecad.py")
    trees = [ast.parse((package / name).read_text(encoding="utf-8")) for name in module_names]
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

    assert not {"build123d", "forgecad", "onshape"}.intersection(imported)
    assert "subprocess" not in imported
    assert not {"exec", "eval"}.intersection(called_names)


def test_public_packages_do_not_depend_on_vendor_authoring_runtimes() -> None:
    repository = Path(__file__).resolve().parents[4]
    pyprojects = (
        repository / "agentic/packages/geometry_authoring_connectors/pyproject.toml",
        repository / "apps/geometry_agent_service/pyproject.toml",
    )
    dependencies = {
        re.split(r"[<>=!~;\[]", dependency, maxsplit=1)[0].strip().lower()
        for pyproject in pyprojects
        for dependency in tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
            "dependencies"
        ]
    }

    assert not {"build123d", "forgecad", "onshape"}.intersection(dependencies)
