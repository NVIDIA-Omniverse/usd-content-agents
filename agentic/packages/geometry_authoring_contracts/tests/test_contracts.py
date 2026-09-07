# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

import geometry_authoring_contracts as provider_module
from geometry_authoring_contracts import (
    ArtifactJsonGeometryAuthoringProvider,
    GeometryArtifactBinding,
    GeometryAuthoringCapabilityReport,
    GeometryAuthoringExportRequest,
    GeometryAuthoringFailure,
    GeometryAuthoringFamilyRequest,
    GeometryAuthoringInputDrift,
    GeometryAuthoringInvalidResponse,
    GeometryAuthoringParameterValue,
    GeometryAuthoringProvider,
    GeometryAuthoringProviderError,
    GeometryAuthoringProviderIdentity,
    GeometryAuthoringProviderReceipt,
    GeometryAuthoringReference,
    GeometryAuthoringRequest,
    GeometryAuthoringRevisionRequest,
    GeometryAuthoringVariant,
    GeometryCoordinateSystem,
    GeometryPartBinding,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySemanticParameter,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    GeometryVerificationAssertion,
    HttpJsonGeometryAuthoringProvider,
    geometry_authoring_request_digest,
    geometry_source_bundle_id,
    resolve_semantic_parameter_overrides,
    validate_geometry_source_bundle_identity,
    validate_returned_parameter_state,
)

_PROVIDER_ID = "reference-build123d"
_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64
_DIGEST_C = "c" * 64


def _binding(path: Path) -> GeometryArtifactBinding:
    return GeometryArtifactBinding(
        path=str(path.resolve()),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        size_bytes=path.stat().st_size,
    )


def _identity(provider_id: str = _PROVIDER_ID) -> GeometryAuthoringProviderIdentity:
    return GeometryAuthoringProviderIdentity(
        provider_id=provider_id,
        provider_version="2026.08.22",
    )


def _capabilities(
    provider_id: str = _PROVIDER_ID,
) -> GeometryAuthoringCapabilityReport:
    return GeometryAuthoringCapabilityReport(
        provider=_identity(provider_id),
        operations=("generate", "revise", "export"),
        input_modalities=("text", "image", "text_image", "existing_source"),
        output_formats=("step", "usd"),
        supports_semantic_parameters=True,
        returns_native_source=True,
        max_reference_artifacts=8,
        max_prompt_characters=32_768,
    )


def _write_model(path: Path, model: Any) -> Path:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    return path


def _source_bundle(
    tmp_path: Path,
    *,
    request_digest: str,
    provider_id: str = _PROVIDER_ID,
    parent_bundle_id: str | None = None,
    revision: str = "revision-1",
    filename: str = "source.step",
    source_text: str = "ISO-10303-21;\nEND-ISO-10303-21;\n",
    role: str = "design_exchange",
    format: str = "step",
    provenance_inputs: tuple[GeometryArtifactBinding, ...] = (),
    upstream_edit_uri: str | None = None,
) -> GeometrySourceBundle:
    source_path = tmp_path / filename
    source_path.write_text(source_text, encoding="utf-8")
    producer = _identity(provider_id)
    coordinate_system = GeometryCoordinateSystem(
        meters_per_unit=0.001,
        up_axis="Z",
        forward_axis="+Y",
    )
    representations = (
        GeometryRepresentationBinding(
            representation_id=f"{format}-source",
            role=role,
            format=format,
            media_type="model/step" if format == "step" else "model/vnd.usd",
            artifact=_binding(source_path),
        ),
    )
    parts = (
        GeometryPartBinding(
            part_id="body",
            name="Body",
            representation_ids=(f"{format}-source",),
        ),
    )
    parameters = (
        GeometrySemanticParameter(
            name="width",
            value=40.0,
            unit="mm",
            minimum=1.0,
            maximum=100.0,
        ),
    )
    provenance = GeometrySourceProvenance(
        request_digest=request_digest,
        parent_bundle_id=parent_bundle_id,
        input_artifacts=provenance_inputs,
        upstream_edit_uri=upstream_edit_uri,
    )
    rights = GeometryRightsAssertion(
        assertion="Provider confirms this source may enter the workflow.",
        license_identifier="Apache-2.0",
    )
    return GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=producer,
            source_revision=revision,
            coordinate_system=coordinate_system,
            representations=representations,
            parts=parts,
            parameters=parameters,
            provenance=provenance,
            rights=rights,
        ),
        producer=producer,
        source_revision=revision,
        coordinate_system=coordinate_system,
        representations=representations,
        parts=parts,
        parameters=parameters,
        provenance=provenance,
        rights=rights,
    )


def _with_parameters(
    source: GeometrySourceBundle,
    parameters: tuple[GeometrySemanticParameter, ...],
) -> GeometrySourceBundle:
    updated = source.model_copy(update={"parameters": parameters})
    return updated.model_copy(
        update={
            "bundle_id": geometry_source_bundle_id(
                provider=updated.producer,
                source_revision=updated.source_revision,
                coordinate_system=updated.coordinate_system,
                representations=updated.representations,
                parts=updated.parts,
                parameters=updated.parameters,
                verification_assertions=updated.verification_assertions,
                provenance=updated.provenance,
                rights=updated.rights,
            )
        }
    )


def _generation_request(
    *,
    references: tuple[GeometryAuthoringReference, ...] = (),
) -> GeometryAuthoringRequest:
    return GeometryAuthoringRequest(
        request_id="request-generate-1",
        prompt="Create a rigid mounting bracket with two bolt holes.",
        references=references,
        target_profile="rigid-body-pick-place",
        requested_formats=("step",),
        parameters=(GeometryAuthoringParameterValue(name="width", value=40.0, unit="mm"),),
    )


def _success_receipt(
    operation: str,
    request: (
        GeometryAuthoringRequest | GeometryAuthoringRevisionRequest | GeometryAuthoringExportRequest
    ),
    source: GeometrySourceBundle,
    *,
    provider_id: str = _PROVIDER_ID,
) -> GeometryAuthoringProviderReceipt:
    return GeometryAuthoringProviderReceipt(
        provider=_identity(provider_id),
        operation=operation,
        request_digest=geometry_authoring_request_digest(request),
        disposition="succeeded",
        source_bundle=source,
    )


def _artifact_provider(
    tmp_path: Path,
    *,
    generate_request: GeometryAuthoringRequest | None = None,
    generate_source: GeometrySourceBundle | None = None,
    revision_request: GeometryAuthoringRevisionRequest | None = None,
    revision_source: GeometrySourceBundle | None = None,
    export_request: GeometryAuthoringExportRequest | None = None,
    export_source: GeometrySourceBundle | None = None,
) -> ArtifactJsonGeometryAuthoringProvider:
    capability_path = _write_model(tmp_path / "capabilities.json", _capabilities())
    paths: dict[str, Path] = {}
    if generate_request is not None and generate_source is not None:
        paths["generate"] = _write_model(
            tmp_path / "generate-receipt.json",
            _success_receipt("generate", generate_request, generate_source),
        )
    if revision_request is not None and revision_source is not None:
        paths["revise"] = _write_model(
            tmp_path / "revise-receipt.json",
            _success_receipt("revise", revision_request, revision_source),
        )
    if export_request is not None and export_source is not None:
        paths["export"] = _write_model(
            tmp_path / "export-receipt.json",
            _success_receipt("export", export_request, export_source),
        )
    return ArtifactJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        capabilities_path=capability_path,
        generate_receipt_path=paths.get("generate"),
        revise_receipt_path=paths.get("revise"),
        export_receipt_path=paths.get("export"),
    )


def test_geometry_source_and_request_models_are_strict_and_frozen(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )

    assert source.schema_version == "geometry.source.v1"
    assert request.schema_version.endswith("geometry-authoring-request.v1")
    with pytest.raises(ValidationError):
        request.prompt = "mutated"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        GeometryAuthoringRequest.model_validate(
            {**request.model_dump(mode="json"), "unknown": True}
        )


def test_source_identity_is_path_independent_and_binds_all_semantics(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    assert validate_geometry_source_bundle_identity(source) is source

    relocated_path = tmp_path / "relocated.step"
    relocated_path.write_bytes(Path(source.representations[0].artifact.path).read_bytes())
    relocated_representation = source.representations[0].model_copy(
        update={"artifact": _binding(relocated_path)}
    )
    relocated = source.model_copy(update={"representations": (relocated_representation,)})
    assert validate_geometry_source_bundle_identity(relocated) is relocated

    changed_representation = relocated_representation.model_copy(
        update={"root_prim_path": "/Asset"}
    )
    changed_assertions = (
        GeometryVerificationAssertion(
            assertion_id="provider-topology",
            status="passed",
            summary="Provider reports valid topology.",
        ),
    )
    semantic_ids = {
        geometry_source_bundle_id(
            provider=source.producer,
            source_revision=source.source_revision,
            coordinate_system=source.coordinate_system,
            representations=(changed_representation,),
            parts=source.parts,
            parameters=source.parameters,
            verification_assertions=source.verification_assertions,
            provenance=source.provenance,
            rights=source.rights,
        ),
        geometry_source_bundle_id(
            provider=source.producer,
            source_revision=source.source_revision,
            coordinate_system=source.coordinate_system,
            representations=source.representations,
            parts=source.parts,
            parameters=source.parameters,
            verification_assertions=changed_assertions,
            provenance=source.provenance,
            rights=source.rights,
        ),
        geometry_source_bundle_id(
            provider=source.producer,
            source_revision=source.source_revision,
            coordinate_system=source.coordinate_system,
            representations=source.representations,
            parts=source.parts,
            parameters=source.parameters,
            verification_assertions=source.verification_assertions,
            provenance=source.provenance,
            rights=source.rights.model_copy(
                update={"assertion": "A different authorization assertion."}
            ),
        ),
    }
    assert source.bundle_id not in semantic_ids
    assert len(semantic_ids) == 3


def test_legacy_parameter_documents_keep_their_v1_source_identity(tmp_path: Path) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    payload = source.model_dump(mode="json")
    for parameter in payload["parameters"]:
        for field in (
            "value_type",
            "step",
            "choices",
            "group",
            "semantic_role",
            "effects",
            "affects",
            "visible",
        ):
            parameter.pop(field, None)

    restored = GeometrySourceBundle.model_validate(payload)

    assert restored.parameters[0].value_type == "number"
    assert validate_geometry_source_bundle_identity(restored) is restored


def test_provider_adapter_rejects_a_forged_source_bundle_identity(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    ).model_copy(update={"bundle_id": _DIGEST_C})
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse, match="canonical"):
        provider.generate(request)


def test_source_bundle_rejects_duplicate_representations_and_part_cycles(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    representation = source.representations[0]

    with pytest.raises(ValidationError, match="representation IDs must be unique"):
        GeometrySourceBundle.model_validate(
            {
                **source.model_dump(mode="json"),
                "representations": [
                    representation.model_dump(mode="json"),
                    representation.model_dump(mode="json"),
                ],
            }
        )

    parts = (
        GeometryPartBinding(part_id="a", name="A", parent_part_id="b"),
        GeometryPartBinding(part_id="b", name="B", parent_part_id="a"),
    )
    with pytest.raises(ValidationError, match="hierarchy contains a cycle"):
        GeometrySourceBundle.model_validate({**source.model_dump(mode="json"), "parts": parts})


def test_models_reject_invalid_axes_ranges_requests_and_rights(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="axes must be different"):
        GeometryCoordinateSystem(
            meters_per_unit=1.0,
            up_axis="Z",
            forward_axis="-Z",
        )
    with pytest.raises(ValidationError, match="below its minimum"):
        GeometrySemanticParameter(name="width", value=0.5, minimum=1.0)
    with pytest.raises(ValidationError):
        GeometryCoordinateSystem(meters_per_unit=float("nan"))
    with pytest.raises(ValidationError, match="requires text or reference"):
        GeometryAuthoringRequest(
            request_id="empty-request",
            target_profile="rigid",
            requested_formats=("step",),
        )
    with pytest.raises(ValidationError, match="formats must be unique"):
        GeometryAuthoringExportRequest(
            request_id="duplicate-formats",
            source_bundle=_source_bundle(tmp_path, request_digest=_DIGEST_A),
            requested_formats=("step", "step"),
        )
    with pytest.raises(ValidationError):
        GeometryRightsAssertion(
            authorized_for_requested_use=False,
            assertion="Not permitted.",
        )


def test_rich_semantic_parameters_validate_choices_steps_and_family_rows(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    parameters = (
        GeometrySemanticParameter(
            name="width_mm",
            value=40.0,
            value_type="number",
            unit="mm",
            minimum=20.0,
            maximum=100.0,
            step=5.0,
            semantic_role="overall_width",
            effects=("exact_geometry",),
            affects=("body",),
        ),
        GeometrySemanticParameter(
            name="layout",
            value="ansi",
            choices=("ansi", "iso", "ortholinear"),
            group="Layout",
            effects=("topology",),
        ),
    )
    source = source.model_copy(update={"parameters": parameters})
    source = source.model_copy(
        update={
            "bundle_id": geometry_source_bundle_id(
                provider=source.producer,
                source_revision=source.source_revision,
                coordinate_system=source.coordinate_system,
                representations=source.representations,
                parts=source.parts,
                parameters=source.parameters,
                verification_assertions=source.verification_assertions,
                provenance=source.provenance,
                rights=source.rights,
            )
        }
    )
    family = GeometryAuthoringFamilyRequest(
        request_id="keyboard-family",
        source_bundle=source,
        variants=(
            GeometryAuthoringVariant(
                variant_id="compact-iso",
                parameter_overrides=(
                    GeometryAuthoringParameterValue(name="width_mm", value=60.0),
                    GeometryAuthoringParameterValue(name="layout", value="iso"),
                ),
            ),
        ),
        requested_formats=("step",),
    )

    resolved = resolve_semantic_parameter_overrides(
        parameters,
        family.variants[0].parameter_overrides,
    )
    assert {item.name: item.value for item in resolved} == {
        "width_mm": 60.0,
        "layout": "iso",
    }
    with pytest.raises(ValidationError, match="step"):
        GeometryAuthoringFamilyRequest(
            request_id="bad-step",
            source_bundle=source,
            variants=(
                GeometryAuthoringVariant(
                    variant_id="bad",
                    parameter_overrides=(
                        GeometryAuthoringParameterValue(name="width_mm", value=61.0),
                    ),
                ),
            ),
        )
    with pytest.raises(ValidationError, match="choices"):
        GeometryAuthoringFamilyRequest(
            request_id="bad-choice",
            source_bundle=source,
            variants=(
                GeometryAuthoringVariant(
                    variant_id="bad",
                    parameter_overrides=(
                        GeometryAuthoringParameterValue(name="layout", value="invented"),
                    ),
                ),
            ),
        )


def test_complete_parameter_state_rejects_definition_and_membership_drift() -> None:
    expected = (
        GeometrySemanticParameter(
            name="width_mm",
            value=40.0,
            unit="mm",
            minimum=20.0,
            maximum=100.0,
            step=5.0,
            effects=("exact_geometry",),
        ),
    )

    validate_returned_parameter_state(expected, expected)

    changed_definition = (expected[0].model_copy(update={"minimum": 10.0}),)
    with pytest.raises(ValueError, match="definition or value"):
        validate_returned_parameter_state(changed_definition, expected)
    with pytest.raises(ValueError, match="missing width_mm"):
        validate_returned_parameter_state((), expected)


def test_upstream_edit_uri_is_credential_free_and_never_fetched(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationError, match="credential-free HTTP URL"):
        GeometrySourceProvenance(
            request_digest=_DIGEST_A,
            upstream_edit_uri="https://user:secret@cad.example.test/document/1",
        )

    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        upstream_edit_uri="https://cad.example.test/document/1/version/2",
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    assert provider.generate(request).source_bundle == source


def test_artifact_adapter_supports_all_operations_and_protocol(
    tmp_path: Path,
) -> None:
    generate_request = _generation_request()
    generated = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(generate_request),
        filename="generated.step",
    )
    revision_request = GeometryAuthoringRevisionRequest(
        request_id="request-revise-1",
        source_bundle=generated,
        instructions="Increase width to 50 mm while preserving both holes.",
        parameter_overrides=(GeometryAuthoringParameterValue(name="width", value=50.0, unit="mm"),),
        requested_formats=("step",),
    )
    revised = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(revision_request),
        parent_bundle_id=generated.bundle_id,
        revision="revision-2",
        filename="revised.step",
    )
    revised = _with_parameters(
        revised,
        (revised.parameters[0].model_copy(update={"value": 50.0}),),
    )
    export_request = GeometryAuthoringExportRequest(
        request_id="request-export-1",
        source_bundle=revised,
        requested_formats=("usd",),
    )
    exported = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(export_request),
        parent_bundle_id=revised.bundle_id,
        revision="revision-2",
        filename="exported.usda",
        source_text="#usda 1.0\n",
        role="render_geometry",
        format="usd",
    )
    exported = _with_parameters(
        exported,
        (exported.parameters[0].model_copy(update={"value": 50.0}),),
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=generate_request,
        generate_source=generated,
        revision_request=revision_request,
        revision_source=revised,
        export_request=export_request,
        export_source=exported,
    )

    assert isinstance(provider, GeometryAuthoringProvider)
    assert provider.capabilities() == _capabilities()
    assert provider.generate(generate_request).source_bundle == generated
    assert provider.revise(revision_request).source_bundle == revised
    assert provider.export(export_request).source_bundle == exported


@pytest.mark.parametrize("adapter", ("artifact", "http"))
def test_explicit_adapters_allow_first_parameters_on_revision(
    tmp_path: Path,
    adapter: str,
) -> None:
    generated_request = _generation_request()
    generated = _with_parameters(
        _source_bundle(
            tmp_path,
            request_digest=geometry_authoring_request_digest(generated_request),
            filename="unparameterized.step",
        ),
        (),
    )
    revision_request = GeometryAuthoringRevisionRequest(
        request_id="request-first-parameter",
        source_bundle=generated,
        parameter_overrides=(
            GeometryAuthoringParameterValue(
                name="height",
                value=18.0,
                unit="mm",
            ),
        ),
        requested_formats=("step",),
    )
    revised = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(revision_request),
        parent_bundle_id=generated.bundle_id,
        revision="revision-first-parameter",
        filename="parameterized.step",
    )
    revised = _with_parameters(
        revised,
        (
            GeometrySemanticParameter(
                name="height",
                value=18.0,
                unit="mm",
            ),
        ),
    )
    if adapter == "artifact":
        provider: GeometryAuthoringProvider = _artifact_provider(
            tmp_path,
            revision_request=revision_request,
            revision_source=revised,
        )
    else:
        session = _HttpSession(
            _HttpResponse(_capabilities()),
            _HttpResponse(_success_receipt("revise", revision_request, revised)),
        )
        provider = HttpJsonGeometryAuthoringProvider(
            provider_id=_PROVIDER_ID,
            endpoint_alias="first-parameter-reference",
            endpoint_url="https://authoring.example.test/v1",
            artifact_roots=(tmp_path,),
            session=session,
        )

    assert provider.revise(revision_request).source_bundle == revised


def test_artifact_adapter_does_not_execute_native_source(tmp_path: Path) -> None:
    marker = tmp_path / "must-not-exist"
    request = GeometryAuthoringRequest(
        request_id="native-source-request",
        prompt="Create a bracket and retain its provider-native source.",
        target_profile="rigid-body-pick-place",
        requested_formats=("step",),
    )
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    native_path = tmp_path / "provider_source.py"
    native_path.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )
    native_source = GeometryRepresentationBinding(
        representation_id="provider-native-source",
        role="native_source",
        format="py",
        media_type="text/x-python",
        artifact=_binding(native_path),
    )
    source = source.model_copy(update={"representations": (*source.representations, native_source)})
    source = source.model_copy(
        update={
            "bundle_id": geometry_source_bundle_id(
                provider=source.producer,
                source_revision=source.source_revision,
                coordinate_system=source.coordinate_system,
                representations=source.representations,
                parts=source.parts,
                parameters=source.parameters,
                verification_assertions=source.verification_assertions,
                provenance=source.provenance,
                rights=source.rights,
            )
        }
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    assert provider.generate(request).source_bundle == source
    assert not marker.exists()


def test_missing_artifact_operation_fails_without_fallback(tmp_path: Path) -> None:
    capability_path = _write_model(tmp_path / "capabilities.json", _capabilities())
    provider = ArtifactJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        capabilities_path=capability_path,
    )

    with pytest.raises(GeometryAuthoringProviderError) as exc_info:
        provider.generate(_generation_request())

    assert exc_info.value.failure.code == "unsupported_operation"
    assert "no fallback" in exc_info.value.failure.summary


def test_artifact_response_drift_is_rejected(tmp_path: Path) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )
    receipt_path = tmp_path / "generate-receipt.json"
    receipt_path.write_text('{"changed":true}\n', encoding="utf-8")

    with pytest.raises(GeometryAuthoringInputDrift) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure.code == "input_drift"
    assert exc_info.value.failure.drifted_artifacts == (str(receipt_path.resolve()),)


def test_request_input_drift_is_rejected_before_artifact_response(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"original-image")
    reference = GeometryAuthoringReference(
        reference_id="front-view",
        kind="image",
        media_type="image/png",
        artifact=_binding(image_path),
    )
    request = _generation_request(references=(reference,))
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )
    image_path.write_bytes(b"changed-image")

    with pytest.raises(GeometryAuthoringInputDrift) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure.drifted_artifacts == (str(image_path.resolve()),)


def test_output_artifact_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )
    Path(source.representations[0].artifact.path).write_text(
        "substituted output\n",
        encoding="utf-8",
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure.code == "artifact_mismatch"


@pytest.mark.parametrize(
    ("change", "message"),
    (
        ({"request_digest": _DIGEST_C}, "exact selected request"),
        ({"operation": "export"}, "exact selected request"),
    ),
)
def test_receipt_must_bind_exact_request_and_operation(
    tmp_path: Path,
    change: dict[str, str],
    message: str,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    receipt = _success_receipt("generate", request, source)
    receipt_path = tmp_path / "generate-receipt.json"
    document = receipt.model_dump(mode="json")
    document.update(change)
    receipt_path.write_text(json.dumps(document), encoding="utf-8")
    capability_path = _write_model(tmp_path / "capabilities.json", _capabilities())
    provider = ArtifactJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        capabilities_path=capability_path,
        generate_receipt_path=receipt_path,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse, match=message):
        provider.generate(request)


def test_receipt_must_bind_requested_formats_and_lineage(tmp_path: Path) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        parent_bundle_id=_DIGEST_A,
        filename="wrong-lineage.step",
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse, match="lineage"):
        provider.generate(request)


def test_generic_usd_request_accepts_a_specific_usda_representation(
    tmp_path: Path,
) -> None:
    request = _generation_request().model_copy(update={"requested_formats": ("usd",)})
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        filename="source.usda",
        source_text="#usda 1.0\n",
        role="render_geometry",
        format="usda",
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    assert provider.generate(request).source_bundle == source


@pytest.mark.parametrize("role", ("native_source", "supporting_asset"))
def test_receipt_does_not_count_non_deliverable_artifacts_as_requested_exports(
    tmp_path: Path,
    role: str,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        filename=f"{role}.step",
        role=role,
        format="step",
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    with pytest.raises(
        GeometryAuthoringInvalidResponse,
        match="requested exports",
    ):
        provider.generate(request)


@pytest.mark.parametrize(
    ("revision", "returned_width"), (("revision-1", 50.0), ("revision-2", 40.0))
)
def test_revision_receipt_requires_a_new_revision_and_complete_parameter_state(
    tmp_path: Path,
    revision: str,
    returned_width: float,
) -> None:
    generation_request = _generation_request()
    baseline = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(generation_request),
        filename="baseline.step",
    )
    request = GeometryAuthoringRevisionRequest(
        request_id="revision-contract-test",
        source_bundle=baseline,
        instructions="Make the body 50 mm wide.",
        parameter_overrides=(GeometryAuthoringParameterValue(name="width", value=50.0, unit="mm"),),
        requested_formats=("step",),
    )
    returned = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        parent_bundle_id=baseline.bundle_id,
        revision=revision,
        filename=f"returned-{returned_width}.step",
    )
    returned = _with_parameters(
        returned,
        (returned.parameters[0].model_copy(update={"value": returned_width}),),
    )
    provider = _artifact_provider(
        tmp_path,
        revision_request=request,
        revision_source=returned,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse, match="immutable revision"):
        provider.revise(request)


def test_export_receipt_preserves_complete_parameter_state(tmp_path: Path) -> None:
    generation_request = _generation_request()
    baseline = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(generation_request),
        filename="baseline.step",
    )
    request = GeometryAuthoringExportRequest(
        request_id="export-contract-test",
        source_bundle=baseline,
        requested_formats=("step",),
    )
    returned = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        parent_bundle_id=baseline.bundle_id,
        revision=baseline.source_revision,
        filename="exported.step",
    )
    returned = _with_parameters(
        returned,
        (returned.parameters[0].model_copy(update={"minimum": 2.0}),),
    )
    provider = _artifact_provider(
        tmp_path,
        export_request=request,
        export_source=returned,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse, match="parameter state"):
        provider.export(request)


def test_provider_declared_failure_remains_typed_and_does_not_fallback(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    failure = GeometryAuthoringFailure(
        provider_id=_PROVIDER_ID,
        operation="generate",
        code="provider_failure",
        summary="Provider could not construct valid geometry.",
    )
    receipt = GeometryAuthoringProviderReceipt(
        provider=_identity(),
        operation="generate",
        request_digest=geometry_authoring_request_digest(request),
        disposition="failed",
        failure=failure,
    )
    capability_path = _write_model(tmp_path / "capabilities.json", _capabilities())
    receipt_path = _write_model(tmp_path / "failure.json", receipt)
    provider = ArtifactJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        capabilities_path=capability_path,
        generate_receipt_path=receipt_path,
    )

    with pytest.raises(GeometryAuthoringProviderError) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure == failure
    assert exc_info.value.receipt == receipt
    assert exc_info.value.receipt is not None
    assert exc_info.value.receipt.fallback_used is False


class _HttpResponse:
    def __init__(
        self,
        payload: Any,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        raw: bytes | None = None,
        chunks: tuple[bytes, ...] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._raw = raw if raw is not None else payload.model_dump_json().encode()
        self._chunks = chunks
        self.closed = False

    def iter_content(self, *, chunk_size: int) -> Any:
        assert chunk_size == 64 * 1024
        if self._chunks is not None:
            yield from self._chunks
        else:
            yield self._raw

    def close(self) -> None:
        self.closed = True


class _HttpSession:
    def __init__(
        self,
        capability_response: _HttpResponse,
        operation_response: _HttpResponse,
    ) -> None:
        self.capability_response = capability_response
        self.operation_response = operation_response
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> _HttpResponse:
        self.calls.append({"method": "get", "url": url, **kwargs})
        return self.capability_response

    def post(self, url: str, **kwargs: Any) -> _HttpResponse:
        self.calls.append({"method": "post", "url": url, **kwargs})
        return self.operation_response


def test_http_adapter_sends_exact_json_with_separate_bearer_and_no_redirects(
    tmp_path: Path,
) -> None:
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    capability_response = _HttpResponse(_capabilities())
    operation_response = _HttpResponse(_success_receipt("generate", request, source))
    session = _HttpSession(capability_response, operation_response)
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="build123d-reference",
        endpoint_url="https://authoring.example.test/v1/geometry/",
        bearer_token="separate-secret",
        timeout_seconds=9.0,
        artifact_roots=(tmp_path,),
        session=session,
    )

    assert provider.capabilities() == _capabilities()
    assert provider.generate(request).source_bundle == source
    assert len(session.calls) == 2
    capability_call, generate_call = session.calls
    assert capability_call["url"].endswith("/capabilities")
    assert capability_call["allow_redirects"] is False
    assert capability_call["stream"] is True
    assert generate_call["url"] == ("https://authoring.example.test/v1/geometry/generate")
    assert generate_call["timeout"] == 9.0
    assert generate_call["allow_redirects"] is False
    assert generate_call["stream"] is True
    assert generate_call["headers"]["Authorization"] == "Bearer separate-secret"
    assert generate_call["json"] == request.model_dump(mode="json")
    assert "separate-secret" not in generate_call["url"]
    assert "separate-secret" not in json.dumps(generate_call["json"])
    assert capability_response.closed is True
    assert operation_response.closed is True


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"endpoint_alias": "  "}, "alias must not be empty"),
        (
            {"endpoint_url": "https://user:secret@authoring.example.test/v1"},
            "credential-free HTTP URL",
        ),
        (
            {"endpoint_url": "http://authoring.example.test/v1"},
            "require HTTPS transport",
        ),
        (
            {"endpoint_url": "https://authoring.example.test/v1?token=secret"},
            "credential-free HTTP URL",
        ),
        ({"timeout_seconds": 0.0}, "timeout must be positive"),
        ({"bearer_token": "  "}, "token must be"),
        ({"bearer_token": "secret\nInjected: value"}, "token must be"),
    ),
)
def test_http_adapter_rejects_unsafe_configuration(
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "provider_id": _PROVIDER_ID,
        "endpoint_alias": "reference",
        "endpoint_url": "https://authoring.example.test/v1",
    }
    arguments.update(overrides)

    with pytest.raises(ValueError, match=message):
        HttpJsonGeometryAuthoringProvider(**arguments)


@pytest.mark.parametrize(
    "endpoint_url",
    (
        "http://localhost:8080/v1",
        "http://127.1.2.3:8080/v1",
        "http://[::1]:8080/v1",
    ),
)
def test_http_adapter_allows_explicit_loopback_http(endpoint_url: str) -> None:
    HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="loopback",
        endpoint_url=endpoint_url,
        session=object(),
    )


@pytest.mark.parametrize(
    ("response", "code"),
    (
        (
            _HttpResponse(
                _capabilities(),
                headers={"Content-Length": str(16 * 1024 * 1024 + 1)},
            ),
            "invalid_response",
        ),
        (
            _HttpResponse(_capabilities(), headers={"Content-Length": "invalid"}),
            "invalid_response",
        ),
        (
            _HttpResponse(_capabilities(), status_code=503),
            "http_error",
        ),
        (
            _HttpResponse(_capabilities(), raw=b'{"not":"capabilities"}'),
            "invalid_response",
        ),
    ),
)
def test_http_capability_response_failures_are_typed(
    response: _HttpResponse,
    code: str,
) -> None:
    session = _HttpSession(response, response)
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="failing-reference",
        endpoint_url="https://authoring.example.test/v1",
        session=session,
    )

    with pytest.raises(GeometryAuthoringProviderError) as exc_info:
        provider.capabilities()

    assert exc_info.value.failure.code == code
    assert response.closed is True
    assert len(session.calls) == 1


def test_streamed_http_response_is_bounded_without_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "_MAX_PROVIDER_JSON_BYTES", 10)
    response = _HttpResponse(
        _capabilities(),
        chunks=(b"12345678", b"90123456"),
    )
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="oversized-reference",
        endpoint_url="https://authoring.example.test/v1",
        session=_HttpSession(response, response),
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.capabilities()

    assert exc_info.value.failure.code == "invalid_response"
    assert response.closed is True


class _TransportFailureSession:
    def post(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("sensitive provider detail")


def test_http_transport_failure_has_no_fallback_or_sensitive_detail() -> None:
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="unavailable-reference",
        endpoint_url="https://authoring.example.test/v1",
        session=_TransportFailureSession(),
    )

    with pytest.raises(GeometryAuthoringProviderError) as exc_info:
        provider.generate(_generation_request())

    assert exc_info.value.failure.code == "transport_error"
    assert exc_info.value.failure.retryable is True
    assert "no fallback" in exc_info.value.failure.summary
    assert "sensitive provider detail" not in exc_info.value.failure.summary


def test_missing_http_method_is_a_typed_transport_failure() -> None:
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="invalid-session",
        endpoint_url="https://authoring.example.test/v1",
        session=object(),
    )

    with pytest.raises(GeometryAuthoringProviderError) as exc_info:
        provider.capabilities()

    assert exc_info.value.failure.code == "transport_error"


def test_malformed_http_headers_are_an_invalid_response() -> None:
    response = _HttpResponse(_capabilities())
    response.headers = None  # type: ignore[assignment]
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="malformed-response",
        endpoint_url="https://authoring.example.test/v1",
        session=_HttpSession(response, response),
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.capabilities()

    assert exc_info.value.failure.code == "invalid_response"
    assert response.closed is True


class _MutatingSession:
    def __init__(self, path: Path, response: _HttpResponse) -> None:
        self.path = path
        self.response = response
        self.calls = 0

    def post(self, *_args: Any, **_kwargs: Any) -> _HttpResponse:
        self.calls += 1
        self.path.write_bytes(b"mutated-during-provider-call")
        return self.response


def test_http_adapter_rejects_input_drift_during_invocation(tmp_path: Path) -> None:
    image_path = tmp_path / "reference.png"
    image_path.write_bytes(b"initial-image")
    reference = GeometryAuthoringReference(
        reference_id="reference-image",
        kind="image",
        media_type="image/png",
        artifact=_binding(image_path),
    )
    request = _generation_request(references=(reference,))
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
    )
    session = _MutatingSession(
        image_path,
        _HttpResponse(_success_receipt("generate", request, source)),
    )
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="mutating-reference",
        endpoint_url="https://authoring.example.test/v1",
        artifact_roots=(tmp_path,),
        session=session,
    )

    with pytest.raises(GeometryAuthoringInputDrift) as exc_info:
        provider.generate(request)

    assert session.calls == 1
    assert exc_info.value.failure.code == "input_drift"
    assert exc_info.value.failure.drifted_artifacts == (str(image_path.resolve()),)


def test_http_output_must_stay_under_an_explicit_trusted_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    request = _generation_request()
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        filename="outside.step",
    )
    response = _HttpResponse(_success_receipt("generate", request, source))
    provider = HttpJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        endpoint_alias="untrusted-output",
        endpoint_url="https://authoring.example.test/v1",
        artifact_roots=(allowed_root,),
        session=_HttpSession(_HttpResponse(_capabilities()), response),
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure.code == "artifact_mismatch"
    assert "trusted artifact roots" in exc_info.value.failure.summary


def test_provider_cannot_introduce_unbound_provenance_inputs(tmp_path: Path) -> None:
    request = _generation_request()
    unbound_path = tmp_path / "host-like-input.txt"
    unbound_path.write_text("must not enter provenance\n", encoding="utf-8")
    source = _source_bundle(
        tmp_path,
        request_digest=geometry_authoring_request_digest(request),
        provenance_inputs=(_binding(unbound_path),),
    )
    provider = _artifact_provider(
        tmp_path,
        generate_request=request,
        generate_source=source,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.generate(request)

    assert exc_info.value.failure.code == "artifact_mismatch"
    assert "unbound input artifact" in exc_info.value.failure.summary


def test_artifact_adapter_rejects_symlinked_provider_json(tmp_path: Path) -> None:
    real_path = _write_model(tmp_path / "real-capabilities.json", _capabilities())
    symlink_path = tmp_path / "capabilities.json"
    symlink_path.symlink_to(real_path)

    with pytest.raises(ValueError, match="traverses a symlink"):
        ArtifactJsonGeometryAuthoringProvider(
            provider_id=_PROVIDER_ID,
            capabilities_path=symlink_path,
        )


def test_artifact_read_rejects_a_parent_swapped_to_a_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted"
    nested_root = trusted_root / "nested"
    nested_root.mkdir(parents=True)
    artifact = _write_model(nested_root / "capabilities.json", _capabilities())
    outside_root = tmp_path / "outside"
    outside_root.mkdir()
    _write_model(outside_root / artifact.name, _capabilities("outside-provider"))
    parked_root = trusted_root / "parked"
    real_open = provider_module.os.open
    swapped = False

    def swap_parent_before_open(
        path: str | bytes | int | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == "nested" and dir_fd is not None and not swapped:
            swapped = True
            nested_root.rename(parked_root)
            nested_root.symlink_to(outside_root, target_is_directory=True)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    supported_dir_fd = provider_module.os.supports_dir_fd
    monkeypatch.setattr(provider_module.os, "open", swap_parent_before_open)
    monkeypatch.setattr(
        provider_module.os,
        "supports_dir_fd",
        {*supported_dir_fd, swap_parent_before_open},
    )

    with pytest.raises(ValueError, match="path changed|symbolic link"):
        provider_module._read_regular_artifact(trusted_root, artifact)

    assert swapped is True
    assert (
        parked_root.joinpath(artifact.name).read_bytes()
        != outside_root.joinpath(artifact.name).read_bytes()
    )


def test_artifact_read_rejects_a_root_swapped_before_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted_root = tmp_path / "trusted"
    trusted_root.mkdir()
    artifact = _write_model(trusted_root / "capabilities.json", _capabilities())
    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    _write_model(replacement_root / artifact.name, _capabilities("replacement-provider"))
    parked_root = tmp_path / "parked"
    real_open = provider_module.os.open
    swapped = False

    def swap_root_before_open(
        path: str | bytes | int | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal swapped
        if path == trusted_root and dir_fd is None and not swapped:
            swapped = True
            trusted_root.rename(parked_root)
            replacement_root.rename(trusted_root)
        return real_open(path, flags, mode, dir_fd=dir_fd)

    supported_dir_fd = provider_module.os.supports_dir_fd
    monkeypatch.setattr(provider_module.os, "open", swap_root_before_open)
    monkeypatch.setattr(
        provider_module.os,
        "supports_dir_fd",
        {*supported_dir_fd, swap_root_before_open},
    )

    with pytest.raises(ValueError, match="artifact root changed"):
        provider_module._read_regular_artifact(trusted_root, artifact)

    assert swapped is True
    assert (
        parked_root.joinpath(artifact.name).read_bytes()
        != trusted_root.joinpath(artifact.name).read_bytes()
    )


def test_artifact_adapter_rejects_hardlinked_provider_json(tmp_path: Path) -> None:
    real_path = _write_model(tmp_path / "real-capabilities.json", _capabilities())
    hardlink_path = tmp_path / "capabilities.json"
    hardlink_path.hardlink_to(real_path)

    with pytest.raises(ValueError, match="safe regular file"):
        ArtifactJsonGeometryAuthoringProvider(
            provider_id=_PROVIDER_ID,
            capabilities_path=hardlink_path,
        )


def test_capability_provider_identity_must_match_configuration(tmp_path: Path) -> None:
    capability_path = _write_model(
        tmp_path / "capabilities.json",
        _capabilities("different-provider"),
    )
    provider = ArtifactJsonGeometryAuthoringProvider(
        provider_id=_PROVIDER_ID,
        capabilities_path=capability_path,
    )

    with pytest.raises(GeometryAuthoringInvalidResponse) as exc_info:
        provider.capabilities()

    assert exc_info.value.failure.operation == "capabilities"
    assert exc_info.value.failure.code == "invalid_response"


def test_failure_model_requires_drift_paths_only_for_input_drift() -> None:
    with pytest.raises(ValidationError, match="must identify changed artifacts"):
        GeometryAuthoringFailure(
            provider_id=_PROVIDER_ID,
            operation="generate",
            code="input_drift",
            summary="Input changed.",
        )
    with pytest.raises(ValidationError, match="only input drift"):
        GeometryAuthoringFailure(
            provider_id=_PROVIDER_ID,
            operation="generate",
            code="provider_failure",
            summary="Provider failed.",
            drifted_artifacts=("/tmp/input",),
        )
