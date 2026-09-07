# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import pytest

from content_agent_workflows.articulation import (
    ArticulationProposalAttemptFailed,
    ArticulationProposalProviderFailure,
    ArticulationProposalProviderPublication,
    ArticulationProposalProviderRequest,
    ArticulationProposalProviderRequestV2,
    ArtifactJsonArticulationProposalProvider,
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    HttpJsonArticulationProposalProvider,
    request_embedded_articulation_provider_proposal,
    validate_articulation_proposal_attempt_receipt,
    validate_embedded_articulation_provider_proposal,
)
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    DomainProposalPayload,
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    canonical_json_digest,
)

_DIGEST = "a" * 64
_IMPLEMENTATION_DIGEST = "b" * 64


def test_legacy_provider_request_v1_schema_remains_frozen() -> None:
    schema = ArticulationProposalProviderRequest.model_json_schema()

    assert canonical_json_digest(schema) == (
        "1b9dcfb7ff30bc7aa324675271d889fe9bfaa6f856fd5ceaba13daa65193d59a"
    )
    assert "provider_inputs" not in schema["properties"]


def test_legacy_provider_publication_schema_remains_frozen() -> None:
    assert (
        canonical_json_digest(
            ArticulationProposalProviderPublication.model_json_schema()
        )
        == "20ed85c64474ba37c0c248edded428eaf326ccefd6b9efdc9df399ba311a4361"
    )


def _binding(path: Path) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def _record(
    evidence_id: str,
    evidence_type: Literal[
        "inspection",
        "render",
        "validation",
        "artifact",
        "measurement",
        "critique",
        "capability",
    ],
    *,
    artifact: ExecutionArtifactBinding | None = None,
) -> ProviderNeutralEvidenceRecord:
    return ProviderNeutralEvidenceRecord(
        evidence_id=evidence_id,
        evidence_type=evidence_type,
        status="available",
        summary=f"Exact {evidence_id} evidence.",
        artifacts=(artifact,) if artifact else (),
        facts={} if artifact else {"complete": True},
    )


def _preparation(tmp_path: Path) -> tuple[EmbeddedArticulationPreparation, Path, Path]:
    evidence_path = tmp_path / "inspection.json"
    evidence_path.write_text('{"complete":true}\n', encoding="utf-8")
    evidence = _binding(evidence_path)
    preparation = EmbeddedArticulationPreparation(
        source_sha256=_DIGEST,
        source_dependency_bundle_sha256="c" * 64,
        configuration_sha256="d" * 64,
        evidence_provider=ProducerIdentity(
            producer_id="deterministic-inspection",
            role="evidence_provider",
            implementation="test-inspector",
            implementation_digest=_IMPLEMENTATION_DIGEST,
        ),
        source_hierarchy=_record(
            "source-hierarchy-inspection", "inspection", artifact=evidence
        ),
        source_members=_record("joint-source-member-inspection", "inspection"),
        authoritative_owners=_record(
            "joint-authoritative-owner-inspection", "inspection"
        ),
        capabilities=_record("joint-authoring-capabilities", "capability"),
        renders=_record("joint-render-inspection", "render"),
        scene=_record("joint-scene-inspection", "inspection"),
        proposal_status="not_evaluated",
    )
    path = tmp_path / "preparation.json"
    path.write_text(preparation.model_dump_json(indent=2), encoding="utf-8")
    return preparation, path, evidence_path


def _provider_payload(path: Path) -> DomainProposalPayload:
    payload = DomainProposalPayload(
        schema_version="replacement-provider.example.v1",
        values={"candidate_hints": [{"id": "candidate-1"}]},
    )
    path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
    return payload


def test_artifact_provider_publishes_exact_v2_proposal_without_joint_client(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    preparation, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    native_payload = _provider_payload(payload_path)

    def forbidden_joint_client(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("JointAgentLocalClient must not be constructed")

    monkeypatch.setattr(
        "content_agent_workflows.articulation.client.JointAgentLocalClient.__init__",
        forbidden_joint_client,
    )
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="replacement-artifact-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal",
        intent="Suggest advisory articulation candidates.",
        provider=provider,
    )

    assert publication.fallback_used is False
    assert publication.joint_agent_local_client_invoked is False
    assert publication.result.schema_version.endswith("provider-proposal.v2")
    assert publication.result.producer.producer_id == "replacement-artifact-provider"
    assert publication.result.capability is not None
    assert publication.result.capability.adapter_id == "artifact-json"
    assert publication.result.capability.provider_alias == "artifact-json-payload"
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    assert bound.proposal_status == "available"
    assert bound.proposal == publication.result
    assert publication.result.provider_request == publication.request
    assert publication.result.native_payload == publication.native_payload
    validate_embedded_articulation_provider_proposal(bound)
    wrapped = publication.result.payload.model_dump(mode="json")["values"]
    assert wrapped["native_payload"] == native_payload.model_dump(mode="json")
    assert preparation.proposal_status == "not_evaluated"
    terminal = validate_articulation_proposal_attempt_receipt(
        publication.terminal_receipt.path
    )
    assert terminal.disposition == "succeeded"
    assert terminal.request_payload.schema_version.endswith("provider-request.v2")
    assert terminal.provider_inputs == (_binding(payload_path),)
    assert terminal.request == publication.request
    assert terminal.provider == publication.result.producer
    assert terminal.preparation_digest == publication.result.preparation_digest
    assert Path(terminal.request.path).parent.stat().st_mode & 0o777 == 0o500


def test_bound_artifact_proposal_rechecks_original_provider_input(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal-input-provenance",
        intent="Keep original provider input live in transitive provenance.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="provider-input-provenance",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    payload_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="provider input changed after deterministic capture",
    ):
        validate_embedded_articulation_provider_proposal(bound)


def test_bound_artifact_proposal_rejects_hard_linked_provider_input(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal-hard-link-provenance",
        intent="Reject a hard-linked provider input after capture.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="provider-hard-link-provenance",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    os.link(payload_path, tmp_path / "provider-payload-alias.json")

    with pytest.raises(EmbeddedArticulationError, match="provider input is unsafe"):
        validate_embedded_articulation_provider_proposal(bound)


def test_proposal_leaf_rejects_selected_preparation_substitution(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    selected = _binding(preparation_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    preparation_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="differs from the selected-leaf invocation",
    ):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "attempt",
            intent="Do not invoke on substituted preparation bytes.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="replacement-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
            expected_preparation=selected,
        )
    assert not (tmp_path / "attempt").exists()


def test_artifact_provider_rejects_selected_payload_substitution(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    selected = _binding(payload_path)
    payload_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="differs from the selected-leaf invocation",
    ):
        ArtifactJsonArticulationProposalProvider(
            provider_id="replacement-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
            expected_payload=selected,
        )


def test_terminal_validator_reconstructs_exact_wrapped_native_payload(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal",
        intent="Reject a wrapper that no longer matches provider bytes.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="wrapper-integrity-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    root = Path(publication.terminal_receipt.path).parent
    proposal_path = Path(publication.proposal.path)
    bound_path = Path(publication.bound_preparation.path)
    terminal_path = Path(publication.terminal_receipt.path)
    root.chmod(0o700)
    for path in (proposal_path, bound_path, terminal_path):
        path.chmod(0o644)

    proposal_payload = json.loads(proposal_path.read_text(encoding="utf-8"))
    proposal_payload["payload"]["values"]["native_payload"]["values"][
        "candidate_hints"
    ][0]["id"] = "forged-candidate"
    proposal_path.write_text(json.dumps(proposal_payload, indent=2), encoding="utf-8")
    bound_payload = json.loads(bound_path.read_text(encoding="utf-8"))
    bound_payload["proposal"] = proposal_payload
    bound_path.write_text(json.dumps(bound_payload, indent=2), encoding="utf-8")
    terminal_payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    terminal_payload["proposal"] = _binding(proposal_path).model_dump(mode="json")
    terminal_payload["bound_preparation"] = _binding(bound_path).model_dump(mode="json")
    terminal_path.write_text(json.dumps(terminal_payload, indent=2), encoding="utf-8")
    for path in (proposal_path, bound_path, terminal_path):
        path.chmod(0o444)
    root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="changed after publication"):
        validate_articulation_proposal_attempt_receipt(terminal_path)


def test_terminal_validator_requires_canonical_receipt_filename(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal",
        intent="Preserve canonical receipt discovery.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="canonical-receipt-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    terminal_path = Path(publication.terminal_receipt.path)
    root = terminal_path.parent
    renamed = root / "renamed-terminal.json"
    root.chmod(0o700)
    terminal_path.rename(renamed)
    root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="non-canonical filename"):
        validate_articulation_proposal_attempt_receipt(renamed)


def test_proposal_attempt_reserves_generated_wrapper_headroom(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    preparation_limit = preparation_path.stat().st_size + 1
    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_PREPARATION_JSON_BYTES",
        preparation_limit,
    )
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "headroom-attempt",
        intent="Preserve space for generated request and result wrappers.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="headroom-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )

    assert publication.request.size_bytes > preparation_limit
    assert (
        validate_articulation_proposal_attempt_receipt(
            publication.terminal_receipt.path
        ).disposition
        == "succeeded"
    )


def test_oversized_request_is_rejected_before_provider_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_GENERATED_JSON_BYTES",
        1,
    )
    output_dir = tmp_path / "oversized-request-attempt"
    with pytest.raises(EmbeddedArticulationError, match="bounded limit"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Reject before invocation.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="bounded-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )
    assert not output_dir.exists()


def test_terminal_capacity_is_reserved_before_provider_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    calibration = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "attempt-calibrate",
        intent="Reserve a mandatory terminal before provider invocation.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="bounded-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    terminal_size = Path(calibration.terminal_receipt.path).stat().st_size
    request_limit = calibration.request.size_bytes + 1
    assert terminal_size > request_limit
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_GENERATED_JSON_BYTES",
        request_limit,
    )
    invoked = False

    class InvocationTrackingProvider(ArtifactJsonArticulationProposalProvider):
        def propose(self, request: Any) -> DomainProposalPayload:
            nonlocal invoked
            invoked = True
            return super().propose(request)

    output_dir = tmp_path / "attempt-preflight"
    with pytest.raises(EmbeddedArticulationError, match="bounded limit"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Reserve a mandatory terminal before provider invocation.",
            provider=InvocationTrackingProvider(
                provider_id="bounded-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )
    assert invoked is False
    assert not output_dir.exists()


def test_terminal_preflight_budgets_overflow_and_worst_filename_expansion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_json_document = proposal_provider_module._json_document
    observed_largest: Any = None

    def capture_largest(payload: Any) -> bytes:
        nonlocal observed_largest
        if (
            isinstance(
                payload,
                proposal_provider_module.ArticulationProposalAttemptTerminalReceipt,
            )
            and len(payload.partial_outputs)
            == proposal_provider_module._MAX_ARTICULATION_PARTIAL_OUTPUTS
        ):
            observed_largest = payload
        return original_json_document(payload)

    monkeypatch.setattr(
        proposal_provider_module,
        "_json_document",
        capture_largest,
    )
    request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "preflight-budget",
        intent="Budget every bounded terminal expansion before invocation.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="preflight-budget-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )

    assert observed_largest is not None
    assert "excess outputs were discarded" in observed_largest.failure_summary
    assert all(
        len(os.fsencode(Path(item.path).name)) == 255
        for item in observed_largest.partial_outputs
    )
    assert (
        max(
            len(json.dumps(Path(item.path).name, ensure_ascii=True))
            for item in observed_largest.partial_outputs
        )
        >= 1_530
    )


@pytest.mark.parametrize("artifact_name", ("request", "native_payload"))
def test_published_proposal_rejects_transitive_provenance_substitution(
    tmp_path: Path,
    artifact_name: str,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "proposal",
        intent="Suggest advisory articulation candidates.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="replacement-artifact-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    binding = getattr(publication, artifact_name)
    Path(binding.path).chmod(0o644)
    Path(binding.path).write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="changed after deterministic capture",
    ):
        validate_embedded_articulation_provider_proposal(bound)


class _HttpResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: DomainProposalPayload) -> None:
        self._payload = payload

    def iter_content(self, *, chunk_size: int) -> Any:
        assert chunk_size == 64 * 1_024
        yield self._payload.model_dump_json().encode("utf-8")

    def close(self) -> None:
        pass


class _HttpSession:
    def __init__(self, payload: DomainProposalPayload) -> None:
        self.payload = payload
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any) -> _HttpResponse:
        self.calls.append({"url": url, **kwargs})
        return _HttpResponse(self.payload)


def test_http_provider_receives_exact_typed_request_and_no_redirect_fallback(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload = DomainProposalPayload(
        schema_version="replacement-http.example.v1",
        values={"candidate_hints": [{"id": "candidate-2"}]},
    )
    session = _HttpSession(payload)
    provider = HttpJsonArticulationProposalProvider(
        provider_id="replacement-http-provider",
        capability_id="articulation-proposal-v1",
        endpoint_alias="staging-replacement",
        endpoint_url="https://proposal.example.test/v1/articulation",
        timeout_seconds=9.0,
        session=session,
    )

    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "http-proposal",
        intent="Suggest candidates from exact evidence.",
        provider=provider,
    )

    assert publication.result.capability is not None
    assert publication.result.capability.adapter_id == "http-json"
    assert publication.result.capability.provider_alias == "staging-replacement"
    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["allow_redirects"] is False
    assert call["stream"] is True
    assert call["timeout"] == 9.0
    assert call["json"]["schema_version"].endswith("provider-request.v1")
    assert "provider_inputs" not in call["json"]
    assert call["json"]["preparation_payload"]["proposal_status"] == "not_evaluated"
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    validate_embedded_articulation_provider_proposal(bound)


@pytest.mark.parametrize("adapter_id", ("artifact-json", "http-json"))
def test_bound_proposal_rechecks_original_preparation(
    tmp_path: Path,
    adapter_id: str,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    provider: (
        ArtifactJsonArticulationProposalProvider | HttpJsonArticulationProposalProvider
    )
    if adapter_id == "artifact-json":
        payload_path = tmp_path / "provider-payload.json"
        _provider_payload(payload_path)
        provider = ArtifactJsonArticulationProposalProvider(
            provider_id="original-preparation-artifact-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        )
    else:
        provider = HttpJsonArticulationProposalProvider(
            provider_id="original-preparation-http-provider",
            capability_id="articulation-proposal-v1",
            endpoint_alias="original-preparation",
            endpoint_url="https://proposal.example.test/v1/articulation",
            session=_HttpSession(
                DomainProposalPayload(
                    schema_version="replacement-http.example.v1",
                    values={"candidate_hints": [{"id": "candidate-2"}]},
                )
            ),
        )
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / f"{adapter_id}-preparation-provenance",
        intent="Keep the original preparation live in transitive provenance.",
        provider=provider,
    )
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    preparation_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(
        EmbeddedArticulationError,
        match="Replacement proposal request preparation is malformed",
    ):
        validate_embedded_articulation_provider_proposal(bound)


def test_bound_proposal_rejects_unsupported_request_schema(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "unsupported-request-schema",
        intent="Reject an unversioned provider-request substitution.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="unsupported-request-schema-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    bound = EmbeddedArticulationPreparation.model_validate_json(
        Path(publication.bound_preparation.path).read_bytes()
    )
    assert bound.proposal is not None
    request_document = json.loads(Path(publication.request.path).read_text())
    request_document["schema_version"] = "unsupported-provider-request.v0"
    unsupported_path = tmp_path / "unsupported-provider-request.json"
    unsupported_path.write_text(json.dumps(request_document), encoding="utf-8")
    unsupported_proposal = bound.proposal.model_copy(
        update={"provider_request": _binding(unsupported_path)}
    )

    with pytest.raises(
        EmbeddedArticulationError,
        match="request schema is malformed or unsupported",
    ):
        validate_embedded_articulation_provider_proposal(
            bound.model_copy(update={"proposal": unsupported_proposal})
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"endpoint_alias": "   "}, "alias must not be empty"),
        (
            {"endpoint_url": "https://user:secret@proposal.example.test/v1"},
            "credential-free HTTP URL",
        ),
        (
            {"endpoint_url": "http://proposal.example.test/v1"},
            "require HTTPS transport",
        ),
        ({"timeout_seconds": 0.0}, "timeout must be positive"),
    ),
)
def test_http_provider_rejects_unsafe_operator_configuration(
    overrides: dict[str, Any],
    message: str,
) -> None:
    arguments: dict[str, Any] = {
        "provider_id": "operator-provider",
        "capability_id": "articulation-proposal-v1",
        "endpoint_alias": "operator-endpoint",
        "endpoint_url": "https://proposal.example.test/v1/articulation",
        "timeout_seconds": 9.0,
    }
    arguments.update(overrides)
    with pytest.raises(ValueError, match=message):
        HttpJsonArticulationProposalProvider(**arguments)


class _OversizedResponse:
    status_code = 200
    headers = {"Content-Length": str(16 * 1_024 * 1_024 + 1)}

    def iter_content(self, *, chunk_size: int) -> Any:
        raise AssertionError(f"oversized response must not be read: {chunk_size}")

    def close(self) -> None:
        pass


class _OversizedSession:
    def post(self, *_args: Any, **_kwargs: Any) -> _OversizedResponse:
        return _OversizedResponse()


def test_oversized_http_response_is_invalid_response_not_publication_failure(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "oversized-http-attempt",
            intent="Reject an oversized provider response before reading it.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="oversized-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="oversized",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_OversizedSession(),
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "invalid_response"
    assert terminal.failure_code == "invalid_typed_response"


class _StreamingOversizedResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def iter_content(self, *, chunk_size: int) -> Any:
        assert chunk_size == 64 * 1_024
        yield b"12345678"
        yield b"90123456"

    def close(self) -> None:
        pass


class _StreamingOversizedSession:
    def post(self, *_args: Any, **_kwargs: Any) -> _StreamingOversizedResponse:
        return _StreamingOversizedResponse()


def test_streaming_http_response_without_length_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_INPUT_JSON_BYTES",
        10,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "streaming-oversized-attempt",
            intent="Bound a streamed provider body without Content-Length.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="streaming-oversized-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="streaming-oversized",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_StreamingOversizedSession(),
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "invalid_response"


def test_oversized_artifact_response_is_invalid_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "oversized-artifact.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="oversized-artifact-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_INPUT_JSON_BYTES",
        payload_path.stat().st_size - 1,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "oversized-artifact-attempt",
            intent="Reject an oversized exact artifact response.",
            provider=provider,
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "invalid_response"
    assert terminal.failure_code == "invalid_typed_response"


class _FailingResponse:
    status_code = 503

    def close(self) -> None:
        pass


class _FailingSession:
    def post(self, *_args: Any, **_kwargs: Any) -> _FailingResponse:
        return _FailingResponse()


def test_failed_selected_provider_does_not_fallback_or_publish_proposal(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    provider = HttpJsonArticulationProposalProvider(
        provider_id="unavailable-provider",
        capability_id="articulation-proposal-v1",
        endpoint_alias="unavailable",
        endpoint_url="https://proposal.example.test/v1/articulation",
        session=_FailingSession(),
    )
    output_dir = tmp_path / "failed-provider"

    with pytest.raises(
        ArticulationProposalAttemptFailed,
        match="without fallback",
    ) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Suggest candidates.",
            provider=provider,
        )

    assert (output_dir / "articulation_proposal_provider_request.json").is_file()
    assert (output_dir / "articulation_proposal_attempt_terminal.json").is_file()
    assert not (output_dir / "embedded_articulation_provider_proposal.json").exists()
    assert not (output_dir / "embedded_articulation_preparation.json").exists()
    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "provider_failure"
    assert terminal.failure_code == "provider_runtime_failure"
    assert terminal.fallback_used is False


def test_local_interrupt_terminal_is_not_attributed_to_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="interrupt-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )

    def interrupt(_request: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(provider, "propose", interrupt)
    output_dir = tmp_path / "interrupted-attempt"
    with pytest.raises(KeyboardInterrupt):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Preserve a locally interrupted attempt.",
            provider=provider,
        )
    terminal = validate_articulation_proposal_attempt_receipt(
        output_dir / "articulation_proposal_attempt_terminal.json"
    )
    assert terminal.disposition == "publication_failure"
    assert terminal.failure_code == "attempt_interrupted"


def test_proposal_leaf_rejects_existing_attempt_output_as_typed_error(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    output_dir = tmp_path / "existing-attempt"
    output_dir.mkdir()

    with pytest.raises(
        EmbeddedArticulationError,
        match="each provider attempt requires a fresh output directory",
    ):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Suggest candidates.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="replacement-artifact-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )


def test_proposal_leaf_rejects_not_requested_and_already_available_modes(
    tmp_path: Path,
) -> None:
    preparation, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="replacement-artifact-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    not_requested = preparation.model_copy(update={"proposal_status": "not_requested"})
    preparation_path.write_text(
        not_requested.model_dump_json(indent=2), encoding="utf-8"
    )
    with pytest.raises(EmbeddedArticulationError, match="not_evaluated"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "not-requested",
            intent="Suggest candidates.",
            provider=provider,
        )

    selected_root = tmp_path / "selected"
    selected_root.mkdir()
    _, selected_preparation_path, _ = _preparation(selected_root)
    publication = request_embedded_articulation_provider_proposal(
        selected_preparation_path,
        output_dir=tmp_path / "first-proposal",
        intent="Suggest candidates.",
        provider=provider,
    )
    with pytest.raises(EmbeddedArticulationError, match="not_evaluated"):
        request_embedded_articulation_provider_proposal(
            publication.bound_preparation.path,
            output_dir=tmp_path / "second-proposal",
            intent="Suggest candidates again.",
            provider=provider,
        )


class _MutatingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, *, evidence_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.evidence_path = evidence_path

    def propose(
        self, request: Any
    ) -> DomainProposalPayload:  # pragma: no cover - called by the leaf
        payload = super().propose(request)
        self.evidence_path.write_text('{"complete":false}\n', encoding="utf-8")
        return payload


class _RequestMutatingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, *, request_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.request_path = request_path

    def propose(
        self, request: Any
    ) -> DomainProposalPayload:  # pragma: no cover - called by the leaf
        payload = super().propose(request)
        self.request_path.chmod(0o644)
        self.request_path.write_text('{"substituted":true}\n', encoding="utf-8")
        self.request_path.chmod(0o444)
        return payload


class _RequestDeletingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, *, request_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.request_path = request_path

    def propose(self, request: Any) -> DomainProposalPayload:
        payload = super().propose(request)
        self.request_path.unlink()
        return payload


class _PriorTerminalMutatingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, *, prior_terminal_path: Path, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.prior_terminal_path = prior_terminal_path

    def propose(
        self, request: Any
    ) -> DomainProposalPayload:  # pragma: no cover - called by the leaf
        payload = super().propose(request)
        root = self.prior_terminal_path.parent
        root.chmod(0o700)
        self.prior_terminal_path.chmod(0o644)
        self.prior_terminal_path.write_text(
            self.prior_terminal_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        self.prior_terminal_path.chmod(0o444)
        root.chmod(0o500)
        return payload


class _ProviderInputIdentityMutatingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, **kwargs: Any) -> None:
        self._omit_input = False
        super().__init__(**kwargs)

    @property
    def input_artifacts(self) -> tuple[ExecutionArtifactBinding, ...]:
        return () if self._omit_input else super().input_artifacts

    def propose(self, request: Any) -> DomainProposalPayload:
        payload = super().propose(request)
        self._omit_input = True
        return payload


class _ProviderInputSubstitutingProvider(ArtifactJsonArticulationProposalProvider):
    def __init__(self, *, substituted_input: Path, **kwargs: Any) -> None:
        self._substituted_input = substituted_input
        self._substitute_input = False
        super().__init__(**kwargs)

    @property
    def input_artifacts(self) -> tuple[ExecutionArtifactBinding, ...]:
        if self._substitute_input:
            return (_binding(self._substituted_input),)
        return super().input_artifacts

    def propose(self, request: Any) -> DomainProposalPayload:
        payload = super().propose(request)
        self._substitute_input = True
        return payload


def test_proposal_leaf_rejects_evidence_mutation_during_provider_call(
    tmp_path: Path,
) -> None:
    _, preparation_path, evidence_path = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = _MutatingProvider(
        evidence_path=evidence_path,
        provider_id="mutating-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )

    with pytest.raises(
        ArticulationProposalAttemptFailed,
        match="changed after capture",
    ) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "mutated-evidence",
            intent="Suggest candidates.",
            provider=provider,
        )
    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert str(evidence_path.resolve()) in terminal.drifted_inputs
    preparation_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EmbeddedArticulationError, match="changed after capture"):
        validate_articulation_proposal_attempt_receipt(
            exc_info.value.receipt_binding.path
        )


def test_proposal_leaf_terminalizes_provider_input_identity_drift(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "mutated-provider-input-identity",
            intent="Terminalize provider input identity mutation.",
            provider=_ProviderInputIdentityMutatingProvider(
                provider_id="provider-input-mutating-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert isinstance(terminal.request_payload, ArticulationProposalProviderRequestV2)
    assert terminal.provider_inputs == terminal.request_payload.provider_inputs
    assert terminal.drifted_inputs == (
        str(payload_path.resolve()),
        "provider-input-identity",
    )


def test_proposal_leaf_terminalizes_substituted_provider_input_identity(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    substituted_path = tmp_path / "substituted-provider-payload.json"
    _provider_payload(substituted_path)

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "substituted-provider-input-identity",
            intent="Terminalize provider input identity substitution.",
            provider=_ProviderInputSubstitutingProvider(
                substituted_input=substituted_path,
                provider_id="provider-input-substituting-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert isinstance(terminal.request_payload, ArticulationProposalProviderRequestV2)
    assert terminal.provider_inputs == terminal.request_payload.provider_inputs
    assert terminal.drifted_inputs == (
        str(payload_path.resolve()),
        "provider-input-identity",
    )
    assert str(substituted_path.resolve()) not in terminal.drifted_inputs


def test_proposal_leaf_terminalizes_exact_request_drift(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    output_dir = tmp_path / "mutated-request"
    request_path = output_dir / "articulation_proposal_provider_request.json"

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Terminalize request mutation.",
            provider=_RequestMutatingProvider(
                request_path=request_path,
                provider_id="request-mutating-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.drifted_inputs == (str(request_path.resolve()),)
    assert terminal.request_payload.intent == "Terminalize request mutation."
    assert not request_path.exists()


def test_proposal_leaf_terminalizes_deleted_request_drift(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    output_dir = tmp_path / "deleted-request"
    request_path = output_dir / "articulation_proposal_provider_request.json"

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Terminalize request deletion.",
            provider=_RequestDeletingProvider(
                request_path=request_path,
                provider_id="request-deleting-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.drifted_inputs == (str(request_path.resolve()),)
    assert not request_path.exists()


class _InvalidResponse:
    status_code = 200
    headers: dict[str, str] = {}

    def iter_content(self, *, chunk_size: int) -> Any:
        del chunk_size
        yield b'{"schema_version":"invalid-without-values.v1"}'

    def close(self) -> None:
        pass


class _InvalidSession:
    def post(self, *_args: Any, **_kwargs: Any) -> _InvalidResponse:
        return _InvalidResponse()


class _RuntimeFailureSession:
    def post(self, *_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("provider details must not enter the terminal receipt")


@pytest.mark.parametrize(
    ("session", "disposition"),
    [
        (_InvalidSession(), "invalid_response"),
        (_RuntimeFailureSession(), "provider_failure"),
    ],
)
def test_every_invalid_or_runtime_attempt_publishes_typed_terminal(
    tmp_path: Path,
    session: Any,
    disposition: str,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    output_dir = tmp_path / disposition

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Suggest candidates.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id=f"{disposition}-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias=disposition,
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=session,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == disposition
    assert terminal.request.path == str(
        (output_dir / "articulation_proposal_provider_request.json").resolve()
    )
    assert terminal.preparation.path == str(preparation_path.resolve())
    assert terminal.provider.producer_id == f"{disposition}-provider"
    assert "provider details" not in (terminal.failure_summary or "")


def test_terminal_validator_requires_proposal_provider_role(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    output_dir = tmp_path / "wrong-role"
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=output_dir,
            intent="Produce a failed receipt for tamper validation.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="role-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="role-test",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_RuntimeFailureSession(),
            ),
        )
    terminal_path = Path(exc_info.value.receipt_binding.path)
    output_dir.chmod(0o700)
    terminal_path.chmod(0o644)
    payload = json.loads(terminal_path.read_text(encoding="utf-8"))
    payload["provider"]["role"] = "evidence_provider"
    terminal_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    terminal_path.chmod(0o444)
    output_dir.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="terminal identity"):
        validate_articulation_proposal_attempt_receipt(terminal_path)


def test_replacement_attempt_binds_exact_prior_terminal_without_reusing_root(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    first_root = tmp_path / "failed-attempt"
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=first_root,
            intent="Suggest candidates.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="failed-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="failed",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    replacement = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "replacement-attempt",
        intent="Suggest candidates after explicit replacement.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="replacement-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
        replaces_terminal_receipt=first_error.value.receipt_binding.path,
        replacement_reason="Provider availability changed; preserve the failed root.",
    )

    terminal = validate_articulation_proposal_attempt_receipt(
        replacement.terminal_receipt.path
    )
    assert terminal.replacement is not None
    assert terminal.replacement.prior_attempt_id == first_error.value.receipt.attempt_id
    assert Path(terminal.replacement.prior_terminal_receipt.path).parent == first_root
    assert Path(terminal.request.path).parent != first_root

    prior_path = Path(first_error.value.receipt_binding.path)
    prior_path.chmod(0o644)
    prior_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(EmbeddedArticulationError, match="changed after capture"):
        validate_articulation_proposal_attempt_receipt(
            replacement.terminal_receipt.path
        )


def test_replacement_lineage_drift_is_a_verifiable_input_drift_terminal(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    first_root = tmp_path / "failed-attempt"
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=first_root,
            intent="Create an exact failed attempt to replace.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="failed-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="failed",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    prior_terminal = Path(first_error.value.receipt_binding.path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)

    with pytest.raises(ArticulationProposalAttemptFailed) as replacement_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "replacement-attempt",
            intent="Terminalize exact prior-lineage drift.",
            provider=_PriorTerminalMutatingProvider(
                prior_terminal_path=prior_terminal,
                provider_id="replacement-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
            replaces_terminal_receipt=prior_terminal,
            replacement_reason="Provider availability changed.",
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        replacement_error.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.drifted_inputs == (str(prior_terminal.resolve()),)
    assert terminal.replacement is not None
    assert terminal.replacement.prior_attempt_id == first_error.value.receipt.attempt_id


def test_ancestor_replacement_drift_is_a_verifiable_input_drift_terminal(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "first-failed-attempt",
            intent="Create the oldest exact failed attempt.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="failed-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="failed",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    first_terminal = Path(first_error.value.receipt_binding.path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    second = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "second-attempt",
        intent="Create an exact intermediate replacement.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="second-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
        replaces_terminal_receipt=first_terminal,
        replacement_reason="Replace the failed first provider.",
    )

    with pytest.raises(ArticulationProposalAttemptFailed) as third_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "third-attempt",
            intent="Terminalize drift of the oldest bound ancestor.",
            provider=_PriorTerminalMutatingProvider(
                prior_terminal_path=first_terminal,
                provider_id="third-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
            replaces_terminal_receipt=second.terminal_receipt.path,
            replacement_reason="Exercise complete replacement lineage binding.",
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        third_error.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.drifted_inputs == (str(first_terminal.resolve()),)
    assert terminal.replacement is not None
    assert terminal.replacement.prior_terminal_lineage == (
        second.terminal_receipt,
        first_error.value.receipt_binding,
    )
    second_terminal = Path(second.terminal_receipt.path)
    second_root = second_terminal.parent
    second_root.chmod(0o700)
    second_terminal.chmod(0o644)
    second_terminal.write_text(
        second_terminal.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    second_terminal.chmod(0o444)
    second_root.chmod(0o500)
    with pytest.raises(EmbeddedArticulationError, match="changed after capture"):
        validate_articulation_proposal_attempt_receipt(
            third_error.value.receipt_binding.path
        )


def test_drifted_replacement_rejects_fabricated_lineage_tail(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "first-failed-attempt",
            intent="Create the exact oldest attempt.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="first-failed-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="first-failed",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    second = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "second-attempt",
        intent="Create the exact immediate replacement.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="second-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
        replaces_terminal_receipt=first_error.value.receipt_binding.path,
        replacement_reason="Replace the first failed attempt.",
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as unrelated_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "unrelated-failed-attempt",
            intent="Create an unrelated valid terminal.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="unrelated-failed-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="unrelated-failed",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    second_terminal = Path(second.terminal_receipt.path)
    with pytest.raises(ArticulationProposalAttemptFailed) as third_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "third-attempt",
            intent="Terminalize drift of the immediate prior receipt.",
            provider=_PriorTerminalMutatingProvider(
                prior_terminal_path=second_terminal,
                provider_id="third-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
            replaces_terminal_receipt=second_terminal,
            replacement_reason="Exercise drifted lineage validation.",
        )
    terminal = third_error.value.receipt
    assert terminal.replacement is not None
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    forged_replacement = terminal.replacement.model_copy(
        update={
            "prior_terminal_lineage": (
                *terminal.replacement.prior_terminal_lineage,
                unrelated_error.value.receipt_binding,
            )
        }
    )
    forged = terminal.model_copy(
        update={
            "replacement": forged_replacement,
            "attempt_id": proposal_provider_module._attempt_id(
                terminal.request,
                forged_replacement,
            ),
        }
    )
    terminal_path = Path(third_error.value.receipt_binding.path)
    terminal_root = terminal_path.parent
    terminal_root.chmod(0o700)
    terminal_path.chmod(0o644)
    terminal_path.write_text(forged.model_dump_json(indent=2), encoding="utf-8")
    terminal_path.chmod(0o444)
    terminal_root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="non-drifted ancestor"):
        validate_articulation_proposal_attempt_receipt(terminal_path)

    over_depth_lineage = tuple(
        terminal.replacement.prior_terminal_receipt.model_copy(
            update={"path": str(tmp_path / f"missing-terminal-{index:03d}.json")}
        )
        for index in range(proposal_provider_module._MAX_REPLACEMENT_LINEAGE_DEPTH + 1)
    )
    over_depth_replacement = terminal.replacement.model_copy(
        update={
            "prior_terminal_receipt": over_depth_lineage[0],
            "prior_terminal_lineage": over_depth_lineage,
        }
    )
    over_depth = terminal.model_copy(
        update={
            "replacement": over_depth_replacement,
            "drifted_inputs": tuple(item.path for item in over_depth_lineage),
            "attempt_id": proposal_provider_module._attempt_id(
                terminal.request,
                over_depth_replacement,
            ),
        }
    )
    terminal_root.chmod(0o700)
    terminal_path.chmod(0o644)
    terminal_path.write_text(over_depth.model_dump_json(indent=2), encoding="utf-8")
    terminal_path.chmod(0o444)
    terminal_root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="exceeds its bounded depth"):
        validate_articulation_proposal_attempt_receipt(terminal_path)

    cyclic_replacement = terminal.replacement.model_copy(
        update={
            "prior_terminal_lineage": (
                *terminal.replacement.prior_terminal_lineage,
                third_error.value.receipt_binding,
            )
        }
    )
    cyclic = terminal.model_copy(
        update={
            "replacement": cyclic_replacement,
            "drifted_inputs": tuple(
                item.path for item in cyclic_replacement.prior_terminal_lineage
            ),
            "attempt_id": proposal_provider_module._attempt_id(
                terminal.request,
                cyclic_replacement,
            ),
        }
    )
    terminal_root.chmod(0o700)
    terminal_path.chmod(0o644)
    terminal_path.write_text(cyclic.model_dump_json(indent=2), encoding="utf-8")
    terminal_path.chmod(0o444)
    terminal_root.chmod(0o500)

    with pytest.raises(EmbeddedArticulationError, match="contain a cycle"):
        validate_articulation_proposal_attempt_receipt(terminal_path)


def test_replacement_depth_is_rejected_before_provider_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(proposal_provider_module, "_MAX_REPLACEMENT_LINEAGE_DEPTH", 2)
    with pytest.raises(ArticulationProposalAttemptFailed) as first_error:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "first-depth-attempt",
            intent="Create the first bounded attempt.",
            provider=HttpJsonArticulationProposalProvider(
                provider_id="failed-depth-provider",
                capability_id="articulation-proposal-v1",
                endpoint_alias="failed-depth",
                endpoint_url="https://proposal.example.test/v1/articulation",
                session=_FailingSession(),
            ),
        )
    second = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "second-depth-attempt",
        intent="Create the last valid bounded attempt.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="second-depth-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
        replaces_terminal_receipt=first_error.value.receipt_binding.path,
        replacement_reason="Create a two-terminal lineage.",
    )
    rejected_root = tmp_path / "rejected-depth-attempt"

    with pytest.raises(EmbeddedArticulationError, match="cannot accept another"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=rejected_root,
            intent="Do not invoke beyond the lineage bound.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="rejected-depth-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
            replaces_terminal_receipt=second.terminal_receipt.path,
            replacement_reason="This attempt must be rejected before invocation.",
        )
    assert not rejected_root.exists()


def test_attempt_rejects_hard_linked_input_and_symlinked_destination(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    os.link(preparation_path, tmp_path / "preparation-alias.json")
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="replacement-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    with pytest.raises(EmbeddedArticulationError, match="not a regular file"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "hard-link-rejected",
            intent="Suggest candidates.",
            provider=provider,
        )

    clean_root = tmp_path / "clean"
    clean_root.mkdir()
    _, clean_preparation, _ = _preparation(clean_root)
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    symlink_parent = tmp_path / "symlink-parent"
    symlink_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(EmbeddedArticulationError, match="traverses a symlink"):
        request_embedded_articulation_provider_proposal(
            clean_preparation,
            output_dir=symlink_parent / "attempt",
            intent="Suggest candidates.",
            provider=provider,
        )


def test_terminal_validation_rejects_unbound_attempt_root_mutation(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "attempt",
        intent="Suggest candidates.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="replacement-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    (tmp_path / "attempt").chmod(0o700)
    (tmp_path / "attempt" / "unbound.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(EmbeddedArticulationError, match="unbound or mutable"):
        validate_articulation_proposal_attempt_receipt(
            publication.terminal_receipt.path
        )


def test_post_output_failure_binds_partial_artifacts_in_verifiable_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_write = proposal_provider_module._write_json_create_only
    failed = False

    def fail_bound_preparation_once(
        root: Path,
        name: str,
        payload: Any,
    ) -> Path:
        nonlocal failed
        if name == "embedded_articulation_preparation.json" and not failed:
            failed = True
            raise OSError("simulated bound-preparation storage failure")
        return original_write(root, name, payload)

    monkeypatch.setattr(
        proposal_provider_module,
        "_write_json_create_only",
        fail_bound_preparation_once,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "partial-attempt",
            intent="Suggest candidates.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="partial-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "embedded_articulation_provider_proposal.json",
    )


def test_partial_directory_cleanup_handles_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    attempt_root = tmp_path / "partial-directory-attempt"
    attempt_root.mkdir()
    partial_directory = attempt_root / "provider-partial"
    partial_directory.mkdir()
    original_unlink = proposal_provider_module.os.unlink
    failed_once = False

    def portable_unlink(path: Any, *args: Any, **kwargs: Any) -> None:
        nonlocal failed_once
        if path == partial_directory.name and not failed_once:
            failed_once = True
            raise PermissionError("simulated BSD directory unlink result")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(proposal_provider_module.os, "unlink", portable_unlink)
    proposal_provider_module._discard_unpublishable_partial(
        attempt_root,
        partial_directory.name,
    )
    assert failed_once is True
    assert not partial_directory.exists()


def test_transient_partial_read_failure_preserves_and_binds_regular_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_write = proposal_provider_module._write_json_create_only
    original_binding = proposal_provider_module._published_binding
    failed_write = False
    proposal_reads = 0

    def fail_bound_preparation_once(
        root: Path,
        name: str,
        payload: Any,
    ) -> Path:
        nonlocal failed_write
        if name == "embedded_articulation_preparation.json" and not failed_write:
            failed_write = True
            raise OSError("simulated bound-preparation storage failure")
        return original_write(root, name, payload)

    def fail_first_partial_proposal_read(
        root: Path,
        name: str,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> ExecutionArtifactBinding:
        nonlocal proposal_reads
        if name == "embedded_articulation_provider_proposal.json":
            proposal_reads += 1
            if proposal_reads == 2:
                raise EmbeddedArticulationError("simulated transient read failure")
        return original_binding(root, name, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(
        proposal_provider_module,
        "_write_json_create_only",
        fail_bound_preparation_once,
    )
    monkeypatch.setattr(
        proposal_provider_module,
        "_published_binding",
        fail_first_partial_proposal_read,
    )

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "transient-partial-attempt",
            intent="Preserve regular partial evidence across one read failure.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="transient-partial-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "embedded_articulation_provider_proposal.json",
    )


def test_persistent_partial_read_failure_discards_output_and_terminalizes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_write = proposal_provider_module._write_json_create_only
    original_binding = proposal_provider_module._published_binding
    failed_write = False
    proposal_reads = 0

    def fail_bound_preparation_once(
        root: Path,
        name: str,
        payload: Any,
    ) -> Path:
        nonlocal failed_write
        if name == "embedded_articulation_preparation.json" and not failed_write:
            failed_write = True
            raise OSError("simulated bound-preparation storage failure")
        return original_write(root, name, payload)

    def fail_partial_proposal_reads(
        root: Path,
        name: str,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> ExecutionArtifactBinding:
        nonlocal proposal_reads
        if name == "embedded_articulation_provider_proposal.json":
            proposal_reads += 1
            if proposal_reads >= 2:
                raise EmbeddedArticulationError("persistent partial read failure")
        return original_binding(root, name, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(
        proposal_provider_module,
        "_write_json_create_only",
        fail_bound_preparation_once,
    )
    monkeypatch.setattr(
        proposal_provider_module,
        "_published_binding",
        fail_partial_proposal_reads,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "persistent-partial-attempt",
            intent="Terminalize despite persistent partial read failure.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="persistent-partial-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert terminal.partial_outputs == ()
    assert terminal.partial_outputs_discarded
    assert not (
        Path(exc_info.value.receipt_binding.path).parent
        / "embedded_articulation_provider_proposal.json"
    ).exists()


def test_provider_extra_regular_output_is_bound_before_terminalization(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    attempt_root = tmp_path / "extra-output-attempt"

    class ExtraOutputFailingProvider(ArtifactJsonArticulationProposalProvider):
        def propose(self, request: Any) -> DomainProposalPayload:
            del request
            (attempt_root / "provider-extra.json").write_text(
                '{"provider":"diagnostic"}\n', encoding="utf-8"
            )
            raise ArticulationProposalProviderFailure(
                "provider failed after writing a diagnostic"
            )

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=attempt_root,
            intent="Bind every provider-created regular output.",
            provider=ExtraOutputFailingProvider(
                provider_id="extra-output-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "provider_failure"
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "provider-extra.json",
    )
    assert not terminal.partial_outputs_discarded
    assert (attempt_root / "provider-extra.json").stat().st_mode & 0o777 == 0o444


def test_excess_provider_outputs_preserve_bounded_terminal_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    attempt_root = tmp_path / "excess-output-attempt"
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_PARTIAL_OUTPUTS",
        1,
    )

    class ExcessOutputFailingProvider(ArtifactJsonArticulationProposalProvider):
        def propose(self, request: Any) -> DomainProposalPayload:
            del request
            for index in range(256):
                (attempt_root / f"provider-extra-{index:04d}.json").write_text(
                    f'{{"provider":{index}}}\n', encoding="utf-8"
                )
            raise ArticulationProposalProviderFailure(
                "provider exceeded the bounded forensic inventory"
            )

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=attempt_root,
            intent="Retain a mandatory bounded terminal.",
            provider=ExcessOutputFailingProvider(
                provider_id="excess-output-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "provider_failure"
    assert terminal.failure_code == "provider_runtime_failure"
    assert terminal.partial_outputs_discarded
    assert "bounded terminal inventory" in (terminal.failure_summary or "")
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "provider-extra-0000.json",
    )
    assert not any(
        (attempt_root / f"provider-extra-{index:04d}.json").exists()
        for index in range(1, 256)
    )


def test_input_drift_remains_authoritative_when_partial_inventory_overflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    attempt_root = tmp_path / "drift-and-overflow-attempt"
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    monkeypatch.setattr(
        proposal_provider_module,
        "_MAX_ARTICULATION_PARTIAL_OUTPUTS",
        1,
    )

    class DriftAndOverflowProvider(ArtifactJsonArticulationProposalProvider):
        def propose(self, request: Any) -> DomainProposalPayload:
            del request
            payload_path.write_text('{"changed":true}\n', encoding="utf-8")
            (attempt_root / "a-provider-extra.json").write_text(
                '{"provider":"first"}\n', encoding="utf-8"
            )
            (attempt_root / "b-provider-extra.json").write_text(
                '{"provider":"second"}\n', encoding="utf-8"
            )
            raise ArticulationProposalProviderFailure(
                "provider drifted its bound input and overflowed outputs"
            )

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=attempt_root,
            intent="Preserve input drift as the authoritative terminal disposition.",
            provider=DriftAndOverflowProvider(
                provider_id="drift-overflow-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.failure_code == "bound_input_drift"
    assert terminal.partial_outputs_discarded
    assert str(payload_path.resolve()) in terminal.drifted_inputs
    assert "bounded terminal inventory" in (terminal.failure_summary or "")


def test_failed_attempt_self_validates_before_raising_typed_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "invalid-provider-payload.json"
    payload_path.write_text('{"not":"a domain proposal"}\n', encoding="utf-8")
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_validate = (
        proposal_provider_module.validate_articulation_proposal_attempt_receipt
    )
    validation_calls = 0

    def observe_validation(path: str | Path) -> Any:
        nonlocal validation_calls
        validation_calls += 1
        return original_validate(path)

    monkeypatch.setattr(
        proposal_provider_module,
        "validate_articulation_proposal_attempt_receipt",
        observe_validation,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "self-validated-failure",
            intent="Require a self-validated terminal.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="self-validation-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    assert validation_calls == 1
    assert exc_info.value.receipt.disposition == "invalid_response"


def test_terminal_validator_rechecks_non_drifted_inputs_at_final_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, evidence_path = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    publication = request_embedded_articulation_provider_proposal(
        preparation_path,
        output_dir=tmp_path / "final-recheck-attempt",
        intent="Recheck every non-drifted input before acceptance.",
        provider=ArtifactJsonArticulationProposalProvider(
            provider_id="final-recheck-provider",
            capability_id="articulation-proposal-v1",
            payload_path=payload_path,
        ),
    )
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_verify = proposal_provider_module._verify_binding
    mutated = False

    def mutate_preparation_after_evidence(
        binding: ExecutionArtifactBinding,
        *,
        label: str,
    ) -> None:
        nonlocal mutated
        original_verify(binding, label=label)
        if binding.path == str(evidence_path.resolve()) and not mutated:
            mutated = True
            preparation_path.write_text('{"substituted":true}\n', encoding="utf-8")

    monkeypatch.setattr(
        proposal_provider_module,
        "_verify_binding",
        mutate_preparation_after_evidence,
    )
    with pytest.raises(EmbeddedArticulationError, match="changed after capture"):
        validate_articulation_proposal_attempt_receipt(
            publication.terminal_receipt.path
        )


def test_unsafe_substituted_partial_is_discarded_before_terminalization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_write = proposal_provider_module._write_json_create_only
    failed = False
    outside = tmp_path / "outside.json"
    outside.write_text('{"outside": true}\n', encoding="utf-8")

    def substitute_partial_before_failure(
        root: Path,
        name: str,
        payload: Any,
    ) -> Path:
        nonlocal failed
        if name == "embedded_articulation_preparation.json" and not failed:
            failed = True
            proposal_path = root / "embedded_articulation_provider_proposal.json"
            proposal_path.unlink()
            proposal_path.symlink_to(outside)
            raise OSError("simulated failure after partial substitution")
        return original_write(root, name, payload)

    monkeypatch.setattr(
        proposal_provider_module,
        "_write_json_create_only",
        substitute_partial_before_failure,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "substituted-partial-attempt",
            intent="Terminalize an unsafe substituted partial.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="substituted-partial-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert terminal.partial_outputs == ()
    assert outside.read_text(encoding="utf-8") == '{"outside": true}\n'


def test_native_payload_readback_failure_binds_created_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_binding = proposal_provider_module._published_binding
    failed = False

    def fail_native_payload_binding_once(
        root: Path,
        name: str,
        *,
        label: str,
        max_bytes: int | None = None,
    ) -> ExecutionArtifactBinding:
        nonlocal failed
        if name == "articulation_proposal_provider_payload.json" and not failed:
            failed = True
            raise OSError("simulated native-payload readback failure")
        return original_binding(root, name, label=label, max_bytes=max_bytes)

    monkeypatch.setattr(
        proposal_provider_module,
        "_published_binding",
        fail_native_payload_binding_once,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "partial-native-payload-attempt",
            intent="Bind a created native payload after readback failure.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="partial-native-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert terminal.native_payload is None
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "articulation_proposal_provider_payload.json",
    )


def test_success_terminal_write_failure_becomes_publication_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_write = proposal_provider_module._write_json_create_only
    failed = False

    def fail_success_terminal_once(
        root: Path,
        name: str,
        payload: Any,
    ) -> Path:
        nonlocal failed
        if name == "articulation_proposal_attempt_terminal.json" and not failed:
            failed = True
            raise OSError("simulated terminal storage failure")
        return original_write(root, name, payload)

    monkeypatch.setattr(
        proposal_provider_module,
        "_write_json_create_only",
        fail_success_terminal_once,
    )
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "terminal-storage-failure",
            intent="Terminalize local finalization failure.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="terminal-storage-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "publication_failure"
    assert terminal.failure_code == "attempt_publication_failure"
    assert tuple(Path(item.path).name for item in terminal.partial_outputs) == (
        "embedded_articulation_provider_proposal.json",
        "embedded_articulation_preparation.json",
    )


def test_stat_open_race_still_emits_input_drift_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, preparation_path, evidence_path = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    from content_agent_workflows.articulation import (
        proposal_provider as proposal_provider_module,
    )

    original_sha256 = proposal_provider_module.file_sha256
    evidence_hashes = 0

    def race_file_sha256(path: str | Path) -> str:
        nonlocal evidence_hashes
        if Path(path) == evidence_path:
            evidence_hashes += 1
            if evidence_hashes == 2:
                raise FileNotFoundError("simulated disappearance after stat")
        return original_sha256(path)

    monkeypatch.setattr(proposal_provider_module, "file_sha256", race_file_sha256)
    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "stat-open-race-attempt",
            intent="Terminalize an exact input race.",
            provider=ArtifactJsonArticulationProposalProvider(
                provider_id="racing-provider",
                capability_id="articulation-proposal-v1",
                payload_path=payload_path,
            ),
        )

    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert str(evidence_path.resolve()) in terminal.drifted_inputs
    assert (tmp_path / "stat-open-race-attempt").stat().st_mode & 0o777 == 0o500


def test_provider_payload_substitution_is_exact_input_drift(
    tmp_path: Path,
) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="substituted-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    payload_path.write_text('{"substituted":true}\n', encoding="utf-8")

    with pytest.raises(ArticulationProposalAttemptFailed) as exc_info:
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "substituted-provider-attempt",
            intent="Suggest candidates.",
            provider=provider,
        )
    terminal = validate_articulation_proposal_attempt_receipt(
        exc_info.value.receipt_binding.path
    )
    assert terminal.disposition == "input_drift"
    assert terminal.drifted_inputs == (str(payload_path.resolve()),)


def test_replacement_requires_receipt_and_nonempty_reason(tmp_path: Path) -> None:
    _, preparation_path, _ = _preparation(tmp_path)
    payload_path = tmp_path / "provider-payload.json"
    _provider_payload(payload_path)
    provider = ArtifactJsonArticulationProposalProvider(
        provider_id="replacement-provider",
        capability_id="articulation-proposal-v1",
        payload_path=payload_path,
    )
    with pytest.raises(ValueError, match="requires a prior terminal"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "reason-only",
            intent="Suggest candidates.",
            provider=provider,
            replacement_reason="No prior receipt.",
        )
    with pytest.raises(ValueError, match="explicit reason"):
        request_embedded_articulation_provider_proposal(
            preparation_path,
            output_dir=tmp_path / "receipt-only",
            intent="Suggest candidates.",
            provider=provider,
            replaces_terminal_receipt=tmp_path / "missing-terminal.json",
        )
