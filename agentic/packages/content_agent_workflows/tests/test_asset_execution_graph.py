# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic conformance tests for the outer-supplied asset execution graph."""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict, ValidationError

import content_agent_workflows.asset_composition.cli as asset_cli
import content_agent_workflows.asset_composition.coordinator as asset_coordinator
import content_agent_workflows.asset_composition.execution as asset_execution
import content_agent_workflows.asset_composition.state as asset_state
from content_agent_workflows.asset_composition import (
    ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
    ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION,
    ASSET_LEAF_CATALOG_SCHEMA_VERSION,
    ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
    LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
    LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION,
    LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION,
    LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
    ArtifactBinding,
    AssetCompositionRun,
    AssetCompositionStateError,
    AssetCoordinatorSession,
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetGraphTerminalReceipt,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjection,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafReceipt,
    AssetLeafRuntimeBinding,
    AssetLeafRuntimeCatalog,
    AssetRunRequest,
    AssetRuntimeRequest,
    AssetSoleCoordinatorIdentity,
    AssetSourceStaging,
    begin_leaf,
    cancel_leaf,
    canonical_asset_digest,
    complete_leaf,
    compose_asset_leaf_runtime_catalog,
    create_run,
    fail_leaf,
    finalize_graph_run,
    freeze_execution_graph,
    leaf_directory,
    load_verified_run,
    recover_leaf,
    run_interactive_asset_coordinator,
    validate_terminal,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
    PROVIDED_VALIDATION_INGRESS_LEAF_ID,
    CanonicalOvrtxEvidenceLeafInvocation,
    CanonicalOvrtxEvidenceLeafTerminalResult,
    FocusedValidationOperationLeafInvocation,
    FocusedValidationOperationLeafResult,
    ProvidedValidationIngressLeafInvocation,
    ProvidedValidationIngressLeafResult,
    SharedValidationLeafTerminalResult,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition.cli import main as asset_state_main
from content_agent_workflows.asset_composition.release_leaf_adapters import (
    FINAL_OVRTX_EVIDENCE_LEAF_ID,
    RELEASE_COMPOSED_LEAF_IDS,
    release_composed_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.validation import (
    CanonicalVisualEvidenceRequest,
    execution_artifact_binding,
)

_RUNTIME_CATALOGS: dict[str, AssetLeafRuntimeCatalog] = {}


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _FixtureInvocation(_FrozenModel):
    leaf_id: str
    kind: str


class _FixtureResult(_FrozenModel):
    leaf_id: str
    native_disposition: str
    native_status: str
    native_terminal_receipt: ArtifactBinding
    operation_indexes: tuple[ArtifactBinding, ...] = ()
    evidence_indexes: tuple[ArtifactBinding, ...] = ()
    evidence: tuple[ArtifactBinding, ...] = ()
    saved_stage_readbacks: tuple[ArtifactBinding, ...] = ()
    resource_claims: tuple[str, ...] = ()
    resource_release_receipts: tuple[ArtifactBinding, ...] = ()
    summary: str
    error: str | None = None


class _NativeArtifactChain(_FrozenModel):
    invocation_artifact: ArtifactBinding
    result_artifact: ArtifactBinding


class _ReceiptBoundFixtureResult(_FixtureResult):
    native_artifact_chain: _NativeArtifactChain


class _InvocationReceiptBoundFixtureResult(_FixtureResult):
    native_invocation_artifact: ArtifactBinding


def _fixture_projection_payload(native: _FixtureResult) -> AssetLeafProjectionPayload:
    return AssetLeafProjectionPayload(
        native_disposition=native.native_disposition,  # type: ignore[arg-type]
        native_status=native.native_status,
        native_terminal_receipt=native.native_terminal_receipt,
        operation_indexes=native.operation_indexes,
        evidence_indexes=native.evidence_indexes,
        evidence=native.evidence,
        saved_stage_readbacks=native.saved_stage_readbacks,
        resource_claims=native.resource_claims,
        resource_release_receipts=native.resource_release_receipts,
        summary=native.summary,
        error=native.error,
    )


def _fixture_projector(
    invocation: BaseModel,
    result: BaseModel,
    _context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = _FixtureInvocation.model_validate(invocation)
    native = _FixtureResult.model_validate(result)
    if request.leaf_id != native.leaf_id:
        raise ValueError("fixture result belongs to another invocation")
    return _fixture_projection_payload(native)


def _receipt_bound_fixture_projector(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = _FixtureInvocation.model_validate(invocation)
    native = _ReceiptBoundFixtureResult.model_validate(result)
    if request.leaf_id != native.leaf_id:
        raise ValueError("fixture result belongs to another invocation")
    context.require_native_artifact_chain(
        invocation_artifact=native.native_artifact_chain.invocation_artifact,
        result_artifact=native.native_artifact_chain.result_artifact,
    )
    return _fixture_projection_payload(native)


def _invocation_receipt_bound_fixture_projector(
    invocation: BaseModel,
    result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    request = _FixtureInvocation.model_validate(invocation)
    native = _InvocationReceiptBoundFixtureResult.model_validate(result)
    if request.leaf_id != native.leaf_id:
        raise ValueError("fixture result belongs to another invocation")
    context.require_invocation_artifact(native.native_invocation_artifact)
    return _fixture_projection_payload(native)


@pytest.fixture(autouse=True)
def _resolve_fixture_repository_catalog(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    _RUNTIME_CATALOGS.clear()

    def resolve(candidate: AssetLeafCatalog) -> AssetLeafRuntimeCatalog:
        runtime = _RUNTIME_CATALOGS.get(candidate.catalog_digest)
        if runtime is None or runtime.catalog != candidate:
            raise ValueError("fixture catalog does not resolve")
        return runtime

    monkeypatch.setattr(asset_state, "resolve_repository_asset_leaf_catalog", resolve)
    yield
    _RUNTIME_CATALOGS.clear()


def _binding(path: Path) -> ArtifactBinding:
    resolved = path.resolve()
    return ArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def test_catalog_cli_preserves_typed_composed_leaf_failure_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation_path = tmp_path / "invocation.json"
    launcher = tmp_path / "content-workflow-composed-leaf"
    terminal = ArtifactBinding(
        path=str((tmp_path / "terminal.json").resolve()),
        sha256="0" * 64,
        size_bytes=1,
    )
    expected = _FixtureResult(
        leaf_id="material.assignment.v1",
        native_disposition="failed",
        native_status="exception",
        native_terminal_receipt=terminal,
        summary="Material failed closed.",
        error="fixture failure",
    )
    observed: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        return subprocess.CompletedProcess(
            command,
            returncode=2,
            stdout=expected.model_dump_json(),
            stderr="fixture failure",
        )

    monkeypatch.setattr(
        asset_execution,
        "_content_workflow_composed_leaf",
        lambda: launcher,
    )
    monkeypatch.setattr(asset_execution.subprocess, "run", run)

    result = asset_execution._execute_cli_entrypoint(
        "content-workflow-composed-leaf material --invocation",
        invocation_path=invocation_path,
        result_model=_FixtureResult,
    )

    assert result == expected
    assert observed == [
        [str(launcher), "material", "--invocation", str(invocation_path)]
    ]


def test_release_composed_catalog_has_a_trusted_executor_for_every_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    composed_launcher = tmp_path / "content-workflow-composed-leaf"
    monkeypatch.setattr(
        asset_execution,
        "_content_workflow_composed_leaf",
        lambda: composed_launcher,
    )
    bindings = release_composed_asset_leaf_runtime_bundle().bindings

    assert {binding.descriptor.leaf_id for binding in bindings} == set(
        RELEASE_COMPOSED_LEAF_IDS
    )
    for binding in bindings:
        descriptor = binding.descriptor
        if descriptor.leaf_id == FINAL_OVRTX_EVIDENCE_LEAF_ID:
            assert descriptor.entrypoint == (
                "content_agent_workflows.validation.produce_canonical_visual_evidence"
            )
            continue
        executable, arguments = asset_execution._catalog_cli_command(
            descriptor.entrypoint
        )
        assert executable == composed_launcher
        assert arguments[-1] == "--invocation"


def test_catalog_cli_rejects_nonzero_exit_without_typed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    command = ["fixture"]
    monkeypatch.setattr(
        asset_execution,
        "_content_workflow_composed_leaf",
        lambda: tmp_path / "content-workflow-composed-leaf",
    )
    monkeypatch.setattr(
        asset_execution.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            command,
            returncode=2,
            stdout="not typed JSON",
            stderr="fixture failure",
        ),
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="exit code 2 without a valid typed result",
    ):
        asset_execution._execute_cli_entrypoint(
            "content-workflow-composed-leaf material --invocation",
            invocation_path=tmp_path / "invocation.json",
            result_model=_FixtureResult,
        )


def test_catalog_cli_rejects_unknown_composed_leaf_operation(tmp_path: Path) -> None:
    with pytest.raises(
        AssetCompositionStateError,
        match="not an admitted invocation form",
    ):
        asset_execution._execute_cli_entrypoint(
            "content-workflow-composed-leaf arbitrary --invocation",
            invocation_path=tmp_path / "invocation.json",
            result_model=_FixtureResult,
        )


def test_shared_executor_dispatches_final_ovrtx_by_frozen_entrypoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal = ArtifactBinding(
        path=str((tmp_path / "terminal.json").resolve()),
        sha256="0" * 64,
        size_bytes=1,
    )
    invocation = _FixtureInvocation(leaf_id="asset.final-ovrtx-evidence.v1", kind="x")
    expected = _FixtureResult(
        leaf_id=invocation.leaf_id,
        native_disposition="passed",
        native_status="completed",
        native_terminal_receipt=terminal,
        summary="OVRTX evidence completed.",
    )
    import content_agent_workflows.validation as validation

    monkeypatch.setattr(
        validation,
        "produce_canonical_visual_evidence",
        lambda **_values: expected,
    )

    result = asset_execution._execute_shared_entrypoint(
        "content_agent_workflows.validation.produce_canonical_visual_evidence",
        invocation=invocation,
        result_model=_FixtureResult,
    )

    assert result == expected


def _descriptor(
    leaf_id: str,
    *,
    required_dependencies: list[str] | None = None,
    required_dependents: list[str] | None = None,
    incompatible_leaf_ids: list[str] | None = None,
) -> AssetLeafDescriptor:
    return _runtime_binding(
        leaf_id,
        required_dependencies=required_dependencies,
        required_dependents=required_dependents,
        incompatible_leaf_ids=incompatible_leaf_ids,
    ).descriptor


def _runtime_binding(
    leaf_id: str,
    *,
    required_dependencies: list[str] | None = None,
    required_dependents: list[str] | None = None,
    incompatible_leaf_ids: list[str] | None = None,
    required_artifact_categories: tuple[str, ...] = (),
) -> AssetLeafRuntimeBinding:
    return AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint=f"opaque-fixture --leaf {leaf_id}",
        invocation_model=_FixtureInvocation,
        result_model=_FixtureResult,
        projector_id=f"projector.{leaf_id}",
        projector=_fixture_projector,
        required_artifact_categories=required_artifact_categories,  # type: ignore[arg-type]
        required_dependencies=required_dependencies or (),
        required_dependents=required_dependents or (),
        incompatible_leaf_ids=incompatible_leaf_ids or (),
    )


def _receipt_bound_runtime_binding(leaf_id: str) -> AssetLeafRuntimeBinding:
    return AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint=f"receipt-bound-fixture --leaf {leaf_id}",
        invocation_model=_FixtureInvocation,
        result_model=_ReceiptBoundFixtureResult,
        projector_id=f"projector.receipt-bound.{leaf_id}",
        projector=_receipt_bound_fixture_projector,
    )


def _invocation_receipt_bound_runtime_binding(
    leaf_id: str,
) -> AssetLeafRuntimeBinding:
    return AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint=f"invocation-receipt-bound-fixture --leaf {leaf_id}",
        invocation_model=_FixtureInvocation,
        result_model=_InvocationReceiptBoundFixtureResult,
        projector_id=f"projector.invocation-receipt-bound.{leaf_id}",
        projector=_invocation_receipt_bound_fixture_projector,
    )


def _artifact_identity(
    path: str,
    *,
    sha256: str,
    size_bytes: int,
) -> ArtifactBinding:
    return ArtifactBinding(path=path, sha256=sha256, size_bytes=size_bytes)


def _project_receipt_bound_fixture(
    *,
    invocation_artifact: ArtifactBinding,
    result_artifact: ArtifactBinding,
    native_invocation_artifact: ArtifactBinding,
    native_result_artifact: ArtifactBinding,
) -> AssetLeafProjection:
    leaf_id = "receipt-bound.v1"
    runtime = _receipt_bound_runtime_binding(leaf_id)
    invocation = _FixtureInvocation(leaf_id=leaf_id, kind="fixture")
    result = _ReceiptBoundFixtureResult(
        leaf_id=leaf_id,
        native_disposition="passed",
        native_status="succeeded",
        native_terminal_receipt=_artifact_identity(
            "/run/native-terminal.json",
            sha256="f" * 64,
            size_bytes=64,
        ),
        native_artifact_chain=_NativeArtifactChain(
            invocation_artifact=native_invocation_artifact,
            result_artifact=native_result_artifact,
        ),
        summary="Receipt-bound fixture completed.",
    )
    return runtime.project(
        invocation,
        result,
        invocation_artifact=invocation_artifact,
        result_artifact=result_artifact,
    )


def test_projection_context_rejects_identical_bytes_at_different_path() -> None:
    selected_invocation = _artifact_identity(
        "/run/selected-invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )
    substituted_invocation = selected_invocation.model_copy(
        update={"path": "/run/substituted-invocation.json"}
    )

    with pytest.raises(ValueError, match="native invocation artifact differs"):
        _project_receipt_bound_fixture(
            invocation_artifact=substituted_invocation,
            result_artifact=selected_result,
            native_invocation_artifact=selected_invocation,
            native_result_artifact=selected_result,
        )


def test_projection_context_rejects_swapped_invocation_and_result_bindings() -> None:
    selected_invocation = _artifact_identity(
        "/run/invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )

    with pytest.raises(ValueError, match="native invocation artifact differs"):
        _project_receipt_bound_fixture(
            invocation_artifact=selected_invocation,
            result_artifact=selected_result,
            native_invocation_artifact=selected_result,
            native_result_artifact=selected_invocation,
        )


def test_projection_context_rejects_stale_native_binding() -> None:
    selected_invocation = _artifact_identity(
        "/run/attempt-2/invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/attempt-2/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )
    stale_result = _artifact_identity(
        "/run/attempt-1/result.json",
        sha256="c" * 64,
        size_bytes=47,
    )

    with pytest.raises(ValueError, match="native result artifact differs"):
        _project_receipt_bound_fixture(
            invocation_artifact=selected_invocation,
            result_artifact=selected_result,
            native_invocation_artifact=selected_invocation,
            native_result_artifact=stale_result,
        )


def test_projection_context_rejects_digest_drift() -> None:
    selected_invocation = _artifact_identity(
        "/run/invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )
    digest_drift = selected_result.model_copy(update={"sha256": "c" * 64})

    with pytest.raises(ValueError, match="native result artifact differs"):
        _project_receipt_bound_fixture(
            invocation_artifact=selected_invocation,
            result_artifact=selected_result,
            native_invocation_artifact=selected_invocation,
            native_result_artifact=digest_drift,
        )


def test_projection_context_rejects_size_drift() -> None:
    selected_invocation = _artifact_identity(
        "/run/invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )
    size_drift = selected_result.model_copy(update={"size_bytes": 49})

    with pytest.raises(ValueError, match="native result artifact differs"):
        _project_receipt_bound_fixture(
            invocation_artifact=selected_invocation,
            result_artifact=selected_result,
            native_invocation_artifact=selected_invocation,
            native_result_artifact=size_drift,
        )


def test_domain_projector_accepts_exact_frozen_native_receipt_chain() -> None:
    selected_invocation = _artifact_identity(
        "/run/invocation.json",
        sha256="a" * 64,
        size_bytes=32,
    )
    selected_result = _artifact_identity(
        "/run/result.json",
        sha256="b" * 64,
        size_bytes=48,
    )

    projection = _project_receipt_bound_fixture(
        invocation_artifact=selected_invocation,
        result_artifact=selected_result,
        native_invocation_artifact=selected_invocation,
        native_result_artifact=selected_result,
    )

    assert projection.schema_version == "content-agents.asset-leaf-projection.v2"
    assert (
        projection.context.schema_version
        == "content-agents.asset-leaf-projection-context.v1"
    )
    assert (
        projection.context.projector_api_version
        == "content-agents.asset-leaf-projector-api.v2"
    )
    assert projection.context.invocation_artifact == selected_invocation
    assert projection.context.result_artifact == selected_result
    assert (
        projection.projection_context_schema_digest
        == projection.context.context_schema_digest
    )
    assert (
        AssetLeafProjectionContext.model_validate_json(
            projection.context.model_dump_json()
        )
        == projection.context
    )


def _create_agentic_run(
    tmp_path: Path,
    *,
    descriptors: list[AssetLeafDescriptor] | None = None,
    runtime_bindings: list[AssetLeafRuntimeBinding] | None = None,
    requires_parent_resource_release: bool = False,
    legacy_catalog: bool = False,
    required_leaf_ids: list[str] | None = None,
    required_terminal_leaf_ids: list[str] | None = None,
    required_leaf_dependencies: dict[str, list[str]] | None = None,
    exact_leaf_scope: bool = False,
) -> tuple[Path, AssetRunRequest]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = tmp_path / "source.usda"
    staged = run_dir / "source.usda"
    source_text = '#usda 1.0\ndef Xform "World" {}\n'
    original.write_text(source_text, encoding="utf-8")
    staged.write_text(source_text, encoding="utf-8")
    original_binding = _binding(original)
    staged_binding = _binding(staged)
    digest_set = hashlib.sha256(
        json.dumps(
            [staged_binding.sha256],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest = run_dir / "source_manifest.json"
    atomic_write_json(
        manifest,
        {
            "source_usd_path": original_binding.path,
            "source_sha256": original_binding.sha256,
            "staged_usd_path": staged_binding.path,
            "dependency_digest_set_sha256": digest_set,
            "unresolved_dependencies": [],
            "files": [
                {
                    "source_path": original_binding.path,
                    "staged_path": staged_binding.path,
                    "sha256": staged_binding.sha256,
                    "size_bytes": staged_binding.size_bytes,
                }
            ],
        },
    )
    staging = AssetSourceStaging(
        original_source=original_binding,
        original_dependencies=[original_binding],
        staged_source=staged_binding,
        staged_dependencies=[staged_binding],
        manifest=_binding(manifest),
        dependency_digest_set_sha256=digest_set,
    )
    if descriptors is not None and runtime_bindings is not None:
        raise ValueError("provide descriptors or runtime bindings, not both")
    if legacy_catalog and runtime_bindings is not None:
        raise ValueError("legacy catalogs cannot register v2 runtime bindings")
    if legacy_catalog:
        catalog = AssetLeafCatalog.create(
            descriptors
            or [
                AssetLeafDescriptor.create(
                    leaf_id="inspect.v1",
                    entrypoint="legacy inspect",
                    invocation_schema_digest="1" * 64,
                    result_schema_digest="2" * 64,
                ),
                AssetLeafDescriptor.create(
                    leaf_id="optional-check.v1",
                    entrypoint="legacy optional check",
                    invocation_schema_digest="3" * 64,
                    result_schema_digest="4" * 64,
                ),
                AssetLeafDescriptor.create(
                    leaf_id="publish.v1",
                    entrypoint="legacy publish",
                    invocation_schema_digest="5" * 64,
                    result_schema_digest="6" * 64,
                    required_dependencies=["inspect.v1"],
                ),
            ]
        )
    elif runtime_bindings is not None:
        runtime_catalog = compose_asset_leaf_runtime_catalog(runtime_bindings)
    else:
        selected_descriptors = descriptors or [
            _descriptor("inspect.v1"),
            _descriptor("optional-check.v1"),
            _descriptor("publish.v1", required_dependencies=["inspect.v1"]),
        ]
        runtime_catalog = compose_asset_leaf_runtime_catalog(
            _runtime_binding(
                descriptor.leaf_id,
                required_dependencies=descriptor.required_dependencies,
                required_dependents=descriptor.required_dependents,
                incompatible_leaf_ids=descriptor.incompatible_leaf_ids,
                required_artifact_categories=tuple(
                    descriptor.required_artifact_categories
                ),
            )
            for descriptor in selected_descriptors
        )
    if not legacy_catalog:
        catalog = runtime_catalog.catalog
        _RUNTIME_CATALOGS[catalog.catalog_digest] = runtime_catalog
    coordinator = AssetSoleCoordinatorIdentity.create(
        coordinator_id="asset-coordinator:test-run",
        invocation_id="single-prompt:test-run",
        actor="test-outer-reasoner",
        implementation="codex-interactive",
    )
    runtime = AssetRuntimeRequest(
        runner="codex",
        scene_tool_timeout_seconds=60.0,
        child_timeout_seconds=0.0,
        codex_sandbox_mode="workspace-write",
        claude_permission_mode="default",
        claude_execution_mode="sdk",
    )
    prompt = "Select only the opaque operations needed for this asset."
    configuration_identity: dict[str, object] = {
        "repository_root": str(tmp_path.resolve()),
        "runtime": runtime.model_dump(mode="json"),
        "requires_parent_resource_release": requires_parent_resource_release,
    }
    if required_leaf_ids or required_terminal_leaf_ids or required_leaf_dependencies:
        configuration_identity.update(
            {
                "required_leaf_ids": required_leaf_ids or [],
                "required_terminal_leaf_ids": required_terminal_leaf_ids or [],
                "required_leaf_dependencies": required_leaf_dependencies or {},
            }
        )
    if exact_leaf_scope:
        configuration_identity["exact_leaf_scope"] = True
    configuration_digest = canonical_asset_digest(configuration_identity)
    request = AssetRunRequest(
        created_at="2026-08-17T00:00:00+00:00",
        schema_version="content-agents.asset-composition-request.v3",
        selected_mode="agentic",
        run_id="test-run",
        run_dir=str(run_dir.resolve()),
        run_state=str((run_dir / "asset_run.json").resolve()),
        repository_root=str(tmp_path.resolve()),
        source_asset=staged_binding.path,
        source_staging=staging,
        prompt=prompt,
        prompt_digest=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        source_digest=staged_binding.sha256,
        configuration_digest=configuration_digest,
        reference_digest=canonical_asset_digest([]),
        sole_coordinator_identity=coordinator,
        leaf_catalog=catalog,
        required_leaf_ids=required_leaf_ids or [],
        required_terminal_leaf_ids=required_terminal_leaf_ids or [],
        required_leaf_dependencies=required_leaf_dependencies or {},
        exact_leaf_scope=exact_leaf_scope,
        requires_parent_resource_release=requires_parent_resource_release,
        runtime=runtime,
    )
    request_path = run_dir / "request.json"
    atomic_write_json(request_path, request)
    state_path = run_dir / "asset_run.json"
    create_run(
        state_path,
        run_id=request.run_id,
        request_path=request_path,
        source_asset=staged,
    )
    return state_path, request


def _graph(
    request: AssetRunRequest,
    *,
    selected: list[tuple[str, list[str], str, bool]],
    omitted: list[str],
    selection_rationales: dict[str, str] | None = None,
) -> AssetExecutionGraph:
    assert request.sole_coordinator_identity is not None
    assert request.leaf_catalog is not None
    descriptors = {
        descriptor.leaf_id: descriptor
        for descriptor in request.leaf_catalog.descriptors
    }
    nodes = [
        AssetExecutionNode(
            leaf_id=leaf_id,
            depends_on=depends_on,
            requirement=requirement,
            descriptor_digest=descriptors[leaf_id].descriptor_digest,
            terminal_output=terminal_output,
            selection_rationale=(selection_rationales or {}).get(leaf_id),
        )
        for leaf_id, depends_on, requirement, terminal_output in selected
    ]
    return AssetExecutionGraph.create(
        sole_coordinator_identity_digest=(
            request.sole_coordinator_identity.identity_digest
        ),
        prompt_digest=str(request.prompt_digest),
        source_digest=str(request.source_digest),
        configuration_digest=str(request.configuration_digest),
        reference_digest=str(request.reference_digest),
        leaf_catalog_digest=request.leaf_catalog.catalog_digest,
        nodes=nodes,
        omitted_leaf_ids=omitted,
    )


def test_invoke_leaf_executes_only_active_frozen_descriptor_and_seals_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf_id = "inspect.v1"
    runtime_binding = AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint="content-workflow-cli fixture inspect --invocation",
        invocation_model=_FixtureInvocation,
        result_model=_FixtureResult,
        projector_id=f"projector.{leaf_id}",
        projector=_fixture_projector,
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=[runtime_binding],
    )
    graph = _graph(
        request,
        selected=[(leaf_id, [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path, actor="test")
    begin_leaf(state_path, leaf_id, actor="test")
    attempt = leaf_directory(state_path, leaf_id)
    invocation_path = attempt / "invocation.json"
    atomic_write_json(
        invocation_path,
        _FixtureInvocation(leaf_id=leaf_id, kind="fixture"),
    )
    terminal_path = attempt / "native-terminal.json"
    atomic_write_json(terminal_path, {"status": "completed"})
    expected_result = _FixtureResult(
        leaf_id=leaf_id,
        native_disposition="passed",
        native_status="completed",
        native_terminal_receipt=_binding(terminal_path),
        summary="Fixture completed.",
    )
    runtime_catalog = compose_asset_leaf_runtime_catalog([runtime_binding])
    observed: dict[str, object] = {}

    def execute_cli(
        entrypoint: str,
        *,
        invocation_path: Path,
        result_model: type[BaseModel],
    ) -> BaseModel:
        observed.update(
            entrypoint=entrypoint,
            invocation_path=invocation_path,
            result_model=result_model,
        )
        return expected_result

    monkeypatch.setattr(
        asset_execution,
        "repository_asset_leaf_runtime_catalog",
        lambda: runtime_catalog,
    )
    monkeypatch.setattr(asset_execution, "_execute_cli_entrypoint", execute_cli)
    result_path = attempt / "result.json"

    result_binding = asset_execution.execute_active_leaf(
        state_path,
        leaf_id,
        invocation_path=invocation_path.resolve(),
        result_path=result_path.resolve(),
    )

    assert observed == {
        "entrypoint": runtime_binding.descriptor.entrypoint,
        "invocation_path": invocation_path.resolve(),
        "result_model": _FixtureResult,
    }
    assert _binding(result_path) == result_binding
    assert (
        _FixtureResult.model_validate_json(result_path.read_bytes()) == expected_result
    )


def test_invoke_leaf_serializes_attempt_and_reuses_sealed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf_id = "inspect.v1"
    runtime_binding = AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint="content-workflow-cli fixture inspect --invocation",
        invocation_model=_FixtureInvocation,
        result_model=_FixtureResult,
        projector_id=f"projector.{leaf_id}",
        projector=_fixture_projector,
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=[runtime_binding],
    )
    graph = _graph(
        request,
        selected=[(leaf_id, [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path, actor="test")
    begin_leaf(state_path, leaf_id, actor="test")
    attempt = leaf_directory(state_path, leaf_id)
    invocation_path = attempt / "invocation.json"
    atomic_write_json(
        invocation_path,
        _FixtureInvocation(leaf_id=leaf_id, kind="fixture"),
    )
    terminal_path = attempt / "native-terminal.json"
    atomic_write_json(terminal_path, {"status": "completed"})
    expected_result = _FixtureResult(
        leaf_id=leaf_id,
        native_disposition="passed",
        native_status="completed",
        native_terminal_receipt=_binding(terminal_path),
        summary="Fixture completed.",
    )
    runtime_catalog = compose_asset_leaf_runtime_catalog([runtime_binding])
    entered = threading.Event()
    release = threading.Event()
    calls = 0

    def execute_cli(
        _entrypoint: str,
        *,
        invocation_path: Path,
        result_model: type[BaseModel],
    ) -> BaseModel:
        nonlocal calls
        assert invocation_path == attempt / "invocation.json"
        assert result_model is _FixtureResult
        calls += 1
        entered.set()
        assert release.wait(timeout=5)
        return expected_result

    monkeypatch.setattr(
        asset_execution,
        "repository_asset_leaf_runtime_catalog",
        lambda: runtime_catalog,
    )
    monkeypatch.setattr(asset_execution, "_execute_cli_entrypoint", execute_cli)
    result_path = attempt / "result.json"
    outcomes: list[ArtifactBinding | BaseException] = []

    def run_first() -> None:
        try:
            outcomes.append(
                asset_execution.execute_active_leaf(
                    state_path,
                    leaf_id,
                    invocation_path=invocation_path.resolve(),
                    result_path=result_path.resolve(),
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            outcomes.append(exc)

    worker = threading.Thread(target=run_first)
    worker.start()
    assert entered.wait(timeout=5)

    with pytest.raises(
        AssetCompositionStateError,
        match="already owns this active leaf attempt",
    ):
        asset_execution.execute_active_leaf(
            state_path,
            leaf_id,
            invocation_path=invocation_path.resolve(),
            result_path=result_path.resolve(),
        )

    state_lock = FileLock(str(state_path.with_name(f".{state_path.name}.lock")))
    try:
        with pytest.raises(Timeout):
            state_lock.acquire(timeout=0)
    finally:
        state_lock.release()

    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert isinstance(outcomes[0], ArtifactBinding)
    first = outcomes[0]
    assert calls == 1

    repeated = asset_execution.execute_active_leaf(
        state_path,
        leaf_id,
        invocation_path=invocation_path.resolve(),
        result_path=result_path.resolve(),
    )
    assert repeated == first
    assert calls == 1
    with pytest.raises(
        AssetCompositionStateError,
        match="already invoked with different bindings",
    ):
        asset_execution.execute_active_leaf(
            state_path,
            leaf_id,
            invocation_path=invocation_path.resolve(),
            result_path=(attempt / "other-result.json").resolve(),
        )
    assert calls == 1


def test_leaf_invocation_is_read_through_pinned_confined_descriptor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    invocation = attempt / "invocation.json"
    invocation.write_bytes(b'{"leaf_id":"inspect.v1"}\n')
    opened: list[str] = []
    real_open = asset_execution.open_confined_regular_file

    @contextmanager
    def record_open(root_descriptor: int, relative_key: str):
        opened.append(relative_key)
        with real_open(root_descriptor, relative_key) as held:
            yield held

    def reject_path_read(*_args: object, **_kwargs: object) -> bytes:
        raise AssertionError("invocation bytes must not be reopened by path")

    monkeypatch.setattr(asset_execution, "open_confined_regular_file", record_open)
    monkeypatch.setattr(Path, "read_bytes", reject_path_read)

    assert (
        asset_execution._confined_regular_file(
            invocation.resolve(),
            within=attempt,
            label="leaf invocation",
        )
        == b'{"leaf_id":"inspect.v1"}\n'
    )
    assert opened == ["invocation.json"]


def test_invoke_leaf_rejects_result_outside_active_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    leaf_id = "inspect.v1"
    runtime_binding = AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint="content-workflow-cli fixture inspect --invocation",
        invocation_model=_FixtureInvocation,
        result_model=_FixtureResult,
        projector_id=f"projector.{leaf_id}",
        projector=_fixture_projector,
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=[runtime_binding],
    )
    graph = _graph(
        request,
        selected=[(leaf_id, [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path, actor="test")
    begin_leaf(state_path, leaf_id, actor="test")
    attempt = leaf_directory(state_path, leaf_id)
    invocation_path = attempt / "invocation.json"
    atomic_write_json(
        invocation_path,
        _FixtureInvocation(leaf_id=leaf_id, kind="fixture"),
    )
    runtime_catalog = compose_asset_leaf_runtime_catalog([runtime_binding])
    monkeypatch.setattr(
        asset_execution,
        "repository_asset_leaf_runtime_catalog",
        lambda: runtime_catalog,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="direct child of the active attempt",
    ):
        asset_execution.execute_active_leaf(
            state_path,
            leaf_id,
            invocation_path=invocation_path.resolve(),
            result_path=(state_path.parent / "escaped.json").resolve(),
        )


def _legacy_graph(
    request: AssetRunRequest,
    *,
    selected: list[tuple[str, list[str], str, bool]],
    omitted: list[str],
) -> AssetExecutionGraph:
    assert request.sole_coordinator_identity is not None
    assert request.leaf_catalog is not None
    descriptors = {
        descriptor.leaf_id: descriptor
        for descriptor in request.leaf_catalog.descriptors
    }
    nodes = [
        AssetExecutionNode(
            leaf_id=leaf_id,
            depends_on=depends_on,
            requirement=requirement,
            descriptor_digest=descriptors[leaf_id].descriptor_digest,
            terminal_output=terminal_output,
        )
        for leaf_id, depends_on, requirement, terminal_output in selected
    ]
    payload = {
        "schema_version": LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
        "selected_mode": "agentic",
        "sole_coordinator_identity_digest": (
            request.sole_coordinator_identity.identity_digest
        ),
        "prompt_digest": str(request.prompt_digest),
        "source_digest": str(request.source_digest),
        "configuration_digest": str(request.configuration_digest),
        "reference_digest": str(request.reference_digest),
        "leaf_catalog_digest": request.leaf_catalog.catalog_digest,
        "selected_leaf_ids": [node.leaf_id for node in nodes],
        "nodes": [node.model_dump(mode="json") for node in nodes],
        "omitted_leaf_ids": omitted,
    }
    return AssetExecutionGraph.model_validate(
        {**payload, "graph_digest": canonical_asset_digest(payload)}
    )


def _write_leaf_artifacts(
    state_path: Path,
    leaf_id: str,
    *,
    include_readback: bool = True,
    disposition: str = "passed",
    native_status: str | None = None,
    error: str | None = None,
) -> tuple[Path, Path]:
    directory = leaf_directory(state_path, leaf_id)
    invocation = directory / "invocation.json"
    result = directory / "result.json"
    evidence = directory / "evidence.json"
    readback = directory / "readback.json"
    terminal = directory / "terminal.json"
    for path, payload in (
        (evidence, {"leaf_id": leaf_id, "kind": "evidence"}),
        (readback, {"leaf_id": leaf_id, "kind": "saved-stage-readback"}),
        (terminal, {"leaf_id": leaf_id, "kind": "native-terminal"}),
    ):
        atomic_write_json(path, payload)
    atomic_write_json(
        invocation,
        _FixtureInvocation(leaf_id=leaf_id, kind="invocation"),
    )
    atomic_write_json(
        result,
        _FixtureResult(
            leaf_id=leaf_id,
            native_disposition=disposition,
            native_status=native_status
            or (
                "qualified"
                if disposition == "passed"
                else ("not_produced" if disposition == "not_evaluated" else disposition)
            ),
            native_terminal_receipt=_binding(terminal),
            evidence=(_binding(evidence),),
            saved_stage_readbacks=(_binding(readback),) if include_readback else (),
            summary=f"{leaf_id} reached {disposition}.",
            error=error,
        ),
    )
    return invocation, result


def _write_native_terminal_artifacts(
    state_path: Path,
    leaf_id: str,
    *,
    disposition: str,
    native_status: str | None = None,
    error: str,
) -> dict[str, Path]:
    directory = leaf_directory(state_path, leaf_id)
    artifacts = {
        "invocation": directory / "native_invocation.json",
        "result": directory / "native_result.json",
        "terminal_receipt": directory / "native_terminal_receipt.json",
        "operation_index": directory / "operation_index.json",
        "evidence_index": directory / "evidence_index.json",
        "evidence": directory / "native_evidence.json",
        "readback": directory / "native_saved_stage_readback.json",
        "release": directory / "native_resource_release.json",
    }
    for kind, path in artifacts.items():
        if kind in {"invocation", "result"}:
            continue
        atomic_write_json(
            path,
            {
                "leaf_id": leaf_id,
                "kind": kind,
                "native_status": "non_accepting",
            },
        )
    atomic_write_json(
        artifacts["invocation"],
        _FixtureInvocation(leaf_id=leaf_id, kind="native-invocation"),
    )
    atomic_write_json(
        artifacts["result"],
        _FixtureResult(
            leaf_id=leaf_id,
            native_disposition=disposition,
            native_status=native_status or disposition,
            native_terminal_receipt=_binding(artifacts["terminal_receipt"]),
            operation_indexes=(_binding(artifacts["operation_index"]),),
            evidence_indexes=(_binding(artifacts["evidence_index"]),),
            evidence=(_binding(artifacts["evidence"]),),
            saved_stage_readbacks=(_binding(artifacts["readback"]),),
            resource_claims=("opaque-fixture-resource",),
            resource_release_receipts=(_binding(artifacts["release"]),),
            summary=f"{leaf_id} reached {disposition}.",
            error=error,
        ),
    )
    return artifacts


def _complete_selected_leaf(
    state_path: Path,
    leaf_id: str,
    *,
    disposition: str = "passed",
) -> None:
    begin_leaf(state_path, leaf_id)
    invocation, result = _write_leaf_artifacts(
        state_path,
        leaf_id,
        include_readback=disposition == "passed",
        disposition=disposition,
    )
    complete_leaf(
        state_path,
        leaf_id,
        invocation_path=invocation,
        result_path=result,
    )


def _finalize_v2_graph(state_path: Path) -> AssetCompositionRun:
    root = state_path.parent
    release = root / "parent_release.json"
    journal = root / "parent_receipt_journal.jsonl"
    checkpoint = root / "parent_receipt_checkpoint.json"
    atomic_write_json(release, {"status": "released"})
    journal.write_text('{"command":"fixture"}\n', encoding="utf-8")
    atomic_write_json(checkpoint, {"status": "sealed"})
    release_binding = _binding(release)
    journal_binding = _binding(journal)
    checkpoint_binding = _binding(checkpoint)
    return finalize_graph_run(
        state_path,
        parent_release_receipt_path=release,
        parent_command_receipt_journal_path=journal,
        parent_command_receipt_checkpoint_path=checkpoint,
        expected_parent_release_receipt=release_binding,
        expected_parent_command_receipt_journal=journal_binding,
        expected_parent_command_receipt_checkpoint=checkpoint_binding,
    )


def test_d462_graph_v1_fixture_bytes_remain_exact_and_nonlexical() -> None:
    # This freezes the exact d462 graph-v1 surface and atomic-writer serialization.
    # The byte checksum makes accidental reformatting or canonicalization visible.
    fixture = (
        Path(__file__).parent
        / "fixtures"
        / "asset_execution_graph_v1_d462_nonlexical.json"
    )
    frozen_bytes = fixture.read_bytes()
    assert hashlib.sha256(frozen_bytes).hexdigest() == (
        "9c9b6a5c9fa50afe3da8f0ba709bd4a419304757311b564d65c3ff3df2e21dbe"
    )
    graph = AssetExecutionGraph.model_validate_json(frozen_bytes)

    assert graph.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
    assert graph.selected_leaf_ids == ["z.prepare.v1", "a.publish.v1"]
    assert graph.selected_leaf_ids != sorted(graph.selected_leaf_ids)
    assert graph.omitted_leaf_ids == ["q.optional.v1", "b.audit.v1"]
    assert graph.graph_digest == (
        "e646a31f01259ce0958af97231b1e92de3912e90e10365012b531fdfa63b162a"
    )
    assert (
        json.dumps(
            graph.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
        == frozen_bytes
    )


def test_graph_v1_failed_cancelled_resume_and_terminal_readback(
    tmp_path: Path,
) -> None:
    descriptors = [
        AssetLeafDescriptor.create(
            leaf_id="z.prepare.v1",
            entrypoint="legacy prepare",
            invocation_schema_digest="1" * 64,
            result_schema_digest="2" * 64,
        ),
        AssetLeafDescriptor.create(
            leaf_id="a.publish.v1",
            entrypoint="legacy publish",
            invocation_schema_digest="3" * 64,
            result_schema_digest="4" * 64,
            required_dependencies=["z.prepare.v1"],
        ),
        AssetLeafDescriptor.create(
            leaf_id="m.omitted.v1",
            entrypoint="legacy omission",
            invocation_schema_digest="5" * 64,
            result_schema_digest="6" * 64,
        ),
    ]
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=descriptors,
        requires_parent_resource_release=True,
        legacy_catalog=True,
    )
    graph = _legacy_graph(
        request,
        selected=[
            ("z.prepare.v1", [], "required", False),
            ("a.publish.v1", ["z.prepare.v1"], "required", True),
        ],
        omitted=["m.omitted.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    frozen = session.freeze_graph(graph_path)

    assert frozen.current_leaf_id == "z.prepare.v1"
    session.begin_leaf("z.prepare.v1")
    failed_artifacts = _write_native_terminal_artifacts(
        state_path,
        "z.prepare.v1",
        disposition="failed",
        error="legacy failure",
    )
    failed = session.fail_leaf(
        "z.prepare.v1",
        reason="legacy failure",
        invocation_path=failed_artifacts["invocation"],
        result_path=failed_artifacts["result"],
        native_terminal_receipt_path=failed_artifacts["terminal_receipt"],
        operation_index_paths=[failed_artifacts["operation_index"]],
        evidence_index_paths=[failed_artifacts["evidence_index"]],
        evidence_paths=[failed_artifacts["evidence"]],
        saved_stage_readback_paths=[failed_artifacts["readback"]],
        resource_claims=["legacy-fixture-resource"],
        resource_release_paths=[failed_artifacts["release"]],
    )
    failed_receipt = failed.leaf_states["z.prepare.v1"].receipt
    assert failed_receipt is not None
    failed_payload = json.loads(Path(failed_receipt.path).read_text(encoding="utf-8"))
    assert failed_payload["schema_version"] == LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    assert "projection" not in failed_payload
    assert failed_payload["native_terminal_receipt"] == _binding(
        failed_artifacts["terminal_receipt"]
    ).model_dump(mode="json")
    session.recover_leaf("z.prepare.v1", reason="legacy failure repaired")
    session.begin_leaf("z.prepare.v1")
    invocation, result = _write_leaf_artifacts(state_path, "z.prepare.v1")
    session.complete_leaf(
        "z.prepare.v1",
        invocation_path=invocation,
        result_path=result,
        evidence_paths=[invocation.parent / "evidence.json"],
        saved_stage_readback_paths=[invocation.parent / "readback.json"],
        native_disposition="passed",
        summary="legacy preparation passed",
    )

    session.begin_leaf("a.publish.v1")
    cancelled_artifacts = _write_native_terminal_artifacts(
        state_path,
        "a.publish.v1",
        disposition="cancelled",
        error="legacy cancellation",
    )
    cancelled = session.cancel_leaf(
        "a.publish.v1",
        reason="legacy cancellation",
        invocation_path=cancelled_artifacts["invocation"],
        result_path=cancelled_artifacts["result"],
        native_terminal_receipt_path=cancelled_artifacts["terminal_receipt"],
        operation_index_paths=[cancelled_artifacts["operation_index"]],
        evidence_index_paths=[cancelled_artifacts["evidence_index"]],
        evidence_paths=[cancelled_artifacts["evidence"]],
        saved_stage_readback_paths=[cancelled_artifacts["readback"]],
        resource_claims=["legacy-fixture-resource"],
        resource_release_paths=[cancelled_artifacts["release"]],
    )
    cancelled_receipt = cancelled.leaf_states["a.publish.v1"].receipt
    assert cancelled_receipt is not None
    session.recover_leaf("a.publish.v1", reason="legacy cancellation cleared")
    session.begin_leaf("a.publish.v1")
    invocation, result = _write_leaf_artifacts(state_path, "a.publish.v1")
    session.complete_leaf(
        "a.publish.v1",
        invocation_path=invocation,
        result_path=result,
        evidence_paths=[invocation.parent / "evidence.json"],
        saved_stage_readback_paths=[invocation.parent / "readback.json"],
        native_disposition="passed",
        summary="legacy publication passed",
    )
    release = state_path.parent / "legacy_parent_release.json"
    atomic_write_json(release, {"status": "released"})
    completed = finalize_graph_run(
        state_path,
        resource_release_paths=[release],
    )

    assert completed.execution_graph == frozen.execution_graph
    assert completed.graph_terminal_receipt is not None
    terminal = json.loads(
        Path(completed.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert terminal["schema_version"] == (
        LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    )
    assert "parent_release_receipt" not in terminal
    assert validate_terminal(state_path).valid is True

    terminal["resource_release_receipts"] = []
    terminal_path = Path(completed.graph_terminal_receipt.path)
    atomic_write_json(terminal_path, terminal)
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["graph_terminal_receipt"] = _binding(terminal_path).model_dump(
        mode="json"
    )
    atomic_write_json(state_path, state_payload)
    with pytest.raises(
        AssetCompositionStateError,
        match="Legacy graph terminal receipt omits parent resource release",
    ):
        load_verified_run(state_path)


@pytest.mark.parametrize("command", ["fail-leaf", "cancel-leaf"])
def test_graph_v1_stop_cli_compatibility_aliases(
    tmp_path: Path,
    command: str,
) -> None:
    descriptor = AssetLeafDescriptor.create(
        leaf_id="legacy.stop.v1",
        entrypoint="legacy stop",
        invocation_schema_digest="1" * 64,
        result_schema_digest="2" * 64,
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[descriptor],
        legacy_catalog=True,
    )
    graph = _legacy_graph(
        request,
        selected=[("legacy.stop.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)

    assert (
        asset_state_main(
            [
                command,
                "--run-state",
                str(state_path),
                "--leaf",
                "legacy.stop.v1",
                "--reason",
                f"compatibility {command}",
            ]
        )
        == 0
    )
    stopped = load_verified_run(state_path)
    receipt = stopped.leaf_states["legacy.stop.v1"].receipt
    assert receipt is not None
    payload = json.loads(Path(receipt.path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION


def test_graph_v1_complete_and_finalize_cli_compatibility_aliases(
    tmp_path: Path,
) -> None:
    descriptor = AssetLeafDescriptor.create(
        leaf_id="legacy.complete.v1",
        entrypoint="legacy complete",
        invocation_schema_digest="1" * 64,
        result_schema_digest="2" * 64,
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[descriptor],
        requires_parent_resource_release=True,
        legacy_catalog=True,
    )
    graph = _legacy_graph(
        request,
        selected=[("legacy.complete.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    assert (
        asset_state_main(
            [
                "begin-leaf",
                "--run-state",
                str(state_path),
                "--leaf",
                "legacy.complete.v1",
            ]
        )
        == 0
    )
    invocation, result = _write_leaf_artifacts(state_path, "legacy.complete.v1")
    assert (
        asset_state_main(
            [
                "complete-leaf",
                "--run-state",
                str(state_path),
                "--leaf",
                "legacy.complete.v1",
                "--invocation",
                str(invocation),
                "--result",
                str(result),
                "--evidence",
                str(invocation.parent / "evidence.json"),
                "--saved-stage-readback",
                str(invocation.parent / "readback.json"),
                "--native-disposition",
                "passed",
                "--summary",
                "legacy CLI completion",
            ]
        )
        == 0
    )
    release = state_path.parent / "legacy_release.json"
    atomic_write_json(release, {"status": "released"})
    assert (
        asset_state_main(
            [
                "finalize-graph",
                "--run-state",
                str(state_path),
                "--resource-release",
                str(release),
            ]
        )
        == 0
    )
    assert load_verified_run(state_path).terminal_status == "completed"


def test_graph_catalog_version_matrix_rejects_upgrade_and_downgrade_before_freeze(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    legacy_state, legacy_request = _create_agentic_run(
        legacy_root,
        legacy_catalog=True,
    )
    assert legacy_request.leaf_catalog is not None
    assert (
        legacy_request.leaf_catalog.schema_version
        == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
    )
    upgraded_graph = _graph(
        legacy_request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    upgraded_path = legacy_state.parent / "upgraded_graph.json"
    atomic_write_json(upgraded_path, upgraded_graph)
    with pytest.raises(
        AssetCompositionStateError,
        match="graph=.*v2, catalog=.*v1",
    ):
        freeze_execution_graph(legacy_state, graph_path=upgraded_path)
    assert load_verified_run(legacy_state).execution_graph is None

    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    repository_state, repository_request = _create_agentic_run(repository_root)
    assert repository_request.leaf_catalog is not None
    assert (
        repository_request.leaf_catalog.schema_version
        == ASSET_LEAF_CATALOG_SCHEMA_VERSION
    )
    downgraded_graph = _legacy_graph(
        repository_request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    downgraded_path = repository_state.parent / "downgraded_graph.json"
    atomic_write_json(downgraded_path, downgraded_graph)
    with pytest.raises(
        AssetCompositionStateError,
        match="graph=.*v1, catalog=.*v3",
    ):
        freeze_execution_graph(repository_state, graph_path=downgraded_path)
    assert load_verified_run(repository_state).execution_graph is None


def test_receipt_and_terminal_parsers_reject_every_mixed_wire_version(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    descriptor = AssetLeafDescriptor.create(
        leaf_id="legacy.complete.v1",
        entrypoint="legacy complete",
        invocation_schema_digest="1" * 64,
        result_schema_digest="2" * 64,
    )
    legacy_state, legacy_request = _create_agentic_run(
        legacy_root,
        descriptors=[descriptor],
        legacy_catalog=True,
    )
    legacy_graph = _legacy_graph(
        legacy_request,
        selected=[("legacy.complete.v1", [], "required", True)],
        omitted=[],
    )
    legacy_graph_path = legacy_state.parent / "execution_graph.json"
    atomic_write_json(legacy_graph_path, legacy_graph)
    freeze_execution_graph(legacy_state, graph_path=legacy_graph_path)
    begin_leaf(legacy_state, "legacy.complete.v1")
    invocation, result = _write_leaf_artifacts(
        legacy_state,
        "legacy.complete.v1",
    )
    legacy_completed = complete_leaf(
        legacy_state,
        "legacy.complete.v1",
        invocation_path=invocation,
        result_path=result,
        evidence_paths=[invocation.parent / "evidence.json"],
        saved_stage_readback_paths=[invocation.parent / "readback.json"],
        native_disposition="passed",
        summary="legacy completion",
    )
    legacy_receipt_binding = legacy_completed.leaf_states["legacy.complete.v1"].receipt
    assert legacy_receipt_binding is not None
    legacy_receipt = json.loads(
        Path(legacy_receipt_binding.path).read_text(encoding="utf-8")
    )
    legacy_terminal_run = finalize_graph_run(legacy_state)
    assert legacy_terminal_run.graph_terminal_receipt is not None
    legacy_terminal = json.loads(
        Path(legacy_terminal_run.graph_terminal_receipt.path).read_text(
            encoding="utf-8"
        )
    )

    current_root = tmp_path / "current"
    current_root.mkdir()
    current_state, current_request = _create_agentic_run(current_root)
    current_graph = _graph(
        current_request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    current_graph_path = current_state.parent / "execution_graph.json"
    atomic_write_json(current_graph_path, current_graph)
    freeze_execution_graph(current_state, graph_path=current_graph_path)
    _complete_selected_leaf(current_state, "inspect.v1")
    current_receipt_binding = (
        load_verified_run(current_state).leaf_states["inspect.v1"].receipt
    )
    assert current_receipt_binding is not None
    current_receipt = json.loads(
        Path(current_receipt_binding.path).read_text(encoding="utf-8")
    )
    current_terminal_run = _finalize_v2_graph(current_state)
    assert current_terminal_run.graph_terminal_receipt is not None
    current_terminal = json.loads(
        Path(current_terminal_run.graph_terminal_receipt.path).read_text(
            encoding="utf-8"
        )
    )

    legacy_receipt_with_v2 = dict(legacy_receipt)
    legacy_receipt_with_v2["descriptor_digest"] = "a" * 64
    with pytest.raises(ValidationError, match="legacy leaf receipt carries v2"):
        AssetLeafReceipt.model_validate(legacy_receipt_with_v2)

    current_receipt_as_v1 = dict(current_receipt)
    current_receipt_as_v1["schema_version"] = LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    with pytest.raises(ValidationError, match="legacy leaf receipt carries v2"):
        AssetLeafReceipt.model_validate(current_receipt_as_v1)

    legacy_receipt_as_v2 = dict(legacy_receipt)
    legacy_receipt_as_v2["schema_version"] = ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    with pytest.raises(ValidationError, match="v2 leaf receipt omits required fields"):
        AssetLeafReceipt.model_validate(legacy_receipt_as_v2)

    current_receipt_missing_v2 = dict(current_receipt)
    current_receipt_missing_v2.pop("projector_digest")
    with pytest.raises(ValidationError, match="v2 leaf receipt omits required fields"):
        AssetLeafReceipt.model_validate(current_receipt_missing_v2)

    current_receipt_default_v2_missing_field = dict(current_receipt_missing_v2)
    current_receipt_default_v2_missing_field.pop("schema_version")
    with pytest.raises(ValidationError, match="v2 leaf receipt omits required fields"):
        AssetLeafReceipt.model_validate(current_receipt_default_v2_missing_field)

    legacy_terminal_with_v2 = dict(legacy_terminal)
    legacy_terminal_with_v2["parent_release_receipt"] = legacy_receipt["invocation"]
    with pytest.raises(
        ValidationError,
        match="legacy graph terminal receipt carries v2",
    ):
        AssetGraphTerminalReceipt.model_validate(legacy_terminal_with_v2)

    current_terminal_as_v1 = dict(current_terminal)
    current_terminal_as_v1["schema_version"] = (
        LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    )
    with pytest.raises(
        ValidationError,
        match="legacy graph terminal receipt carries v2",
    ):
        AssetGraphTerminalReceipt.model_validate(current_terminal_as_v1)

    legacy_terminal_as_v2 = dict(legacy_terminal)
    legacy_terminal_as_v2["schema_version"] = (
        ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    )
    with pytest.raises(
        ValidationError,
        match="v2 graph terminal receipt omits required fields",
    ):
        AssetGraphTerminalReceipt.model_validate(legacy_terminal_as_v2)

    current_terminal_missing_v2 = dict(current_terminal)
    current_terminal_missing_v2.pop("parent_command_receipt_checkpoint")
    with pytest.raises(
        ValidationError,
        match="v2 graph terminal receipt omits required fields",
    ):
        AssetGraphTerminalReceipt.model_validate(current_terminal_missing_v2)

    current_terminal_default_v2_missing_field = dict(current_terminal_missing_v2)
    current_terminal_default_v2_missing_field.pop("schema_version")
    with pytest.raises(
        ValidationError,
        match="v2 graph terminal receipt omits required fields",
    ):
        AssetGraphTerminalReceipt.model_validate(
            current_terminal_default_v2_missing_field
        )


def test_graph_v2_rejects_cross_version_dependency_receipt_before_transition(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[
            ("inspect.v1", [], "required", False),
            ("publish.v1", ["inspect.v1"], "required", True),
        ],
        omitted=["optional-check.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    _complete_selected_leaf(state_path, "inspect.v1")

    run = load_verified_run(state_path)
    receipt_binding = run.leaf_states["inspect.v1"].receipt
    assert receipt_binding is not None
    receipt_path = Path(receipt_binding.path)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt_payload["schema_version"] = LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    for field in (
        "descriptor_digest",
        "invocation_schema_digest",
        "result_schema_digest",
        "projection_schema_digest",
        "projector_id",
        "projector_digest",
        "required_artifact_categories",
        "projection",
        "native_status",
    ):
        receipt_payload.pop(field)
    atomic_write_json(receipt_path, receipt_payload)
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    state_payload["leaf_states"]["inspect.v1"]["receipt"] = _binding(
        receipt_path
    ).model_dump(mode="json")
    atomic_write_json(state_path, state_payload)

    before = state_path.read_bytes()
    with pytest.raises(
        AssetCompositionStateError,
        match="Leaf receipt identity changed for inspect.v1",
    ):
        begin_leaf(state_path, "publish.v1")
    assert state_path.read_bytes() == before


def test_graph_v2_cli_rejects_legacy_and_mixed_flags(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "inspect.v1")
    invocation, result = _write_leaf_artifacts(state_path, "inspect.v1")

    assert (
        asset_state_main(
            [
                "complete-leaf",
                "--run-state",
                str(state_path),
                "--leaf",
                "inspect.v1",
                "--invocation",
                str(invocation),
                "--result",
                str(result),
                "--summary",
                "legacy summary",
            ]
        )
        == 2
    )
    assert "rejects compatibility-only artifact flags" in capsys.readouterr().err

    for command in ("fail-leaf", "cancel-leaf"):
        assert (
            asset_state_main(
                [
                    command,
                    "--run-state",
                    str(state_path),
                    "--leaf",
                    "inspect.v1",
                    "--reason",
                    "missing typed result",
                ]
            )
            == 2
        )
        assert "requires exact invocation and result" in capsys.readouterr().err

    complete_leaf(
        state_path,
        "inspect.v1",
        invocation_path=invocation,
        result_path=result,
    )
    release = state_path.parent / "parent_release.json"
    journal = state_path.parent / "parent_receipt_journal.jsonl"
    checkpoint = state_path.parent / "parent_receipt_checkpoint.json"
    legacy_release = state_path.parent / "legacy_release.json"
    atomic_write_json(release, {"status": "released"})
    journal.write_text('{"command":"fixture"}\n', encoding="utf-8")
    atomic_write_json(checkpoint, {"status": "sealed"})
    atomic_write_json(legacy_release, {"status": "released"})
    assert (
        asset_state_main(
            [
                "finalize-graph",
                "--run-state",
                str(state_path),
                "--parent-release-receipt",
                str(release),
                "--parent-command-receipt-journal",
                str(journal),
                "--parent-command-receipt-checkpoint",
                str(checkpoint),
                "--resource-release",
                str(legacy_release),
            ]
        )
        == 2
    )
    assert "graph v2 finalization is launcher-owned" in capsys.readouterr().err
    assert load_verified_run(state_path).coordinator.next_action == "finalize_receipts"


def test_outer_graph_controls_selective_order_omission_and_terminal_receipt(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[
            ("inspect.v1", [], "required", False),
            ("publish.v1", ["inspect.v1"], "required", True),
        ],
        omitted=["optional-check.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    frozen = freeze_execution_graph(state_path, graph_path=graph_path)

    assert graph.schema_version == ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
    assert frozen.current_leaf_id is None
    assert list(frozen.leaf_states) == ["inspect.v1", "publish.v1"]
    _complete_selected_leaf(state_path, "inspect.v1")
    inspect_receipt_binding = (
        load_verified_run(state_path).leaf_states["inspect.v1"].receipt
    )
    assert inspect_receipt_binding is not None
    inspect_receipt = AssetLeafReceipt.model_validate_json(
        Path(inspect_receipt_binding.path).read_text(encoding="utf-8")
    )
    assert inspect_receipt.schema_version == ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    assert inspect_receipt.projection is not None
    projection = AssetLeafProjection.model_validate_json(
        Path(inspect_receipt.projection.path).read_text(encoding="utf-8")
    )
    assert projection.context.invocation_artifact == inspect_receipt.invocation
    assert projection.context.result_artifact == inspect_receipt.result
    ready = load_verified_run(state_path)
    assert ready.current_leaf_id is None
    assert ready.leaf_states["publish.v1"].status == "ready"
    _complete_selected_leaf(state_path, "publish.v1")
    completed = _finalize_v2_graph(state_path)

    assert completed.terminal_status == "completed"
    receipt = json.loads(
        Path(completed.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert receipt["schema_version"] == ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    assert receipt["parent_release_receipt"] is not None
    assert receipt["parent_command_receipt_journal"] is not None
    assert receipt["parent_command_receipt_checkpoint"] is not None
    assert list(receipt["selected_leaf_receipts"]) == ["inspect.v1", "publish.v1"]
    assert receipt["omitted_leaf_dispositions"] == {
        "optional-check.v1": "not_requested"
    }
    validation = validate_terminal(state_path)
    assert validation.valid is True
    assert validation.graph == completed.execution_graph


def test_agentic_run_state_rejects_fixed_transitions_and_unfrozen_receipt(
    tmp_path: Path,
) -> None:
    state_path, _request = _create_agentic_run(tmp_path)
    pristine = json.loads(state_path.read_text(encoding="utf-8"))

    fixed_transition = json.loads(json.dumps(pristine))
    fixed_transition["transitions"] = [
        {
            "timestamp": "2026-08-17T00:00:01Z",
            "stage": "articulation",
            "from_status": "pending",
            "to_status": "ready",
            "reason": "Stale fixed-stage history.",
            "actor": "test",
            "attempt_count": 0,
            "input_sha256": None,
            "output_sha256": None,
        }
    ]
    atomic_write_json(state_path, fixed_transition)
    with pytest.raises(
        AssetCompositionStateError,
        match="agentic graph run cannot contain fixed stage transitions",
    ):
        asset_state.load_run_state(state_path)

    unfrozen_receipt = json.loads(json.dumps(pristine))
    unfrozen_receipt["graph_terminal_receipt"] = pristine["request"]
    atomic_write_json(state_path, unfrozen_receipt)
    with pytest.raises(
        AssetCompositionStateError,
        match="unfrozen graph run cannot have terminal receipt",
    ):
        asset_state.load_run_state(state_path)


def test_selected_optional_leaf_records_not_evaluated_without_reordering(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[
            ("inspect.v1", [], "required", False),
            ("optional-check.v1", [], "optional", False),
            ("publish.v1", ["inspect.v1"], "required", True),
        ],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)

    _complete_selected_leaf(state_path, "inspect.v1")
    _complete_selected_leaf(
        state_path,
        "optional-check.v1",
        disposition="not_evaluated",
    )
    _complete_selected_leaf(state_path, "publish.v1")
    completed = _finalize_v2_graph(state_path)

    assert completed.leaf_states["optional-check.v1"].status == "not_evaluated"
    assert validate_terminal(state_path).valid is True


@pytest.mark.parametrize(
    ("requirement", "with_dependent", "expected_error"),
    [
        ("required", False, "Required leaf check.v1 cannot be not_evaluated"),
        (
            "optional",
            True,
            "not_evaluated leaf check.v1 is required by another selected leaf",
        ),
    ],
)
def test_rejected_not_evaluated_completion_leaves_no_orphan_and_can_retry(
    tmp_path: Path,
    requirement: str,
    with_dependent: bool,
    expected_error: str,
) -> None:
    descriptors = [_descriptor("check.v1")]
    selected = [("check.v1", [], requirement, not with_dependent)]
    if with_dependent:
        descriptors.append(_descriptor("consumer.v1"))
        selected.append(("consumer.v1", ["check.v1"], "required", True))
    state_path, request = _create_agentic_run(tmp_path, descriptors=descriptors)
    graph = _graph(request, selected=selected, omitted=[])
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "check.v1")
    invocation, result = _write_leaf_artifacts(
        state_path,
        "check.v1",
        include_readback=False,
        disposition="not_evaluated",
    )
    first_attempt = leaf_directory(state_path, "check.v1")

    with pytest.raises(AssetCompositionStateError, match=expected_error):
        complete_leaf(
            state_path,
            "check.v1",
            invocation_path=invocation,
            result_path=result,
        )

    running = load_verified_run(state_path)
    assert running.leaf_states["check.v1"].status == "running"
    assert running.leaf_states["check.v1"].receipt is None
    assert not (first_attempt / "leaf_projection.json").exists()

    reason = "Rejected native result cannot satisfy the frozen graph."
    stopped_artifacts = _write_native_terminal_artifacts(
        state_path,
        "check.v1",
        disposition="failed",
        error=reason,
    )
    failed = fail_leaf(
        state_path,
        "check.v1",
        reason=reason,
        invocation_path=stopped_artifacts["invocation"],
        result_path=stopped_artifacts["result"],
    )
    assert failed.leaf_states["check.v1"].status == "failed"
    assert (first_attempt / "leaf_projection.json").is_file()

    recovered = recover_leaf(
        state_path,
        "check.v1",
        reason="Retry with a fresh native result.",
    )
    assert recovered.leaf_states["check.v1"].status == "ready"
    _complete_selected_leaf(state_path, "check.v1")
    retried = load_verified_run(state_path)
    assert retried.leaf_states["check.v1"].status == "passed"
    assert retried.leaf_states["check.v1"].attempt_count == 2


def test_graph_rejects_unknown_missing_duplicate_cyclic_incompatible_and_stale(
    tmp_path: Path,
) -> None:
    descriptors = [
        _descriptor("a.v1", incompatible_leaf_ids=["b.v1"]),
        _descriptor("b.v1"),
        _descriptor("c.v1", required_dependencies=["a.v1"]),
    ]
    state_path, request = _create_agentic_run(tmp_path, descriptors=descriptors)
    assert request.leaf_catalog is not None
    mapping = {
        descriptor.leaf_id: descriptor
        for descriptor in request.leaf_catalog.descriptors
    }

    with pytest.raises(ValidationError, match="duplicate selected leaf IDs"):
        AssetExecutionGraph.create(
            sole_coordinator_identity_digest=(
                request.sole_coordinator_identity.identity_digest  # type: ignore[union-attr]
            ),
            prompt_digest=str(request.prompt_digest),
            source_digest=str(request.source_digest),
            configuration_digest=str(request.configuration_digest),
            reference_digest=str(request.reference_digest),
            leaf_catalog_digest=request.leaf_catalog.catalog_digest,
            nodes=[
                AssetExecutionNode(
                    leaf_id="a.v1",
                    requirement="required",
                    descriptor_digest=mapping["a.v1"].descriptor_digest,
                    terminal_output=True,
                ),
                AssetExecutionNode(
                    leaf_id="a.v1",
                    requirement="required",
                    descriptor_digest=mapping["a.v1"].descriptor_digest,
                ),
            ],
            omitted_leaf_ids=["b.v1", "c.v1"],
        )

    with pytest.raises(ValidationError, match="must be acyclic"):
        _graph(
            request,
            selected=[
                ("a.v1", ["b.v1"], "required", True),
                ("b.v1", ["a.v1"], "required", False),
            ],
            omitted=["c.v1"],
        )

    incompatible = _graph(
        request,
        selected=[
            ("a.v1", [], "required", False),
            ("b.v1", [], "required", True),
        ],
        omitted=["c.v1"],
    )
    graph_path = state_path.parent / "incompatible.json"
    atomic_write_json(graph_path, incompatible)
    with pytest.raises(AssetCompositionStateError, match="incompatible"):
        freeze_execution_graph(state_path, graph_path=graph_path)

    stale = _graph(
        request,
        selected=[("a.v1", [], "required", True)],
        omitted=["b.v1", "c.v1"],
    ).model_dump(mode="json")
    stale["prompt_digest"] = "f" * 64
    stale["graph_digest"] = canonical_asset_digest(
        {key: value for key, value in stale.items() if key != "graph_digest"}
    )
    atomic_write_json(graph_path, stale)
    with pytest.raises(AssetCompositionStateError, match="identity differs"):
        freeze_execution_graph(state_path, graph_path=graph_path)

    missing_dependency = _graph(
        request,
        selected=[
            ("a.v1", [], "required", False),
            ("c.v1", [], "required", True),
        ],
        omitted=["b.v1"],
    )
    atomic_write_json(graph_path, missing_dependency)
    with pytest.raises(AssetCompositionStateError, match="declared dependencies"):
        freeze_execution_graph(state_path, graph_path=graph_path)


def test_graph_enforces_request_required_leaf_and_terminal_contract(
    tmp_path: Path,
) -> None:
    def create_case(
        name: str,
        *,
        required_leaf_ids: list[str],
        required_terminal_leaf_ids: list[str],
        required_leaf_dependencies: dict[str, list[str]] | None = None,
    ) -> tuple[Path, AssetRunRequest]:
        case_path = tmp_path / name
        case_path.mkdir()
        return _create_agentic_run(
            case_path,
            required_leaf_ids=required_leaf_ids,
            required_terminal_leaf_ids=required_terminal_leaf_ids,
            required_leaf_dependencies=required_leaf_dependencies,
        )

    state_path, request = create_case(
        "omitted",
        required_leaf_ids=["publish.v1"],
        required_terminal_leaf_ids=["publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[("inspect.v1", [], "required", True)],
            omitted=["optional-check.v1", "publish.v1"],
        ),
    )
    with pytest.raises(AssetCompositionStateError, match="omits request-required"):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case(
        "optional",
        required_leaf_ids=["publish.v1"],
        required_terminal_leaf_ids=["publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", False),
                ("publish.v1", ["inspect.v1"], "optional", True),
            ],
            omitted=["optional-check.v1"],
        ),
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="marks request-required leaves optional",
    ):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case(
        "missing-terminal",
        required_leaf_ids=["publish.v1"],
        required_terminal_leaf_ids=["publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", True),
                ("publish.v1", ["inspect.v1"], "required", False),
            ],
            omitted=["optional-check.v1"],
        ),
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="does not mark request-required terminal leaves",
    ):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case(
        "extra-terminal",
        required_leaf_ids=["inspect.v1", "publish.v1"],
        required_terminal_leaf_ids=["publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", True),
                ("publish.v1", ["inspect.v1"], "required", True),
            ],
            omitted=["optional-check.v1"],
        ),
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="terminal outputs outside the frozen request",
    ):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case(
        "missing-required-edge",
        required_leaf_ids=["inspect.v1", "publish.v1"],
        required_terminal_leaf_ids=["publish.v1"],
        required_leaf_dependencies={"publish.v1": ["inspect.v1"]},
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", False),
                ("publish.v1", [], "required", True),
            ],
            omitted=["optional-check.v1"],
        ),
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="request-required dependency edges",
    ):
        freeze_execution_graph(state_path, graph_path=graph_path)


def test_graph_exact_prompt_scope_rejects_over_selection_and_missing_rationale(
    tmp_path: Path,
) -> None:
    required = ["inspect.v1", "publish.v1"]

    def create_case(name: str) -> tuple[Path, AssetRunRequest]:
        case_path = tmp_path / name
        case_path.mkdir()
        return _create_agentic_run(
            case_path,
            required_leaf_ids=required,
            required_terminal_leaf_ids=["publish.v1"],
            required_leaf_dependencies={"publish.v1": ["inspect.v1"]},
            exact_leaf_scope=True,
        )

    state_path, request = create_case("over-selected")
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", False),
                ("optional-check.v1", [], "optional", False),
                ("publish.v1", ["inspect.v1"], "required", True),
            ],
            omitted=[],
            selection_rationales={
                "inspect.v1": "Inspects the prompt target.",
                "optional-check.v1": "Present in the catalog.",
                "publish.v1": "Publishes the requested result.",
            },
        ),
    )
    with pytest.raises(AssetCompositionStateError, match="over-selects leaves"):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case("missing-rationale")
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", False),
                ("publish.v1", ["inspect.v1"], "required", True),
            ],
            omitted=["optional-check.v1"],
            selection_rationales={"inspect.v1": "Inspects the prompt target."},
        ),
    )
    with pytest.raises(AssetCompositionStateError, match="relevance rationale"):
        freeze_execution_graph(state_path, graph_path=graph_path)

    state_path, request = create_case("exact")
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[
                ("inspect.v1", [], "required", False),
                ("publish.v1", ["inspect.v1"], "required", True),
            ],
            omitted=["optional-check.v1"],
            selection_rationales={
                "inspect.v1": "Inspects the prompt target.",
                "publish.v1": "Publishes the requested result.",
            },
        ),
    )
    frozen = freeze_execution_graph(state_path, graph_path=graph_path)
    assert frozen.execution_graph == _binding(graph_path)


def test_direct_interactive_api_cannot_forge_launcher_graph_finalization(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[
            ("inspect.v1", [], "required", False),
            ("publish.v1", ["inspect.v1"], "required", True),
        ],
        omitted=["optional-check.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    prompt_invocations = 0

    def one_prompt(session):  # type: ignore[no-untyped-def]
        nonlocal prompt_invocations
        prompt_invocations += 1
        session.freeze_graph(graph_path)
        for leaf_id in graph.selected_leaf_ids:
            session.begin_leaf(leaf_id)
            invocation, result = _write_leaf_artifacts(state_path, leaf_id)
            session.complete_leaf(
                leaf_id,
                invocation_path=invocation,
                result_path=result,
            )
        root = state_path.parent
        release = root / "interactive_parent_release.json"
        journal = root / "interactive_parent_journal.jsonl"
        checkpoint = root / "interactive_parent_checkpoint.json"
        atomic_write_json(release, {"status": "released"})
        journal.write_text('{"command":"fixture"}\n', encoding="utf-8")
        atomic_write_json(checkpoint, {"status": "sealed"})
        with pytest.raises(
            AssetCompositionStateError,
            match="launcher-bound expected parent receipt identities",
        ):
            session.finalize_graph(
                parent_release_receipt_path=release,
                parent_command_receipt_journal_path=journal,
                parent_command_receipt_checkpoint_path=checkpoint,
            )
        return 0

    result = run_interactive_asset_coordinator(
        state_path,
        reasoning_loop=one_prompt,
    )

    assert prompt_invocations == 1
    assert result.run.terminal_status == "active"
    assert result.run.coordinator.next_action == "finalize_receipts"
    assert result.returncode == 0
    assert not (state_path.parent / "graph_terminal_receipt.json").exists()


def test_graph_enforces_selected_required_dependent_without_global_order(
    tmp_path: Path,
) -> None:
    descriptors = [
        _descriptor("author.v1", required_dependents=["render.v1"]),
        _descriptor("render.v1"),
        _descriptor("unrelated.v1"),
    ]

    (tmp_path / "missing").mkdir()
    missing_state, missing_request = _create_agentic_run(
        tmp_path / "missing",
        descriptors=descriptors,
    )
    missing_graph = _graph(
        missing_request,
        selected=[
            ("author.v1", [], "required", False),
            ("render.v1", [], "required", True),
        ],
        omitted=["unrelated.v1"],
    )
    missing_path = missing_state.parent / "execution_graph.json"
    atomic_write_json(missing_path, missing_graph)
    with pytest.raises(
        AssetCompositionStateError,
        match="omits declared producer dependency.*render.v1.*author.v1",
    ):
        freeze_execution_graph(missing_state, graph_path=missing_path)

    (tmp_path / "ordered").mkdir()
    ordered_state, ordered_request = _create_agentic_run(
        tmp_path / "ordered",
        descriptors=descriptors,
    )
    ordered_graph = _graph(
        ordered_request,
        selected=[
            ("author.v1", [], "required", False),
            ("render.v1", ["author.v1"], "required", True),
        ],
        omitted=["unrelated.v1"],
    )
    ordered_path = ordered_state.parent / "execution_graph.json"
    atomic_write_json(ordered_path, ordered_graph)
    ordered_run = freeze_execution_graph(ordered_state, graph_path=ordered_path)
    assert ordered_run.execution_graph == _binding(ordered_path)

    (tmp_path / "omitted").mkdir()
    omitted_state, omitted_request = _create_agentic_run(
        tmp_path / "omitted",
        descriptors=descriptors,
    )
    omitted_graph = _graph(
        omitted_request,
        selected=[("author.v1", [], "required", True)],
        omitted=["render.v1", "unrelated.v1"],
    )
    omitted_path = omitted_state.parent / "execution_graph.json"
    atomic_write_json(omitted_path, omitted_graph)
    omitted_run = freeze_execution_graph(omitted_state, graph_path=omitted_path)
    assert omitted_run.execution_graph == _binding(omitted_path)


def test_parent_release_requirement_fails_closed_until_receipt_is_bound(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(
        tmp_path,
        requires_parent_resource_release=True,
    )
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    _complete_selected_leaf(state_path, "inspect.v1")

    with pytest.raises(
        AssetCompositionStateError,
        match="requires launcher release receipt, journal, and checkpoint",
    ):
        finalize_graph_run(state_path)
    release = state_path.parent / "parent_release.json"
    journal = state_path.parent / "parent_receipt_journal.jsonl"
    checkpoint = state_path.parent / "parent_receipt_checkpoint.json"
    atomic_write_json(release, {"status": "released"})
    journal.write_text('{"command":"fixture"}\n', encoding="utf-8")
    atomic_write_json(checkpoint, {"status": "sealed"})
    with pytest.raises(
        AssetCompositionStateError,
        match="launcher-bound expected parent receipt identities",
    ):
        finalize_graph_run(
            state_path,
            parent_release_receipt_path=release,
            parent_command_receipt_journal_path=journal,
            parent_command_receipt_checkpoint_path=checkpoint,
        )
    assert not (state_path.parent / "graph_terminal_receipt.json").exists()
    completed = finalize_graph_run(
        state_path,
        parent_release_receipt_path=release,
        parent_command_receipt_journal_path=journal,
        parent_command_receipt_checkpoint_path=checkpoint,
        expected_parent_release_receipt=_binding(release),
        expected_parent_command_receipt_journal=_binding(journal),
        expected_parent_command_receipt_checkpoint=_binding(checkpoint),
    )
    assert completed.terminal_status == "completed"


def test_resume_supersedes_failed_attempt_without_graph_or_order_change(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    frozen = freeze_execution_graph(state_path, graph_path=graph_path)
    frozen_graph = frozen.execution_graph
    begin_leaf(state_path, "inspect.v1")
    failure_reason = "Bounded fixture failure."
    failure_artifacts = _write_native_terminal_artifacts(
        state_path,
        "inspect.v1",
        disposition="failed",
        error=failure_reason,
    )
    failed = fail_leaf(
        state_path,
        "inspect.v1",
        reason=failure_reason,
        invocation_path=failure_artifacts["invocation"],
        result_path=failure_artifacts["result"],
    )
    failed_receipt = failed.leaf_states["inspect.v1"].receipt
    assert failed_receipt is not None

    recovered = recover_leaf(
        state_path,
        "inspect.v1",
        reason="The bounded fixture failure was corrected.",
    )
    assert recovered.execution_graph == frozen_graph
    assert recovered.leaf_states["inspect.v1"].superseded_receipts == [failed_receipt]
    _complete_selected_leaf(state_path, "inspect.v1")
    completed = _finalize_v2_graph(state_path)
    current_receipt = completed.leaf_states["inspect.v1"].receipt
    assert current_receipt is not None
    current_payload = json.loads(Path(current_receipt.path).read_text(encoding="utf-8"))
    assert current_payload["supersedes"] == failed_receipt.model_dump(mode="json")
    assert completed.execution_graph == frozen_graph


def test_graph_storage_is_lexical_without_imposing_execution_order(
    tmp_path: Path,
) -> None:
    descriptors = [
        _descriptor("z-first.v1"),
        _descriptor("b-parallel.v1"),
        _descriptor(
            "a-second.v1",
            required_dependencies=["b-parallel.v1", "z-first.v1"],
        ),
        _descriptor("m-omitted.v1"),
    ]
    state_path, request = _create_agentic_run(tmp_path, descriptors=descriptors)
    graph = _graph(
        request,
        selected=[
            ("z-first.v1", [], "required", False),
            ("b-parallel.v1", [], "required", False),
            (
                "a-second.v1",
                ["z-first.v1", "b-parallel.v1"],
                "required",
                True,
            ),
        ],
        omitted=["m-omitted.v1"],
    )
    equivalent = _graph(
        request,
        selected=[
            (
                "a-second.v1",
                ["b-parallel.v1", "z-first.v1"],
                "required",
                True,
            ),
            ("b-parallel.v1", [], "required", False),
            ("z-first.v1", [], "required", False),
        ],
        omitted=["m-omitted.v1"],
    )
    assert equivalent == graph
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    _complete_selected_leaf(state_path, "z-first.v1")
    _complete_selected_leaf(state_path, "b-parallel.v1")
    _complete_selected_leaf(state_path, "a-second.v1")
    completed = _finalize_v2_graph(state_path)

    terminal = json.loads(
        Path(completed.graph_terminal_receipt.path).read_text(encoding="utf-8")
    )
    assert terminal["selected_leaf_ids"] == [
        "a-second.v1",
        "b-parallel.v1",
        "z-first.v1",
    ]
    assert list(terminal["selected_leaf_receipts"]) == [
        "a-second.v1",
        "b-parallel.v1",
        "z-first.v1",
    ]
    assert load_verified_run(state_path).terminal_status == "completed"


def test_resume_rejects_frozen_graph_digest_drift_and_fixed_transitions(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)

    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    with pytest.raises(
        AssetCompositionStateError,
        match="compatibility_fixed transition is unavailable in agentic mode",
    ):
        session.begin_stage("articulation")
    with pytest.raises(
        AssetCompositionStateError,
        match="Agentic request cannot enter legacy fixed compatibility mode",
    ):
        create_run(
            state_path.parent / "mixed_mode_asset_run.json",
            run_id=request.run_id,
            request_path=state_path.parent / "request.json",
            source_asset=state_path.parent / "source.usda",
            coordinator_mode="legacy",
        )

    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    payload["nodes"][0]["terminal_output"] = False
    atomic_write_json(graph_path, payload)
    with pytest.raises(AssetCompositionStateError, match="graph identity changed"):
        load_verified_run(state_path)


def test_mode_gates_defer_full_verification_to_shared_state_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)

    def reject_duplicate_verification(_path: object) -> None:
        raise AssertionError("mode gate repeated full artifact verification")

    monkeypatch.setattr(
        asset_coordinator, "load_verified_run", reject_duplicate_verification
    )
    monkeypatch.setattr(asset_cli, "load_verified_run", reject_duplicate_verification)

    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    started = session.begin_leaf("inspect.v1")
    assert started.leaf_states["inspect.v1"].status == "running"
    cancellation_reason = (
        "Stop after proving the shared state surface verified the transition."
    )
    cancellation_artifacts = _write_native_terminal_artifacts(
        state_path,
        "inspect.v1",
        disposition="cancelled",
        error=cancellation_reason,
    )
    assert (
        asset_state_main(
            [
                "cancel-leaf",
                "--run-state",
                str(state_path),
                "--leaf",
                "inspect.v1",
                "--reason",
                cancellation_reason,
                "--invocation",
                str(cancellation_artifacts["invocation"]),
                "--result",
                str(cancellation_artifacts["result"]),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert asset_state.load_verified_run(state_path).terminal_status == "cancelled"


def test_terminal_validation_normalizes_missing_superseded_dependencies(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[
            ("inspect.v1", [], "required", False),
            ("publish.v1", ["inspect.v1"], "required", True),
        ],
        omitted=["optional-check.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    _complete_selected_leaf(state_path, "inspect.v1")
    begin_leaf(state_path, "publish.v1")
    failure_reason = "Create a superseded receipt."
    failure_artifacts = _write_native_terminal_artifacts(
        state_path,
        "publish.v1",
        disposition="failed",
        error=failure_reason,
    )
    fail_leaf(
        state_path,
        "publish.v1",
        reason=failure_reason,
        invocation_path=failure_artifacts["invocation"],
        result_path=failure_artifacts["result"],
    )
    recover_leaf(state_path, "publish.v1", reason="Retry the same frozen leaf.")
    _complete_selected_leaf(state_path, "publish.v1")
    _finalize_v2_graph(state_path)

    payload = json.loads(state_path.read_text(encoding="utf-8"))
    dependency_state = payload["leaf_states"]["inspect.v1"]
    dependency_state["status"] = "pending"
    dependency_state["attempt_count"] = 0
    dependency_state["receipt"] = None
    payload["leaf_transitions"] = [
        transition
        for transition in payload["leaf_transitions"]
        if transition["leaf_id"] != "inspect.v1"
    ]
    atomic_write_json(state_path, payload)

    validation = validate_terminal(state_path)
    assert validation.valid is False
    assert any(
        (
            "Dependency receipts are missing for publish.v1" in error
            or "Dependency-ready leaf inspect.v1 remained pending" in error
        )
        for error in validation.errors
    )


def test_terminal_validation_normalizes_receipt_drift_after_verified_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    _complete_selected_leaf(state_path, "inspect.v1")
    completed = _finalize_v2_graph(state_path)
    assert completed.graph_terminal_receipt is not None
    receipt_path = Path(completed.graph_terminal_receipt.path)
    original_load = asset_state.load_verified_run

    def load_then_drift(path: str | Path) -> AssetCompositionRun:
        run = original_load(path)
        receipt_path.write_text('{"tampered":true}\n', encoding="utf-8")
        return run

    monkeypatch.setattr(asset_state, "load_verified_run", load_then_drift)

    validation = validate_terminal(state_path)
    assert validation.valid is False
    assert any("Invalid graph terminal receipt" in error for error in validation.errors)


def test_interactive_failure_preserves_exact_native_terminal_artifacts(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    session = AssetCoordinatorSession(run_state_path=state_path, mode="interactive")
    session.freeze_graph(graph_path)
    session.begin_leaf("inspect.v1")
    failure_reason = "Opaque leaf returned its native non-accepting result."
    artifacts = _write_native_terminal_artifacts(
        state_path,
        "inspect.v1",
        disposition="failed",
        error=failure_reason,
    )

    before_rejected_compatibility = state_path.read_bytes()
    with pytest.raises(
        AssetCompositionStateError,
        match="Graph v2 stop transition rejects compatibility-only artifact flags",
    ):
        session.fail_leaf(
            "inspect.v1",
            reason=failure_reason,
            invocation_path=artifacts["invocation"],
            result_path=artifacts["result"],
            native_terminal_receipt_path=artifacts["terminal_receipt"],
        )
    assert state_path.read_bytes() == before_rejected_compatibility

    failed = session.fail_leaf(
        "inspect.v1",
        reason=failure_reason,
        invocation_path=artifacts["invocation"],
        result_path=artifacts["result"],
    )

    state = failed.leaf_states["inspect.v1"]
    assert failed.terminal_status == "failed"
    assert state.status == "failed"
    assert state.receipt is not None
    receipt = json.loads(Path(state.receipt.path).read_text(encoding="utf-8"))
    assert receipt["native_disposition"] == "failed"
    assert receipt["result"] == _binding(artifacts["result"]).model_dump(mode="json")
    assert receipt["native_terminal_receipt"] == _binding(
        artifacts["terminal_receipt"]
    ).model_dump(mode="json")
    assert receipt["operation_indexes"] == [
        _binding(artifacts["operation_index"]).model_dump(mode="json")
    ]
    assert receipt["evidence_indexes"] == [
        _binding(artifacts["evidence_index"]).model_dump(mode="json")
    ]
    assert receipt["evidence"] == [
        _binding(artifacts["evidence"]).model_dump(mode="json")
    ]
    assert receipt["saved_stage_readbacks"] == [
        _binding(artifacts["readback"]).model_dump(mode="json")
    ]
    assert validate_terminal(state_path).valid is False
    with pytest.raises(
        AssetCompositionStateError,
        match="not awaiting comprehensive receipt finalization",
    ):
        finalize_graph_run(state_path)

    recovered = session.recover_leaf(
        "inspect.v1",
        reason="Retry the same frozen leaf without changing its graph identity.",
    )
    assert recovered.leaf_states["inspect.v1"].superseded_receipts == [state.receipt]
    artifacts["operation_index"].write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(AssetCompositionStateError, match="operation index.*changed"):
        load_verified_run(state_path)


def test_cli_cancellation_preserves_native_disposition_without_upgrade(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path, request = _create_agentic_run(tmp_path)
    graph = _graph(
        request,
        selected=[("inspect.v1", [], "required", True)],
        omitted=["optional-check.v1", "publish.v1"],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "inspect.v1")
    cancellation_reason = "Opaque leaf returned its native cancellation."
    artifacts = _write_native_terminal_artifacts(
        state_path,
        "inspect.v1",
        disposition="cancelled",
        error=cancellation_reason,
    )

    returncode = asset_state_main(
        [
            "cancel-leaf",
            "--run-state",
            str(state_path),
            "--leaf",
            "inspect.v1",
            "--reason",
            cancellation_reason,
            "--invocation",
            str(artifacts["invocation"]),
            "--result",
            str(artifacts["result"]),
        ]
    )

    assert returncode == 0
    capsys.readouterr()
    cancelled = load_verified_run(state_path)
    state = cancelled.leaf_states["inspect.v1"]
    assert cancelled.terminal_status == "cancelled"
    assert state.status == "cancelled"
    assert state.receipt is not None
    receipt = json.loads(Path(state.receipt.path).read_text(encoding="utf-8"))
    assert receipt["native_disposition"] == "cancelled"
    assert receipt["native_disposition"] not in {"passed", "not_evaluated"}
    assert receipt["native_terminal_receipt"]["sha256"] == file_sha256(
        artifacts["terminal_receipt"]
    )
    assert validate_terminal(state_path).valid is False


def test_dependency_ready_leaves_do_not_gain_an_automatic_successor(
    tmp_path: Path,
) -> None:
    descriptors = [_descriptor("left.v1"), _descriptor("right.v1")]
    state_path, request = _create_agentic_run(tmp_path, descriptors=descriptors)
    graph = _graph(
        request,
        selected=[
            ("right.v1", [], "required", True),
            ("left.v1", [], "required", False),
        ],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    frozen = freeze_execution_graph(state_path, graph_path=graph_path)

    assert frozen.current_leaf_id is None
    assert {
        leaf_id
        for leaf_id, state in frozen.leaf_states.items()
        if state.status == "ready"
    } == {"left.v1", "right.v1"}
    _complete_selected_leaf(state_path, "left.v1")
    observed = load_verified_run(state_path)
    assert observed.current_leaf_id is None
    assert observed.leaf_states["right.v1"].status == "ready"
    assert observed.coordinator.next_action == "begin_leaf"


def test_readbackless_pass_is_category_appropriate_and_projection_is_bound(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[_descriptor("proposal.v1")],
    )
    graph = _graph(
        request,
        selected=[("proposal.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "proposal.v1")
    invocation, result = _write_leaf_artifacts(
        state_path,
        "proposal.v1",
        include_readback=False,
    )
    completed = complete_leaf(
        state_path,
        "proposal.v1",
        invocation_path=invocation,
        result_path=result,
    )
    receipt_binding = completed.leaf_states["proposal.v1"].receipt
    assert receipt_binding is not None
    receipt = json.loads(Path(receipt_binding.path).read_text(encoding="utf-8"))
    projection_binding = ArtifactBinding.model_validate(receipt["projection"])
    projection = json.loads(Path(projection_binding.path).read_text(encoding="utf-8"))

    assert receipt["saved_stage_readbacks"] == []
    assert receipt["native_status"] == "qualified"
    assert projection_binding.sha256 == file_sha256(projection_binding.path)
    assert projection_binding.size_bytes == Path(projection_binding.path).stat().st_size
    assert projection["projection_digest"] == canonical_asset_digest(
        {key: value for key, value in projection.items() if key != "projection_digest"}
    )
    assert projection["payload"]["evidence"][0].keys() == {
        "path",
        "sha256",
        "size_bytes",
    }


@pytest.mark.parametrize(
    ("native_status", "native_disposition"),
    [
        ("succeeded", "passed"),
        ("qualified", "passed"),
        ("not_produced", "not_evaluated"),
    ],
)
def test_successful_native_proposal_and_scoring_statuses_are_not_normalized(
    tmp_path: Path,
    native_status: str,
    native_disposition: str,
) -> None:
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[_descriptor("native-status.v1")],
    )
    graph = _graph(
        request,
        selected=[("native-status.v1", [], "optional", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "native-status.v1")
    invocation, result = _write_leaf_artifacts(
        state_path,
        "native-status.v1",
        include_readback=False,
        disposition=native_disposition,
        native_status=native_status,
    )

    completed = complete_leaf(
        state_path,
        "native-status.v1",
        invocation_path=invocation,
        result_path=result,
    )

    receipt_binding = completed.leaf_states["native-status.v1"].receipt
    assert receipt_binding is not None
    receipt = json.loads(Path(receipt_binding.path).read_text(encoding="utf-8"))
    assert receipt["native_disposition"] == native_disposition
    assert receipt["native_status"] == native_status
    assert receipt["saved_stage_readbacks"] == []


@pytest.mark.parametrize(
    "native_status",
    [
        "provider_failure",
        "invalid_response",
        "input_drift",
        "publication_failure",
        "not_qualified",
    ],
)
def test_failed_native_proposal_and_scoring_statuses_are_not_normalized(
    tmp_path: Path,
    native_status: str,
) -> None:
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[_descriptor("native-failure.v1")],
    )
    graph = _graph(
        request,
        selected=[("native-failure.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "native-failure.v1")
    reason = f"Native terminal status: {native_status}"
    artifacts = _write_native_terminal_artifacts(
        state_path,
        "native-failure.v1",
        disposition="failed",
        native_status=native_status,
        error=reason,
    )

    failed = fail_leaf(
        state_path,
        "native-failure.v1",
        reason=reason,
        invocation_path=artifacts["invocation"],
        result_path=artifacts["result"],
    )

    receipt_binding = failed.leaf_states["native-failure.v1"].receipt
    assert receipt_binding is not None
    receipt = json.loads(Path(receipt_binding.path).read_text(encoding="utf-8"))
    assert receipt["native_disposition"] == "failed"
    assert receipt["native_status"] == native_status
    assert receipt["error"] == reason


@pytest.mark.parametrize("disposition", ["failed", "cancelled"])
def test_canonical_ovrtx_native_stop_retains_typed_terminal_result(
    tmp_path: Path,
    disposition: str,
) -> None:
    runtime_binding = next(
        binding
        for binding in shared_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id == CANONICAL_OVRTX_EVIDENCE_LEAF_ID
    )
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=[runtime_binding],
    )
    graph = _graph(
        request,
        selected=[
            (
                CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
                [],
                "required",
                True,
            )
        ],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, CANONICAL_OVRTX_EVIDENCE_LEAF_ID)
    attempt = leaf_directory(state_path, CANONICAL_OVRTX_EVIDENCE_LEAF_ID)
    native_root = attempt / "canonical-ovrtx"
    native_root.mkdir()
    native_request = native_root / "canonical_visual_request.json"
    native_receipt = native_root / "native_terminal_receipt.json"
    native_evidence = native_root / "native_evidence.json"
    native_readback = native_root / "native_readback.json"
    source = state_path.parent / "source.usda"
    atomic_write_json(
        native_request,
        CanonicalVisualEvidenceRequest(
            source=execution_artifact_binding(source),
            post_mutation_output=execution_artifact_binding(source),
            backend="ovrtx",
            views=("+x+y+z",),
            image_width=64,
            image_height=64,
        ),
    )
    atomic_write_json(
        native_receipt,
        {"status": disposition, "renderer": "ovrtx"},
    )
    atomic_write_json(native_evidence, {"renderer": "ovrtx", "retained": True})
    atomic_write_json(native_readback, {"source_unchanged": True})
    request_binding = _binding(native_request)
    receipt_binding = _binding(native_receipt)
    evidence_binding = _binding(native_evidence)
    readback_binding = _binding(native_readback)
    invocation_path = attempt / "invocation.json"
    result_path = attempt / "result.json"
    equivalent_source = state_path.parent / "unused" / ".." / "source.usda"
    atomic_write_json(
        invocation_path,
        CanonicalOvrtxEvidenceLeafInvocation(
            output_dir=str(native_root),
            post_mutation_usd=str(equivalent_source),
            source_usd=str(equivalent_source),
            backend="ovrtx",
            views=("+x+y+z",),
            image_width=64,
            image_height=64,
        ),
    )
    reason = f"Canonical OVRTX attempt was {disposition}."
    atomic_write_json(
        result_path,
        CanonicalOvrtxEvidenceLeafTerminalResult(
            output_dir=str(native_root),
            native_disposition=disposition,  # type: ignore[arg-type]
            native_status=disposition,
            native_request=request_binding,
            native_terminal_receipt=receipt_binding,
            evidence=(evidence_binding,),
            saved_stage_readbacks=(readback_binding,),
            summary=reason,
            error=reason,
        ),
    )

    transition = fail_leaf if disposition == "failed" else cancel_leaf
    stopped = transition(
        state_path,
        CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
        reason=reason,
        invocation_path=invocation_path,
        result_path=result_path,
    )

    leaf_state = stopped.leaf_states[CANONICAL_OVRTX_EVIDENCE_LEAF_ID]
    assert leaf_state.status == disposition
    assert leaf_state.receipt is not None
    receipt = AssetLeafReceipt.model_validate_json(
        Path(leaf_state.receipt.path).read_text(encoding="utf-8")
    )
    assert receipt.native_disposition == disposition
    assert receipt.native_terminal_receipt == receipt_binding
    assert receipt.evidence == [evidence_binding]
    assert receipt.saved_stage_readbacks == [readback_binding]


@pytest.mark.parametrize(
    ("leaf_id", "disposition"),
    [
        (FOCUSED_VALIDATION_OPERATION_LEAF_ID, "failed"),
        (FOCUSED_VALIDATION_OPERATION_LEAF_ID, "cancelled"),
        (PROVIDED_VALIDATION_INGRESS_LEAF_ID, "failed"),
        (PROVIDED_VALIDATION_INGRESS_LEAF_ID, "cancelled"),
    ],
)
def test_shared_validation_native_stop_retains_typed_terminal_result(
    tmp_path: Path,
    leaf_id: str,
    disposition: str,
) -> None:
    validation_bindings = [
        binding
        for binding in shared_asset_leaf_runtime_bundle().bindings
        if binding.descriptor.leaf_id
        in {
            FOCUSED_VALIDATION_OPERATION_LEAF_ID,
            PROVIDED_VALIDATION_INGRESS_LEAF_ID,
        }
    ]
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=validation_bindings,
    )
    omitted_leaf = (
        PROVIDED_VALIDATION_INGRESS_LEAF_ID
        if leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID
        else FOCUSED_VALIDATION_OPERATION_LEAF_ID
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(
        graph_path,
        _graph(
            request,
            selected=[(leaf_id, [], "required", True)],
            omitted=[omitted_leaf],
        ),
    )
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, leaf_id)
    attempt = leaf_directory(state_path, leaf_id)
    native_root = attempt / "native-validation"
    native_root.mkdir()
    invocation_path = attempt / "invocation.json"
    if leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID:
        invocation = FocusedValidationOperationLeafInvocation(
            output_dir=str(native_root),
            template_name="physics_sane",
        )
        result_type = FocusedValidationOperationLeafResult
    else:
        invocation = ProvidedValidationIngressLeafInvocation(
            envelope_path=str(native_root / "provided-envelope.json"),
            output_dir=str(native_root),
        )
        result_type = ProvidedValidationIngressLeafResult
    atomic_write_json(invocation_path, invocation)
    invocation_binding = _binding(invocation_path)
    native_receipt = native_root / "terminal.json"
    native_evidence = native_root / "evidence.json"
    native_readback = native_root / "readback.json"
    atomic_write_json(native_receipt, {"status": disposition})
    atomic_write_json(native_evidence, {"retained": True})
    atomic_write_json(native_readback, {"source_unchanged": True})
    reason = f"Shared Validation attempt was {disposition}."
    result_model = result_type(
        root=SharedValidationLeafTerminalResult(
            output_dir=str(native_root),
            invocation_artifact=invocation_binding,
            native_disposition=disposition,  # type: ignore[arg-type]
            native_status=disposition,
            native_terminal_receipt=_binding(native_receipt),
            evidence=(_binding(native_evidence),),
            saved_stage_readbacks=(_binding(native_readback),),
            summary=reason,
            error=reason,
        )
    )
    result_path = attempt / "result.json"
    atomic_write_json(result_path, result_model)

    transition = fail_leaf if disposition == "failed" else cancel_leaf
    stopped = transition(
        state_path,
        leaf_id,
        reason=reason,
        invocation_path=invocation_path,
        result_path=result_path,
    )

    leaf_state = stopped.leaf_states[leaf_id]
    assert leaf_state.status == disposition
    assert leaf_state.receipt is not None
    receipt = AssetLeafReceipt.model_validate_json(
        Path(leaf_state.receipt.path).read_text(encoding="utf-8")
    )
    assert receipt.native_disposition == disposition
    assert receipt.native_terminal_receipt == _binding(native_receipt)


def test_descriptor_required_readback_and_stale_artifacts_fail_before_transition(
    tmp_path: Path,
) -> None:
    descriptor = _runtime_binding(
        "readback-required.v1",
        required_artifact_categories=("saved_stage_readback",),
    ).descriptor
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[descriptor],
    )
    graph = _graph(
        request,
        selected=[("readback-required.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "readback-required.v1")
    invocation, result = _write_leaf_artifacts(
        state_path,
        "readback-required.v1",
        include_readback=False,
    )
    with pytest.raises(AssetCompositionStateError, match="deterministic projector"):
        complete_leaf(
            state_path,
            "readback-required.v1",
            invocation_path=invocation,
            result_path=result,
        )
    running = load_verified_run(state_path)
    assert running.leaf_states["readback-required.v1"].status == "running"
    assert not (
        leaf_directory(state_path, "readback-required.v1") / "leaf_projection.json"
    ).exists()

    invocation_payload = json.loads(invocation.read_text(encoding="utf-8"))
    invocation_payload["undeclared"] = True
    atomic_write_json(invocation, invocation_payload)
    with pytest.raises(AssetCompositionStateError, match="invocation violates"):
        complete_leaf(
            state_path,
            "readback-required.v1",
            invocation_path=invocation,
            result_path=result,
        )


def test_identical_byte_different_path_substitution_fails_before_transition(
    tmp_path: Path,
) -> None:
    leaf_id = "path-substitution.v1"
    runtime_binding = _invocation_receipt_bound_runtime_binding(leaf_id)
    state_path, request = _create_agentic_run(
        tmp_path,
        runtime_bindings=[runtime_binding],
    )
    graph = _graph(
        request,
        selected=[(leaf_id, [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, leaf_id)
    directory = leaf_directory(state_path, leaf_id)
    original_invocation = directory / "native_invocation.json"
    substituted_invocation = directory / "selected_invocation.json"
    invocation = _FixtureInvocation(leaf_id=leaf_id, kind="invocation")
    atomic_write_json(original_invocation, invocation)
    atomic_write_json(substituted_invocation, invocation)
    original_binding = _binding(original_invocation)
    substituted_binding = _binding(substituted_invocation)
    assert original_binding.path != substituted_binding.path
    assert original_binding.sha256 == substituted_binding.sha256
    assert original_binding.size_bytes == substituted_binding.size_bytes
    terminal = directory / "native_terminal.json"
    atomic_write_json(terminal, {"leaf_id": leaf_id, "status": "succeeded"})
    result_path = directory / "result.json"
    atomic_write_json(
        result_path,
        _InvocationReceiptBoundFixtureResult(
            leaf_id=leaf_id,
            native_disposition="passed",
            native_status="succeeded",
            native_terminal_receipt=_binding(terminal),
            native_invocation_artifact=original_binding,
            summary="Native receipt retained its original invocation identity.",
        ),
    )

    with pytest.raises(AssetCompositionStateError, match="projector rejected"):
        complete_leaf(
            state_path,
            leaf_id,
            invocation_path=substituted_invocation,
            result_path=result_path,
        )

    running = load_verified_run(state_path)
    assert running.leaf_states[leaf_id].status == "running"
    assert not (directory / "leaf_projection.json").exists()


def test_substituted_native_result_or_evidence_fails_before_transition(
    tmp_path: Path,
) -> None:
    state_path, request = _create_agentic_run(
        tmp_path,
        descriptors=[_descriptor("substitution.v1")],
    )
    graph = _graph(
        request,
        selected=[("substitution.v1", [], "required", True)],
        omitted=[],
    )
    graph_path = state_path.parent / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, "substitution.v1")
    invocation, result = _write_leaf_artifacts(state_path, "substitution.v1")

    result_payload = json.loads(result.read_text(encoding="utf-8"))
    result_payload["leaf_id"] = "another.v1"
    atomic_write_json(result, result_payload)
    with pytest.raises(AssetCompositionStateError, match="projector rejected"):
        complete_leaf(
            state_path,
            "substitution.v1",
            invocation_path=invocation,
            result_path=result,
        )

    result_payload["leaf_id"] = "substitution.v1"
    atomic_write_json(result, result_payload)
    evidence_path = Path(result_payload["evidence"][0]["path"])
    evidence_path.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(AssetCompositionStateError, match="projected evidence.*changed"):
        complete_leaf(
            state_path,
            "substitution.v1",
            invocation_path=invocation,
            result_path=result,
        )
