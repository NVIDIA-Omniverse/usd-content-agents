# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import linecache
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict
from world_understanding.validation import (
    ValidationPlan,
    ValidationPlanStep,
    ValidationRequest,
    ValidationTemplateContext,
    ValidationTemplateResult,
)

import content_agent_workflows.asset_composition.catalog as asset_catalog
from content_agent_workflows.articulation.asset_leaf_adapter import (
    articulation_asset_leaf_runtime_bindings,
    articulation_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition import (
    ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP,
    ArtifactBinding,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafRegistrarIdentity,
    AssetLeafRuntimeBinding,
    AssetLeafRuntimeBundle,
    AssetLeafRuntimeRegistrar,
    asset_model_schema_digest,
    asset_projector_digest,
    canonical_asset_digest,
    compose_asset_leaf_runtime_bundles,
    compose_asset_leaf_runtime_catalog,
    discover_repository_asset_leaf_catalog,
    repository_asset_leaf_runtime_catalog,
    resolve_repository_asset_leaf_catalog,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    FOCUSED_VALIDATION_OPERATION_LEAF_ID,
    CanonicalOvrtxEvidenceLeafInvocation,
    CanonicalOvrtxEvidenceLeafResult,
    CanonicalOvrtxEvidenceLeafTerminalResult,
    FocusedValidationOperationLeafInvocation,
    FocusedValidationOperationLeafResult,
    SharedValidationLeafTerminalResult,
    _bound_file,
    _resolved_path,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.asset_composition.release_leaf_adapters import (
    RELEASE_COMPOSED_LEAF_IDS,
    release_composed_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.embedded_domain_decision import (
    canonical_json_digest,
)
from content_agent_workflows.texture.asset_leaf_adapter import (
    texture_asset_leaf_runtime_bundle,
)
from content_agent_workflows.validation import (
    CanonicalVisualEvidenceRequest,
    ValidationArtifactIdentity,
    ValidationOperationPreparation,
    ValidationOperationResult,
    execution_artifact_binding,
    prepare_validation_operations,
    run_validation_operation,
)


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class _Invocation(_FrozenModel):
    value: str


class _Result(_FrozenModel):
    value: str


class _FocusedPreparationExecutor:
    template_versions = {
        "physics_sane": "test.physics-sane.v1",
        "render_valid": "test.render-valid.v1",
        "look_right": "test.look-right.v1",
    }

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        del working_dir
        return ValidationPlan(
            steps=tuple(
                ValidationPlanStep(
                    template_name=name,
                    reason="Selected by the shared-adapter fixture.",
                )
                for name in request.requested_templates
            ),
            reasoning_summary="Preserved the explicit focused operation order.",
        )

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        del context
        return ValidationTemplateResult(template_name=template_name, status="passed")


def _projector(
    _invocation: BaseModel,
    _result: BaseModel,
    _context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    terminal = ArtifactBinding(path="/terminal.json", sha256="a" * 64, size_bytes=1)
    return AssetLeafProjectionPayload(
        native_disposition="passed",
        native_status="succeeded",
        native_terminal_receipt=terminal,
        evidence=(
            ArtifactBinding(path="/evidence.json", sha256="b" * 64, size_bytes=1),
        ),
        saved_stage_readbacks=(
            ArtifactBinding(path="/readback.json", sha256="c" * 64, size_bytes=1),
        ),
        summary="Projected deterministic fixture.",
    )


def _different_projector(
    _invocation: BaseModel,
    _result: BaseModel,
    _context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    return AssetLeafProjectionPayload(
        native_disposition="failed",
        native_status="provider_failure",
        native_terminal_receipt=ArtifactBinding(
            path="/failed.json", sha256="d" * 64, size_bytes=1
        ),
        summary="Different deterministic fixture.",
        error="fixture failure",
    )


def test_projector_digest_binds_same_module_helper_source(tmp_path: Path) -> None:
    module_path = tmp_path / "fixture_projector.py"
    module_path.write_text(
        "def helper():\n    return 'first'\n\n"
        "def projector(invocation, result, context):\n"
        "    return helper()\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("fixture_projector", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        first = asset_projector_digest("fixture.projector.v1", module.projector)
        module_path.write_text(
            "def helper():\n    return 'second'\n\n"
            "def projector(invocation, result, context):\n"
            "    return helper()\n",
            encoding="utf-8",
        )
        linecache.clearcache()
        second = asset_projector_digest("fixture.projector.v1", module.projector)
    finally:
        sys.modules.pop(spec.name, None)

    assert first != second


def _binding(
    leaf_id: str,
    *,
    required_dependencies: tuple[str, ...] = (),
    incompatible_leaf_ids: tuple[str, ...] = (),
) -> AssetLeafRuntimeBinding:
    return AssetLeafRuntimeBinding.create(
        leaf_id=leaf_id,
        entrypoint=f"tests.{leaf_id}",
        invocation_model=_Invocation,
        result_model=_Result,
        projector_id=f"projector.{leaf_id}",
        projector=_projector,
        required_dependencies=required_dependencies,
        incompatible_leaf_ids=incompatible_leaf_ids,
    )


def _texture_bundle_provider() -> AssetLeafRuntimeBundle:
    return AssetLeafRuntimeBundle.create(
        bundle_id="texture",
        bindings=tuple(_binding(f"texture.contract-{index}.v1") for index in range(6)),
    )


def _joint_bundle_provider() -> AssetLeafRuntimeBundle:
    return AssetLeafRuntimeBundle.create(
        bundle_id="joint",
        bindings=(
            _binding(
                "joint.fixed-distance-author-readback.v1",
                required_dependencies=(
                    "articulation.preparation.v1",
                    "articulation.graph-accept-revise.v1",
                ),
            ),
            _binding(
                "joint.retained-static-evidence-score.v1",
                required_dependencies=(
                    "articulation.gate3a.v1",
                    "articulation.gate3b.v1",
                    "articulation.graph-apply-readback-projection.v1",
                ),
            ),
        ),
    )


def _approved_domain_bundle_provider() -> AssetLeafRuntimeBundle:
    return AssetLeafRuntimeBundle.create(
        bundle_id="approved-domain",
        bindings=(_binding("fixture.approved-domain.v1"),),
    )


def test_legacy_domain_catalog_wire_surface_remains_exact() -> None:
    descriptor_payload = {
        "schema_version": "content-agents.asset-leaf-descriptor.v1",
        "leaf_id": "legacy.domain-leaf.v1",
        "entrypoint": "legacy.domain.public_boundary",
        "invocation_schema_digest": "a" * 64,
        "result_schema_digest": "b" * 64,
        "required_dependencies": [],
        "incompatible_leaf_ids": [],
    }
    descriptor = AssetLeafDescriptor.create(
        leaf_id="legacy.domain-leaf.v1",
        entrypoint="legacy.domain.public_boundary",
        invocation_schema_digest="a" * 64,
        result_schema_digest="b" * 64,
    )
    assert descriptor.schema_version == "content-agents.asset-leaf-descriptor.v1"
    assert descriptor.descriptor_digest == canonical_asset_digest(descriptor_payload)
    assert descriptor.model_dump(mode="json") == {
        **descriptor_payload,
        "descriptor_digest": descriptor.descriptor_digest,
    }

    catalog = AssetLeafCatalog.create([descriptor])
    serialized = catalog.model_dump(mode="json")
    assert catalog.schema_version == "content-agents.asset-leaf-catalog.v1"
    assert "registrars" not in serialized
    assert AssetLeafCatalog.model_validate(serialized) == catalog


def _articulation_bundle_provider() -> AssetLeafRuntimeBundle:
    author = "joint.fixed-distance-author-readback.v1"
    preparation = "articulation.preparation.v1"
    accepted = "articulation.graph-accept-revise.v1"
    gate3a = "articulation.gate3a.v1"
    return AssetLeafRuntimeBundle.create(
        bundle_id="articulation",
        bindings=(
            _binding(preparation),
            _binding(
                "articulation.proposal.v1",
                required_dependencies=(preparation,),
            ),
            _binding(accepted, required_dependencies=(preparation,)),
            _binding(
                "articulation.graph-apply-readback-projection.v1",
                required_dependencies=(author,),
            ),
            _binding(gate3a, required_dependencies=(author,)),
            _binding(
                "articulation.gate3b.v1",
                required_dependencies=(author, gate3a),
            ),
            _binding("articulation.dynamic.v1", required_dependencies=(author,)),
        ),
    )


@dataclass(frozen=True)
class _EntryPoint:
    name: str
    provider: object
    group: str = ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP

    def load(self) -> object:
        if isinstance(self.provider, BaseException):
            raise self.provider
        return self.provider


@dataclass(frozen=True)
class _Distribution:
    entry_points: tuple[_EntryPoint, ...]
    name: str = "content-agent-workflows"
    version: str = "fixture"

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


@pytest.fixture(autouse=True)
def _clear_repository_catalog_cache() -> Iterator[None]:
    asset_catalog.repository_asset_leaf_runtime_catalog.cache_clear()
    yield
    asset_catalog.repository_asset_leaf_runtime_catalog.cache_clear()


def _declare_bundles(
    monkeypatch: pytest.MonkeyPatch,
    *entries: _EntryPoint,
) -> None:
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (_Distribution(entry_points=tuple(entries)),),
    )


def test_repository_catalog_is_deterministic_and_runtime_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
    )
    first = discover_repository_asset_leaf_catalog()
    second = discover_repository_asset_leaf_catalog()

    assert first == second
    assert first.schema_version == "content-agents.asset-leaf-catalog.v3"
    assert [item.leaf_id for item in first.descriptors] == [
        "validation.canonical-ovrtx-evidence.v1",
        "validation.focused-operation.v1",
        "validation.provided-ingress.v1",
    ]
    runtime = repository_asset_leaf_runtime_catalog()
    assert tuple(runtime.bindings) == tuple(
        descriptor.leaf_id for descriptor in first.descriptors
    )
    for descriptor in first.descriptors:
        binding = runtime.resolve(descriptor)
        assert binding.descriptor == descriptor
        assert descriptor.projector_id != "asset.projector.unbound.v1"
    canonical_defaults = CanonicalOvrtxEvidenceLeafInvocation(
        output_dir="/run/evidence",
        post_mutation_usd="/run/output.usda",
        source_usd="/run/source.usda",
        backend="ovrtx",
        views=("+x+y+z",),
        image_width=1024,
        image_height=1024,
    )
    assert canonical_defaults.operation_id == "visual.canonical-ovrtx"
    assert canonical_defaults.gate_id == "visual.canonical-evidence"


def test_repository_catalog_accepts_shared_and_approved_domain_registrars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows = _Distribution(
        entry_points=(
            _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
        )
    )
    approved_domain = _Distribution(
        name="joint-agent",
        entry_points=(
            _EntryPoint(
                name="approved-domain",
                provider=_approved_domain_bundle_provider,
            ),
        ),
    )
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (workflows,),
    )
    shared = discover_repository_asset_leaf_catalog()

    asset_catalog.repository_asset_leaf_runtime_catalog.cache_clear()
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (approved_domain, workflows),
    )
    extended = discover_repository_asset_leaf_catalog()

    mandatory_shared_ids = {
        binding.descriptor.leaf_id
        for binding in shared_asset_leaf_runtime_bundle().bindings
    }
    shared_ids = [descriptor.leaf_id for descriptor in shared.descriptors]
    extended_ids = [descriptor.leaf_id for descriptor in extended.descriptors]
    assert set(shared_ids) == mandatory_shared_ids
    assert mandatory_shared_ids < set(extended_ids)
    assert extended_ids == sorted(extended_ids)
    assert [registrar.registrar_id for registrar in extended.registrars] == [
        "content-agent-workflows",
        "joint-agent",
    ]
    assert resolve_repository_asset_leaf_catalog(extended).catalog == extended


def test_clean_release_catalog_discovers_every_composed_domain_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows = _Distribution(
        entry_points=(
            _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
            _EntryPoint(
                name="release-composed",
                provider=release_composed_asset_leaf_runtime_bundle,
            ),
        )
    )
    joint = _Distribution(
        name="joint-agent",
        entry_points=(
            _EntryPoint(
                name="articulation",
                provider=articulation_asset_leaf_runtime_bundle,
            ),
        ),
    )
    texture = _Distribution(
        name="texture-agent",
        entry_points=(
            _EntryPoint(
                name="texture",
                provider=texture_asset_leaf_runtime_bundle,
            ),
        ),
    )
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (texture, workflows, joint),
    )

    catalog = discover_repository_asset_leaf_catalog()
    discovered = {descriptor.leaf_id for descriptor in catalog.descriptors}

    assert set(RELEASE_COMPOSED_LEAF_IDS).issubset(discovered)
    assert {
        binding.descriptor.leaf_id
        for binding in articulation_asset_leaf_runtime_bundle().bindings
    }.issubset(discovered)
    assert {
        binding.descriptor.leaf_id
        for binding in texture_asset_leaf_runtime_bundle().bindings
    }.issubset(discovered)
    assert [registrar.registrar_id for registrar in catalog.registrars] == [
        "content-agent-workflows",
        "joint-agent",
        "texture-agent",
    ]


def test_repository_catalog_rejects_reordered_wire_descriptors() -> None:
    runtime = compose_asset_leaf_runtime_catalog(
        (
            _binding("fixture.zeta.v1"),
            _binding("fixture.alpha.v1"),
        )
    )
    assert [descriptor.leaf_id for descriptor in runtime.catalog.descriptors] == [
        "fixture.alpha.v1",
        "fixture.zeta.v1",
    ]

    reordered = runtime.catalog.model_dump(mode="json")
    reordered["descriptors"].reverse()
    digest_payload = dict(reordered)
    digest_payload.pop("catalog_digest")
    reordered["catalog_digest"] = canonical_asset_digest(digest_payload)
    with pytest.raises(ValueError, match="descriptors must be sorted"):
        AssetLeafCatalog.model_validate(reordered)


def test_projected_file_binding_uses_one_verified_byte_identity(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(b"first")
    digest = hashlib.sha256(b"first").hexdigest()

    assert _bound_file(str(artifact), digest) == ArtifactBinding(
        path=str(artifact.resolve()),
        sha256=digest,
        size_bytes=5,
    )

    artifact.write_bytes(b"other")
    with pytest.raises(ValueError, match="digest is stale"):
        _bound_file(str(artifact), digest)

    artifact.unlink()
    with pytest.raises(ValueError, match="missing or unsafe"):
        _bound_file(str(artifact), digest)


def test_selected_leaf_adapter_rejects_cwd_relative_paths() -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        _resolved_path("relative/native-output")


def _canonical_binding() -> AssetLeafRuntimeBinding:
    runtime = shared_asset_leaf_runtime_bundle()
    return next(
        candidate
        for candidate in runtime.bindings
        if candidate.descriptor.leaf_id == CANONICAL_OVRTX_EVIDENCE_LEAF_ID
    )


@pytest.mark.parametrize(
    "defect",
    [
        "substituted-request",
        "forged-source-digest",
        "aliased-terminal",
        "aliased-path-terminal",
        "aliased-request-terminal",
    ],
)
def test_canonical_terminal_projection_rejects_substitution_and_category_alias(
    tmp_path: Path,
    defect: str,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "canonical"
    output_dir.mkdir()
    (output_dir / "sub").mkdir()
    native_request_path = output_dir / "canonical_visual_request.json"
    terminal_path = output_dir / "terminal.json"
    evidence_path = output_dir / "evidence.json"
    readback_path = output_dir / "readback.json"
    invocation = CanonicalOvrtxEvidenceLeafInvocation(
        output_dir=str(output_dir),
        post_mutation_usd=str(source),
        source_usd=str(source),
        backend="ovrtx",
        views=("+x+y+z",),
        image_width=64,
        image_height=64,
    )
    source_binding = execution_artifact_binding(source)
    if defect == "forged-source-digest":
        source_binding = source_binding.model_copy(update={"sha256": "0" * 64})
    native_request = CanonicalVisualEvidenceRequest(
        source=source_binding,
        post_mutation_output=execution_artifact_binding(source),
        backend="remote" if defect == "substituted-request" else "ovrtx",
        views=invocation.views,
        image_width=invocation.image_width,
        image_height=invocation.image_height,
    )
    native_request_path.write_text(native_request.model_dump_json(), encoding="utf-8")
    terminal_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    evidence_path.write_text('{"retained":true}\n', encoding="utf-8")
    readback_path.write_text('{"unchanged":true}\n', encoding="utf-8")
    terminal_binding = (
        _artifact(native_request_path)
        if defect == "aliased-request-terminal"
        else _artifact(terminal_path)
    )
    evidence_binding = _artifact(evidence_path)
    if defect == "aliased-path-terminal":
        evidence_binding = terminal_binding.model_copy(
            update={"path": str(output_dir / "sub" / ".." / terminal_path.name)}
        )
    result = CanonicalOvrtxEvidenceLeafResult(
        root=CanonicalOvrtxEvidenceLeafTerminalResult(
            output_dir=str(output_dir),
            native_disposition="failed",
            native_status="failed",
            native_request=_artifact(native_request_path),
            native_terminal_receipt=terminal_binding,
            evidence=(
                terminal_binding if defect == "aliased-terminal" else evidence_binding,
            ),
            saved_stage_readbacks=(_artifact(readback_path),),
            summary="Canonical OVRTX failed.",
            error="Canonical OVRTX failed.",
        )
    )
    invocation_path = tmp_path / "invocation.json"
    result_path = tmp_path / "result.json"
    invocation_path.write_text(invocation.model_dump_json(), encoding="utf-8")
    result_path.write_text(result.model_dump_json(), encoding="utf-8")

    if defect == "substituted-request":
        match = "differs from its invocation"
    elif defect == "forged-source-digest":
        match = "native source is missing, unsafe, or stale"
    else:
        match = "pairwise distinct"
    with pytest.raises(ValueError, match=match):
        _canonical_binding().project(
            invocation,
            result,
            invocation_artifact=_artifact(invocation_path),
            result_artifact=_artifact(result_path),
        )


def test_shared_validation_terminal_rejects_identical_invocation_at_other_path(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "focused"
    output_dir.mkdir()
    invocation = FocusedValidationOperationLeafInvocation(
        output_dir=str(output_dir),
        template_name="physics_sane",
    )
    invocation_path = tmp_path / "invocation.json"
    substituted_path = tmp_path / "substituted-invocation.json"
    invocation_path.write_text(invocation.model_dump_json(), encoding="utf-8")
    substituted_path.write_bytes(invocation_path.read_bytes())
    terminal_path = output_dir / "terminal.json"
    evidence_path = output_dir / "evidence.json"
    readback_path = output_dir / "readback.json"
    terminal_path.write_text('{"status":"failed"}\n', encoding="utf-8")
    evidence_path.write_text('{"retained":true}\n', encoding="utf-8")
    readback_path.write_text('{"unchanged":true}\n', encoding="utf-8")
    result = FocusedValidationOperationLeafResult(
        root=SharedValidationLeafTerminalResult(
            output_dir=str(output_dir),
            invocation_artifact=_artifact(substituted_path),
            native_disposition="failed",
            native_status="failed",
            native_terminal_receipt=_artifact(terminal_path),
            evidence=(_artifact(evidence_path),),
            saved_stage_readbacks=(_artifact(readback_path),),
            summary="Focused Validation failed.",
            error="Focused Validation failed.",
        )
    )
    result_path = tmp_path / "result.json"
    result_path.write_text(result.model_dump_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="substituted its invocation"):
        _focused_binding().project(
            invocation,
            result,
            invocation_artifact=_artifact(invocation_path),
            result_artifact=_artifact(result_path),
        )


def _focused_binding() -> AssetLeafRuntimeBinding:
    runtime = shared_asset_leaf_runtime_bundle()
    return next(
        candidate
        for candidate in runtime.bindings
        if candidate.descriptor.leaf_id == FOCUSED_VALIDATION_OPERATION_LEAF_ID
    )


def _artifact(path: Path) -> ArtifactBinding:
    payload = path.read_bytes()
    return ArtifactBinding(
        path=str(path.resolve()),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _prepare_focused_fixture(
    output_dir: Path,
    source: Path,
    *templates: str,
) -> ValidationOperationPreparation:
    return prepare_validation_operations(
        ValidationRequest(
            task_description="Validate the shared focused-operation adapter.",
            inputs=(str(source),),
            requested_templates=templates,
        ),
        output_dir=output_dir,
        config_base_dir=source.parent,
        executor=_FocusedPreparationExecutor(),
    )


def _write_focused_result(
    preparation: ValidationOperationPreparation,
    template_name: str,
    *,
    evidence_artifacts: tuple[ValidationArtifactIdentity, ...] = (),
) -> tuple[ValidationOperationResult, Path, Path]:
    root = Path(preparation.output_dir)
    operation_dir = root / "operations" / template_name
    operation_dir.mkdir(parents=True)
    template_result = ValidationTemplateResult(
        template_name=template_name,
        status="passed",
    )
    template_path = operation_dir / "template_result.json"
    template_path.write_text(template_result.model_dump_json(), encoding="utf-8")
    role = {
        "physics_sane": "deterministic_check",
        "render_valid": "render_evidence_leaf",
        "look_right": "optional_advisory_critique",
    }[template_name]
    operation = ValidationOperationResult(
        preparation_digest=canonical_json_digest(preparation),
        template_name=template_name,
        role=role,
        mandatory=template_name != "look_right",
        template_result=template_result,
        template_result_path=str(template_path),
        template_result_sha256=hashlib.sha256(template_path.read_bytes()).hexdigest(),
        evidence_artifacts=evidence_artifacts,
        source_before=preparation.workflow_identity.source_artifacts,
        source_after=preparation.workflow_identity.source_artifacts,
    )
    operation_path = operation_dir / "operation_result.json"
    operation_path.write_text(operation.model_dump_json(), encoding="utf-8")
    return operation, operation_path, template_path


def _project_focused_fixture(
    tmp_path: Path,
    invocation_model: FocusedValidationOperationLeafInvocation,
    result_model: ValidationOperationResult,
    *,
    attempt_name: str,
) -> AssetLeafProjectionPayload:
    attempt = tmp_path / attempt_name
    attempt.mkdir()
    invocation = attempt / "invocation.json"
    result = attempt / "result.json"
    invocation.write_text(invocation_model.model_dump_json(), encoding="utf-8")
    leaf_result = FocusedValidationOperationLeafResult.model_validate(
        result_model.model_dump(mode="json")
    )
    result.write_text(leaf_result.model_dump_json(), encoding="utf-8")
    return (
        _focused_binding()
        .project(
            invocation_model,
            leaf_result,
            invocation_artifact=_artifact(invocation),
            result_artifact=_artifact(result),
        )
        .payload
    )


def test_focused_projection_accepts_public_operation_result(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "validation"
    executor = _FocusedPreparationExecutor()
    _prepare_focused_fixture(output_dir, source, "physics_sane")
    result_model = run_validation_operation(
        output_dir,
        template_name="physics_sane",
        executor=executor,
    )

    projection = _project_focused_fixture(
        tmp_path,
        FocusedValidationOperationLeafInvocation(
            output_dir=str(output_dir),
            template_name="physics_sane",
        ),
        result_model,
        attempt_name="public-result-attempt",
    )

    operation_path = output_dir / "operations/physics_sane/operation_result.json"
    assert projection.native_disposition == "passed"
    assert projection.native_terminal_receipt.path == str(operation_path.resolve())


def test_focused_projection_binds_preparation_and_canonical_operation_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "inputs" / "source.usda"
    source.parent.mkdir()
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "validation"
    preparation = _prepare_focused_fixture(output_dir, source, "physics_sane")
    generated = output_dir / "generated-evidence.json"
    generated.write_text('{"passed":true}\n', encoding="utf-8")
    source_identity = preparation.workflow_identity.source_artifacts[0]
    result_model, operation_path, terminal = _write_focused_result(
        preparation,
        "physics_sane",
        evidence_artifacts=(
            ValidationArtifactIdentity(
                role="evidence",
                **source_identity.model_dump(exclude={"role"}),
            ),
            ValidationArtifactIdentity(
                role="evidence",
                path=str(generated),
                kind="file",
                sha256=hashlib.sha256(generated.read_bytes()).hexdigest(),
            ),
        ),
    )
    invocation_model = FocusedValidationOperationLeafInvocation(
        output_dir=str(output_dir),
        template_name="physics_sane",
    )

    projection = _project_focused_fixture(
        tmp_path,
        invocation_model,
        result_model,
        attempt_name="positive-attempt",
    )

    assert result_model.evidence_artifacts[0].path == str(source)
    assert projection.native_terminal_receipt.path == str(operation_path.resolve())
    assert [item.path for item in projection.evidence] == [
        str((output_dir / "validation_operation_preparation.json").resolve()),
        str(operation_path.resolve()),
        str(terminal.resolve()),
        str(generated.resolve()),
    ]


def test_focused_projection_accepts_exact_prepared_prior_result_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    output_dir = tmp_path / "validation"
    preparation = _prepare_focused_fixture(
        output_dir,
        source,
        "render_valid",
        "look_right",
    )
    _, prior_path, prior_terminal = _write_focused_result(
        preparation,
        "render_valid",
    )
    result_model, operation_path, _ = _write_focused_result(
        preparation,
        "look_right",
    )

    projection = _project_focused_fixture(
        tmp_path,
        FocusedValidationOperationLeafInvocation(
            output_dir=str(output_dir),
            template_name="look_right",
            prior_result_paths=(str(prior_path),),
        ),
        result_model,
        attempt_name="prior-chain-attempt",
    )

    assert projection.native_disposition == "passed"
    assert projection.native_terminal_receipt.path == str(operation_path.resolve())
    assert [item.path for item in projection.saved_stage_readbacks[1:3]] == [
        str(prior_path.resolve()),
        str(prior_terminal.resolve()),
    ]


def test_focused_projection_rejects_same_template_from_another_preparation(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    expected = _prepare_focused_fixture(
        tmp_path / "expected",
        source,
        "physics_sane",
    )
    substituted = _prepare_focused_fixture(
        tmp_path / "substituted",
        source,
        "physics_sane",
    )
    substituted_result, _, _ = _write_focused_result(
        substituted,
        "physics_sane",
    )

    with pytest.raises(ValueError, match="another preparation"):
        _project_focused_fixture(
            tmp_path,
            FocusedValidationOperationLeafInvocation(
                output_dir=expected.output_dir,
                template_name="physics_sane",
            ),
            substituted_result,
            attempt_name="substituted-preparation-attempt",
        )


def test_focused_projection_rejects_substituted_canonical_operation_result(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    preparation = _prepare_focused_fixture(
        tmp_path / "validation",
        source,
        "physics_sane",
    )
    result_model, _, _ = _write_focused_result(preparation, "physics_sane")
    substituted_result = result_model.model_copy(
        update={"completed_at": result_model.completed_at.replace(microsecond=1)}
    )

    with pytest.raises(ValueError, match="canonical operation result"):
        _project_focused_fixture(
            tmp_path,
            FocusedValidationOperationLeafInvocation(
                output_dir=preparation.output_dir,
                template_name="physics_sane",
            ),
            substituted_result,
            attempt_name="substituted-operation-attempt",
        )


def test_focused_projection_rejects_substituted_prior_result_chain(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    preparation = _prepare_focused_fixture(
        tmp_path / "validation",
        source,
        "render_valid",
        "look_right",
    )
    _write_focused_result(preparation, "render_valid")
    result_model, _, _ = _write_focused_result(preparation, "look_right")
    other_preparation = _prepare_focused_fixture(
        tmp_path / "other-validation",
        source,
        "render_valid",
    )
    _, substituted_prior, _ = _write_focused_result(
        other_preparation,
        "render_valid",
    )

    with pytest.raises(ValueError, match="prior-result chain"):
        _project_focused_fixture(
            tmp_path,
            FocusedValidationOperationLeafInvocation(
                output_dir=preparation.output_dir,
                template_name="look_right",
                prior_result_paths=(str(substituted_prior),),
            ),
            result_model,
            attempt_name="substituted-prior-attempt",
        )


def test_texture_six_and_joint_bundle_fixtures_compose_stably() -> None:
    texture = _texture_bundle_provider()
    articulation = _articulation_bundle_provider()
    joint = _joint_bundle_provider()
    first = compose_asset_leaf_runtime_bundles((texture, articulation, joint))
    second = compose_asset_leaf_runtime_bundles((joint, texture, articulation))

    assert first.catalog == second.catalog
    assert first.catalog.catalog_digest == second.catalog.catalog_digest
    assert tuple(first.bindings) == tuple(sorted(first.bindings))
    assert len(first.bindings) == 15
    assert first.bindings[
        "articulation.graph-accept-revise.v1"
    ].descriptor.required_dependencies == ["articulation.preparation.v1"]
    assert first.bindings[
        "articulation.proposal.v1"
    ].descriptor.required_dependencies == ["articulation.preparation.v1"]
    assert (
        "articulation.proposal.v1"
        not in first.bindings[
            "articulation.graph-accept-revise.v1"
        ].descriptor.required_dependencies
    )
    assert first.bindings[
        "joint.retained-static-evidence-score.v1"
    ].descriptor.required_dependencies == [
        "articulation.gate3a.v1",
        "articulation.gate3b.v1",
        "articulation.graph-apply-readback-projection.v1",
    ]


def test_merged_articulation_schema_exports_fit_runtime_binding_contract() -> None:
    bindings = articulation_asset_leaf_runtime_bindings()

    assert len(bindings) == 6
    for binding in bindings:
        assert binding.descriptor.invocation_schema_digest == asset_model_schema_digest(
            binding.invocation_model
        )
        assert binding.descriptor.result_schema_digest == asset_model_schema_digest(
            binding.result_model
        )
        binding.validate_identity()


def test_multiple_approved_registrars_bind_source_and_returned_sets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows = _Distribution(
        entry_points=(
            _EntryPoint(name="articulation", provider=_articulation_bundle_provider),
            _EntryPoint(name="texture", provider=_texture_bundle_provider),
            _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
        )
    )
    joint = _Distribution(
        name="joint-agent",
        entry_points=(_EntryPoint(name="joint", provider=_joint_bundle_provider),),
    )
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (joint, workflows),
    )
    first = discover_repository_asset_leaf_catalog()
    asset_catalog.repository_asset_leaf_runtime_catalog.cache_clear()
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (workflows, joint),
    )
    second = discover_repository_asset_leaf_catalog()

    assert first == second
    assert [item.registrar_id for item in first.registrars] == [
        "content-agent-workflows",
        "joint-agent",
    ]
    assert [
        bundle.bundle_id
        for registrar in first.registrars
        for bundle in registrar.bundles
    ] == ["articulation", "shared", "texture", "joint"]
    assert all(
        bundle.source_sha256 != "0" * 64
        for registrar in first.registrars
        for bundle in registrar.bundles
    )
    assert {
        bundle.implementation.rsplit(".", 1)[-1]
        for registrar in first.registrars
        for bundle in registrar.bundles
    } == {
        "_articulation_bundle_provider",
        "shared_asset_leaf_runtime_bundle",
        "_texture_bundle_provider",
        "_joint_bundle_provider",
    }


def test_registrar_composition_rejects_partial_or_substituted_returned_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
    )
    (registrar,) = asset_catalog._load_repository_asset_leaf_runtime_registrars()
    shared = registrar.bundles[0]
    substituted = AssetLeafRuntimeBundle.create(
        bundle_id=shared.bundle_id,
        bindings=shared.bindings[:-1],
    )

    with pytest.raises(ValueError, match="returned a stale bundle"):
        asset_catalog.compose_asset_leaf_runtime_registrars(
            (
                AssetLeafRuntimeRegistrar(
                    identity=registrar.identity,
                    bundles=(substituted,),
                ),
            )
        )

    stale_identity = registrar.identity.model_copy(update={"source_sha256": "f" * 64})
    with pytest.raises(ValueError, match="source identity differs from bundles"):
        asset_catalog.compose_asset_leaf_runtime_registrars(
            (
                AssetLeafRuntimeRegistrar(
                    identity=stale_identity,
                    bundles=registrar.bundles,
                ),
            )
        )

    for field, value, match in (
        (
            "implementation",
            "tests.substituted_bundle_provider",
            "implementation identity differs from bundles",
        ),
        ("source_sha256", "e" * 64, "source identity differs from bundles"),
    ):
        payload = registrar.identity.model_dump(mode="json")
        payload[field] = value
        digest_payload = dict(payload)
        digest_payload.pop("registrar_digest")
        payload["registrar_digest"] = canonical_asset_digest(digest_payload)
        with pytest.raises(ValueError, match=match):
            AssetLeafRegistrarIdentity.model_validate(payload)


def test_bundle_composition_rejects_duplicate_bundle_and_leaf() -> None:
    first = AssetLeafRuntimeBundle.create(
        bundle_id="first",
        bindings=(_binding("fixture.bundle-leaf.v1"),),
    )
    duplicate_bundle = AssetLeafRuntimeBundle.create(
        bundle_id="first",
        bindings=(_binding("fixture.other-leaf.v1"),),
    )
    with pytest.raises(ValueError, match="duplicate repository asset leaf bundle"):
        compose_asset_leaf_runtime_bundles((first, duplicate_bundle))

    duplicate_leaf = AssetLeafRuntimeBundle.create(
        bundle_id="second",
        bindings=(_binding("fixture.bundle-leaf.v1"),),
    )
    with pytest.raises(
        ValueError, match="duplicate repository asset leaf registration"
    ):
        compose_asset_leaf_runtime_bundles((first, duplicate_leaf))

    with pytest.raises(TypeError, match="invalid binding"):
        AssetLeafRuntimeBundle.create(
            bundle_id="type-drift",
            bindings=(object(),),  # type: ignore[arg-type]
        )


def test_repository_bundle_discovery_fails_closed_for_absent_or_broken_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_bundles(monkeypatch)
    with pytest.raises(RuntimeError, match="mandatory.*registrars are missing"):
        discover_repository_asset_leaf_catalog()

    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="broken", provider=ImportError("fixture unavailable")),
    )
    with pytest.raises(RuntimeError, match="bundle failed to load:.*broken"):
        discover_repository_asset_leaf_catalog()

    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="wrong-type", provider=lambda: object()),
    )
    with pytest.raises(RuntimeError, match="bundle failed to load:.*wrong-type"):
        discover_repository_asset_leaf_catalog()


def test_one_broken_approved_domain_bundle_rejects_partial_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflows = _Distribution(
        entry_points=(
            _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
        )
    )
    texture = _Distribution(
        name="texture-agent",
        entry_points=(
            _EntryPoint(name="texture", provider=ImportError("cycle detected")),
        ),
    )
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (workflows, texture),
    )

    with pytest.raises(RuntimeError, match="texture-agent/texture"):
        discover_repository_asset_leaf_catalog()


def test_repository_bundle_discovery_loads_only_explicit_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unexpected_loads: list[str] = []

    def optional_domain_provider() -> AssetLeafRuntimeBundle:
        unexpected_loads.append("optional-domain")
        return AssetLeafRuntimeBundle.create(
            bundle_id="optional-domain",
            bindings=(_binding("optional.contract.v1"),),
        )

    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
    )
    # Merely defining an optional provider cannot register it or trigger imports.
    assert optional_domain_provider is not None
    discover_repository_asset_leaf_catalog()
    assert unexpected_loads == []
    source = Path(asset_catalog.__file__).read_text(encoding="utf-8")
    assert "import joint_agent" not in source
    assert "content_agent_workflows.texture" not in source


def test_repository_bundle_discovery_rejects_unapproved_registrar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    external = _Distribution(
        name="external-provider",
        entry_points=(
            _EntryPoint(name="optional-domain", provider=_texture_bundle_provider),
        ),
    )
    monkeypatch.setattr(
        asset_catalog.importlib_metadata,
        "distributions",
        lambda: (external,),
    )
    with pytest.raises(RuntimeError, match="unapproved asset leaf registrar"):
        discover_repository_asset_leaf_catalog()


def test_catalog_composition_rejects_duplicate_unknown_and_incompatible() -> None:
    duplicate = _binding("fixture.duplicate.v1")
    with pytest.raises(ValueError, match="duplicate repository asset leaf"):
        compose_asset_leaf_runtime_catalog((duplicate, duplicate))

    unknown = _binding(
        "fixture.unknown.v1",
        required_dependencies=("fixture.missing.v1",),
    )
    with pytest.raises(ValueError, match="unknown constraints"):
        compose_asset_leaf_runtime_catalog((unknown,))

    dependency = _binding(
        "fixture.dependency.v1",
        incompatible_leaf_ids=("fixture.owner.v1",),
    )
    owner = _binding(
        "fixture.owner.v1",
        required_dependencies=("fixture.dependency.v1",),
    )
    with pytest.raises(ValueError, match="requires incompatible registration"):
        compose_asset_leaf_runtime_catalog((owner, dependency))


def test_catalog_composition_rejects_schema_and_projector_drift() -> None:
    binding = _binding("fixture.drift.v1")
    stale_descriptor = binding.descriptor.model_copy(
        update={"invocation_schema_digest": "f" * 64}
    )
    with pytest.raises(ValueError, match="invocation_schema_digest drifted"):
        compose_asset_leaf_runtime_catalog(
            (
                AssetLeafRuntimeBinding(
                    descriptor=stale_descriptor,
                    invocation_model=binding.invocation_model,
                    result_model=binding.result_model,
                    projector=binding.projector,
                ),
            )
        )

    with pytest.raises(ValueError, match="projector_digest drifted"):
        compose_asset_leaf_runtime_catalog(
            (
                AssetLeafRuntimeBinding(
                    descriptor=binding.descriptor,
                    invocation_model=binding.invocation_model,
                    result_model=binding.result_model,
                    projector=_different_projector,
                ),
            )
        )

    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        AssetLeafRuntimeBinding.create(
            leaf_id="fixture.type-drift.v1",
            entrypoint="fixture.invalid",
            invocation_model=dict,  # type: ignore[arg-type]
            result_model=_Result,
            projector_id="projector.fixture.type-drift.v1",
            projector=_projector,
        )


def test_external_catalog_must_resolve_exact_repository_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _declare_bundles(
        monkeypatch,
        _EntryPoint(name="shared", provider=shared_asset_leaf_runtime_bundle),
    )
    repository = discover_repository_asset_leaf_catalog()
    assert resolve_repository_asset_leaf_catalog(repository).catalog == repository

    fixture = compose_asset_leaf_runtime_catalog((_binding("fixture.external.v1"),))
    with pytest.raises(ValueError, match="does not resolve"):
        resolve_repository_asset_leaf_catalog(fixture.catalog)


def test_selected_projection_rejects_not_requested_native_status() -> None:
    with pytest.raises(ValueError, match="graph omission"):
        AssetLeafProjectionPayload(
            native_disposition="passed",
            native_status="not_requested",
            native_terminal_receipt=ArtifactBinding(
                path="/terminal.json",
                sha256="a" * 64,
                size_bytes=1,
            ),
            summary="Invalid selected omission.",
        )
