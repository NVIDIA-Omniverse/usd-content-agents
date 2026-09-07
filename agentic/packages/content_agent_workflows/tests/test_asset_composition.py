# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for durable single-asset workflow coordination."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from geometry_authoring_contracts import (
    GeometryArtifactBinding,
    GeometryAuthoringProviderIdentity,
    GeometryCoordinateSystem,
    GeometryRepresentationBinding,
    GeometryRightsAssertion,
    GeometrySourceBundle,
    GeometrySourceProvenance,
    geometry_source_bundle_id,
)
from pydantic import ValidationError
from world_understanding.utils.usd.package import write_usdz_package_from_directory

import content_agent_workflows.asset_composition.state as asset_state
from content_agent_workflows.asset_composition import (
    LEGACY_STAGE_ORDER,
    AssetCompositionStateError,
    begin_stage,
    build_combined_report,
    cancel_stage,
    complete_stage,
    create_run,
    load_embedded_articulation_human_acceptance,
    load_verified_run,
    record_coordinator_evidence_review,
    record_coordinator_plan,
    record_review_decisions,
    recover_stage,
    require_review,
    stage_directory,
    validate_terminal,
)
from content_agent_workflows.asset_composition.cli import main
from content_agent_workflows.asset_composition.models import (
    ArtifactBinding,
    AssetGeometryRequest,
    AssetRunRequest,
    AssetRuntimeRequest,
    AssetSourceStaging,
    LegacyAssetCadModelingRequest,
)
from content_agent_workflows.common.artifacts import file_sha256
from content_agent_workflows.common.validation_evidence import (
    physics_validation_evidence,
)


def _create(
    tmp_path: Path,
    *,
    coordinator_mode: Literal["legacy", "single_reasoning_loop"] = "legacy",
) -> tuple[Path, Path]:
    source = tmp_path / "source.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = run_dir / "request.json"
    request.write_text('{"prompt":"compose asset"}\n', encoding="utf-8")
    state_path = run_dir / "asset_run.json"
    create_run(
        state_path,
        run_id="asset-test",
        request_path=request,
        source_asset=source,
        coordinator_mode=coordinator_mode,
    )
    return state_path, source


def test_asset_state_keeps_final_usdz_validator_out_of_module_imports() -> None:
    source = Path(asset_state.__file__).read_text(encoding="utf-8")
    module = ast.parse(source)

    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "world_understanding.utils.usd.package"
        for node in module.body
    )


def test_supporting_assets_do_not_satisfy_requested_geometry_formats() -> None:
    bundle = SimpleNamespace(
        representations=(
            SimpleNamespace(role="supporting_asset", format="mtl"),
            SimpleNamespace(role="render_geometry", format="gltf"),
        )
    )

    assert asset_state._requested_source_formats_satisfied(bundle, ("gltf",))
    assert not asset_state._requested_source_formats_satisfied(bundle, ("mtl",))


def test_revision_bundle_rebases_representations_and_provenance_inputs(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "source.usda"
    geometry.write_text("#usda 1.0\n", encoding="utf-8")
    image = tmp_path / "reference.png"
    image.write_bytes(b"reference image")

    def binding(path: Path) -> GeometryArtifactBinding:
        data = path.read_bytes()
        return GeometryArtifactBinding(
            path=path.name,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    representation = GeometryRepresentationBinding(
        representation_id="render-usd",
        role="render_geometry",
        format="usda",
        media_type="model/vnd.usda",
        artifact=binding(geometry),
    )
    provenance = GeometrySourceProvenance(
        request_digest="a" * 64,
        input_artifacts=(binding(image),),
    )
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    coordinates = GeometryCoordinateSystem(
        meters_per_unit=1.0,
        up_axis="Z",
        forward_axis="+Y",
    )
    rights = GeometryRightsAssertion(assertion="Authorized test fixture.")
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=provider,
            source_revision="revision-1",
            coordinate_system=coordinates,
            representations=(representation,),
            provenance=provenance,
            rights=rights,
        ),
        producer=provider,
        source_revision="revision-1",
        coordinate_system=coordinates,
        representations=(representation,),
        provenance=provenance,
        rights=rights,
    )
    manifest = tmp_path / "geometry.source.json"
    manifest.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")

    loaded = asset_state.load_external_source_bundle(manifest)
    rebased = asset_state._source_bundle_with_absolute_artifacts(manifest, loaded)

    assert rebased.bundle_id == bundle.bundle_id
    assert rebased.representations[0].artifact.path == str(geometry.resolve())
    assert rebased.provenance.input_artifacts[0].path == str(image.resolve())
    assert rebased.representations[0].artifact.sha256 == representation.artifact.sha256
    assert (
        rebased.provenance.input_artifacts[0].sha256
        == provenance.input_artifacts[0].sha256
    )


def test_external_source_bundle_rejects_provenance_input_drift(
    tmp_path: Path,
) -> None:
    geometry = tmp_path / "source.usda"
    geometry.write_text("#usda 1.0\n", encoding="utf-8")
    image = tmp_path / "reference.png"
    original_image = b"reference image"
    image.write_bytes(original_image)

    def binding(path: Path) -> GeometryArtifactBinding:
        data = path.read_bytes()
        return GeometryArtifactBinding(
            path=path.name,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    representation = GeometryRepresentationBinding(
        representation_id="render-usd",
        role="render_geometry",
        format="usda",
        media_type="model/vnd.usda",
        artifact=binding(geometry),
    )
    provenance = GeometrySourceProvenance(
        request_digest="a" * 64,
        input_artifacts=(binding(image),),
    )
    provider = GeometryAuthoringProviderIdentity(
        provider_id="fixture-provider",
        provider_version="1.0",
    )
    coordinates = GeometryCoordinateSystem(
        meters_per_unit=1.0,
        up_axis="Z",
        forward_axis="+Y",
    )
    rights = GeometryRightsAssertion(assertion="Authorized test fixture.")
    bundle = GeometrySourceBundle(
        bundle_id=geometry_source_bundle_id(
            provider=provider,
            source_revision="revision-1",
            coordinate_system=coordinates,
            representations=(representation,),
            provenance=provenance,
            rights=rights,
        ),
        producer=provider,
        source_revision="revision-1",
        coordinate_system=coordinates,
        representations=(representation,),
        provenance=provenance,
        rights=rights,
    )
    manifest = tmp_path / "geometry.source.json"
    manifest.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    image.write_bytes(b"changed after publication")

    with pytest.raises(ValueError, match="provenance input differs"):
        asset_state.load_external_source_bundle(manifest)


def test_create_run_preflights_unavailable_final_usdz_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def block_final_validator(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "world_understanding.utils.usd.package":
            raise ImportError("validator intentionally unavailable")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", block_final_validator)

    with pytest.raises(
        AssetCompositionStateError,
        match="required shared USDZ package validator is unavailable",
    ):
        create_run(
            tmp_path / "run" / "asset_run.json",
            run_id="validator-preflight",
            request_path=tmp_path / "missing-request.json",
            source_asset=tmp_path / "missing-source.usda",
        )


def _artifact(path: Path) -> ArtifactBinding:
    return ArtifactBinding(
        path=str(path.resolve()),
        sha256=file_sha256(path),
        size_bytes=path.stat().st_size,
    )


def test_v3_state_serializes_the_original_strict_payload_shape(tmp_path: Path) -> None:
    state_path, _source = _create(tmp_path)

    current = load_verified_run(state_path)
    legacy = current.model_copy(
        update={"schema_version": "content-agent-workflows.asset-composition-run.v3"}
    )
    payload = json.loads(legacy.model_dump_json())
    state_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert payload["schema_version"] == (
        "content-agent-workflows.asset-composition-run.v3"
    )
    for field in (
        "selected_mode",
        "execution_graph",
        "graph_started_at",
        "current_leaf_id",
        "leaf_states",
        "leaf_transitions",
        "graph_terminal_receipt",
    ):
        assert field not in payload
    assert load_verified_run(state_path).selected_mode == "compatibility_fixed"


def _staged_asset_request(tmp_path: Path) -> tuple[AssetRunRequest, ArtifactBinding]:
    original_root = tmp_path / "original"
    original_root.mkdir()
    original_source = original_root / "asset.usda"
    original_source.write_text('#usda 1.0\ndef Xform "Asset" {}\n', encoding="utf-8")

    run_dir = tmp_path / "staged-run"
    staged_source = run_dir / "inputs" / "asset_source" / "asset.usda"
    staged_source.parent.mkdir(parents=True)
    staged_source.write_bytes(original_source.read_bytes())
    original_binding = _artifact(original_source)
    staged_binding = _artifact(staged_source)
    dependency_digest_set = hashlib.sha256(
        json.dumps(
            [staged_binding.sha256],
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = run_dir / "raw" / "staged_input_asset_source.json"
    manifest_path.parent.mkdir()
    manifest_path.write_text(
        json.dumps(
            {
                "source_usd_path": original_binding.path,
                "source_sha256": original_binding.sha256,
                "staged_usd_path": staged_binding.path,
                "dependency_digest_set_sha256": dependency_digest_set,
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
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    staging = AssetSourceStaging(
        original_source=original_binding,
        original_dependencies=[original_binding],
        staged_source=staged_binding,
        staged_dependencies=[staged_binding],
        manifest=_artifact(manifest_path),
        dependency_digest_set_sha256=dependency_digest_set,
    )

    joint_config = tmp_path / "joint.yaml"
    joint_config.write_text("joint: config\n", encoding="utf-8")
    materials_yaml = tmp_path / "materials.yaml"
    materials_yaml.write_text("materials: []\n", encoding="utf-8")
    materials_usd = tmp_path / "materials.usda"
    materials_usd.write_text("#usda 1.0\n", encoding="utf-8")
    request = AssetRunRequest(
        created_at="2026-08-16T00:00:00Z",
        run_id="staged-test",
        run_dir=str(run_dir.resolve()),
        run_state=str((run_dir / "asset_run.json").resolve()),
        repository_root=str(tmp_path.resolve()),
        source_asset=staged_binding.path,
        source_staging=staging,
        prompt="Compose the staged asset.",
        joint_config=str(joint_config.resolve()),
        joint_config_binding=_artifact(joint_config),
        materials_yaml=str(materials_yaml.resolve()),
        materials_yaml_binding=_artifact(materials_yaml),
        materials_usd=str(materials_usd.resolve()),
        materials_usd_binding=_artifact(materials_usd),
        materials_usd_dependencies=asset_state.bind_usd_dependency_closure(
            materials_usd
        ),
        runtime=AssetRuntimeRequest(
            runner="codex",
            scene_tool_timeout_seconds=60,
            child_timeout_seconds=60,
            codex_sandbox_mode="workspace-write",
            claude_permission_mode="default",
            claude_execution_mode="sdk",
        ),
    )
    return request, staged_binding


def test_v4_cad_request_retains_pre_provider_policy_shape(tmp_path: Path) -> None:
    request, _ = _staged_asset_request(tmp_path)
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "schema_version": "content-agents.asset-composition-request.v4",
            "source_mode": "cad_modeling",
            "source_staging": None,
            "geometry": AssetGeometryRequest(
                target_profile="geometry-agent.static-visual-asset.v1"
            ).model_dump(mode="json"),
            "cad_modeling": {
                "model": "legacy-model",
                "model_backend": "legacy-backend",
                "quality_mode": "maximum",
                "repair_budget": 4,
                "compose_retry_budget": 2,
                "parameter_values": {"width": 24.0},
                "parameter_variants": [],
                "required_outputs": ["usd"],
            },
        }
    )

    restored = AssetRunRequest.model_validate(payload)

    assert isinstance(restored.cad_modeling, LegacyAssetCadModelingRequest)
    assert restored.cad_modeling.model == "legacy-model"
    assert restored.model_dump(mode="json")["cad_modeling"] == payload["cad_modeling"]
    with pytest.raises(
        ValidationError,
        match="v5 requires provider-delegated CAD modeling policy",
    ):
        AssetRunRequest.model_validate(
            {
                **payload,
                "schema_version": "content-agents.asset-composition-request.v5",
            }
        )


def test_v5_cad_request_requires_provider_delegated_policy(tmp_path: Path) -> None:
    request, _ = _staged_asset_request(tmp_path)
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "schema_version": "content-agents.asset-composition-request.v5",
            "source_mode": "cad_modeling",
            "source_staging": None,
            "geometry": AssetGeometryRequest(
                target_profile="geometry-agent.static-visual-asset.v1"
            ).model_dump(mode="json"),
            "cad_modeling": {
                "provider_id": "selected-authoring-provider",
                "target_profile": "geometry-agent.static-visual-asset.v1",
                "parameter_values": {"width": 24.0},
                "parameter_variants": [],
                "required_outputs": ["usd"],
            },
        }
    )

    restored = AssetRunRequest.model_validate(payload)

    assert restored.cad_modeling is not None
    assert restored.cad_modeling.provider_id == "selected-authoring-provider"
    with pytest.raises(
        ValidationError,
        match="v4 requires the pre-provider CAD modeling policy",
    ):
        AssetRunRequest.model_validate(
            {
                **payload,
                "schema_version": "content-agents.asset-composition-request.v4",
            }
        )


def _staged_non_usd_asset_request(
    tmp_path: Path,
) -> tuple[AssetRunRequest, ArtifactBinding]:
    request, _ = _staged_asset_request(tmp_path)
    original_root = tmp_path / "non-usd-original"
    original_mesh = original_root / "meshes" / "link.stl"
    original_mesh.parent.mkdir(parents=True)
    original_source = original_root / "model.urdf"
    original_source.write_text(
        '<robot name="fixture"><link name="base"><visual><geometry>'
        '<mesh filename="meshes/link.stl"/></geometry></visual></link></robot>\n',
        encoding="utf-8",
    )
    original_mesh.write_text("solid link\nendsolid link\n", encoding="utf-8")

    run_dir = Path(request.run_dir)
    staged_root = run_dir / "inputs" / "asset_source"
    staged_source = staged_root / "model.urdf"
    staged_mesh = staged_root / "meshes" / "link.stl"
    staged_source.write_bytes(original_source.read_bytes())
    staged_mesh.parent.mkdir(parents=True)
    staged_mesh.write_bytes(original_mesh.read_bytes())
    original_bindings = sorted(
        [_artifact(original_source), _artifact(original_mesh)],
        key=lambda binding: binding.path,
    )
    staged_bindings = sorted(
        [_artifact(staged_source), _artifact(staged_mesh)],
        key=lambda binding: binding.path,
    )
    original_source_binding = next(
        binding for binding in original_bindings if binding.path == str(original_source)
    )
    staged_source_binding = next(
        binding for binding in staged_bindings if binding.path == str(staged_source)
    )
    dependency_digest_set = hashlib.sha256(
        json.dumps(
            sorted(binding.sha256 for binding in staged_bindings),
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    manifest_files = []
    for original_binding, staged_binding in zip(
        original_bindings,
        staged_bindings,
        strict=True,
    ):
        manifest_files.append(
            {
                "source_path": original_binding.path,
                "staged_path": staged_binding.path,
                "relative_path": Path(staged_binding.path)
                .relative_to(staged_root)
                .as_posix(),
                "sha256": staged_binding.sha256,
                "size_bytes": staged_binding.size_bytes,
            }
        )
    manifest_path = run_dir / "raw" / "staged_source_asset_source.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": ("content-workflow-cli.immutable-source-closure.v1"),
                "source_path": original_source_binding.path,
                "source_sha256": original_source_binding.sha256,
                "staged_path": staged_source_binding.path,
                "common_source_root": str(original_root),
                "dependency_discovery_strategy": "urdf_xml_reference_graph",
                "file_count": len(manifest_files),
                "total_size_bytes": sum(
                    binding.size_bytes for binding in staged_bindings
                ),
                "dependency_digest_set_sha256": dependency_digest_set,
                "unresolved_dependencies": [],
                "files": manifest_files,
                "self_containment": {
                    "status": "verified",
                    "escaped_paths": [],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    staging = AssetSourceStaging(
        original_source=original_source_binding,
        original_dependencies=original_bindings,
        staged_source=staged_source_binding,
        staged_dependencies=staged_bindings,
        manifest=_artifact(manifest_path),
        dependency_digest_set_sha256=dependency_digest_set,
    )
    return (
        request.model_copy(
            update={
                "source_asset": staged_source_binding.path,
                "source_staging": staging,
            }
        ),
        staged_source_binding,
    )


def _replace_staging_manifest(
    request: AssetRunRequest,
    payload: object,
    *,
    raw_text: str | None = None,
) -> AssetRunRequest:
    assert request.source_staging is not None
    manifest_path = Path(request.source_staging.manifest.path)
    manifest_path.write_text(
        raw_text if raw_text is not None else json.dumps(payload) + "\n",
        encoding="utf-8",
    )
    staging = request.source_staging.model_copy(
        update={"manifest": _artifact(manifest_path)}
    )
    return request.model_copy(update={"source_staging": staging})


@pytest.mark.parametrize(
    (
        "original_source",
        "original_dependencies",
        "staged_source",
        "staged_dependencies",
        "error",
    ),
    [
        ("a", ["a"], "b", ["b"], "dependencies differ"),
        ("a", ["b"], "b", ["b"], "original source is missing"),
        ("a", ["a"], "b", ["a"], "staged source is missing"),
    ],
)
def test_asset_source_staging_rejects_inconsistent_dependency_identity(
    original_source: str,
    original_dependencies: list[str],
    staged_source: str,
    staged_dependencies: list[str],
    error: str,
) -> None:
    def binding(name: str) -> dict[str, object]:
        return {"path": f"/{name}", "sha256": name * 64, "size_bytes": 1}

    with pytest.raises(ValidationError, match=error):
        AssetSourceStaging.model_validate(
            {
                "original_source": binding(original_source),
                "original_dependencies": [
                    binding(item) for item in original_dependencies
                ],
                "staged_source": binding(staged_source),
                "staged_dependencies": [binding(item) for item in staged_dependencies],
                "manifest": binding("c"),
                "dependency_digest_set_sha256": "d" * 64,
            }
        )


@pytest.mark.parametrize(
    ("schema_version", "source_update", "drop_staging", "error"),
    [
        (
            "content-agents.asset-composition-request.v2",
            "/different/source.usda",
            False,
            "source must be the staged run-confined source",
        ),
        (
            "content-agents.asset-composition-request.v2",
            None,
            True,
            "v2 requires source staging identity",
        ),
        (
            "content-agents.asset-composition-request.v1",
            None,
            False,
            "legacy asset request cannot carry v2 source staging",
        ),
    ],
)
def test_asset_request_schema_binds_staging_to_v2_only(
    tmp_path: Path,
    schema_version: str,
    source_update: str | None,
    drop_staging: bool,
    error: str,
) -> None:
    request, _ = _staged_asset_request(tmp_path)
    payload = request.model_dump(mode="json")
    payload["schema_version"] = schema_version
    if source_update is not None:
        payload["source_asset"] = source_update
    if drop_staging:
        payload["source_staging"] = None

    with pytest.raises(ValidationError, match=error):
        AssetRunRequest.model_validate(payload)


def test_verify_frozen_asset_inputs_accepts_exact_staged_manifest(
    tmp_path: Path,
) -> None:
    request, run_source = _staged_asset_request(tmp_path)

    asset_state.verify_frozen_asset_inputs(request, run_source=run_source)


def test_non_usd_staging_manifest_binds_durable_dependency_closure(
    tmp_path: Path,
) -> None:
    request, run_source = _staged_non_usd_asset_request(tmp_path)
    assert request.source_staging is not None
    source_dependencies = [
        binding
        for binding in request.source_staging.staged_dependencies
        if binding.path != run_source.path
    ]

    asset_state.verify_frozen_asset_inputs(
        request,
        run_source=run_source,
        run_source_dependencies=source_dependencies,
    )
    request_path = Path(request.run_dir) / "request.json"
    request_path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
    asset_state.create_run(
        request.run_state,
        run_id=request.run_id,
        request_path=request_path,
        source_asset=request.source_asset,
    )
    run = load_verified_run(request.run_state)
    assert run.source_asset == run_source
    assert run.source_dependencies == source_dependencies


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("file_count", 1),
        ("total_size_bytes", 1),
        ("dependency_discovery_strategy", ""),
        ("self_containment", {"status": "verified", "escaped_paths": ["escape"]}),
    ],
)
def test_non_usd_staging_rejects_false_closure_metadata(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    request, run_source = _staged_non_usd_asset_request(tmp_path)
    assert request.source_staging is not None
    manifest_path = Path(request.source_staging.manifest.path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = value
    request = _replace_staging_manifest(request, payload)

    with pytest.raises(AssetCompositionStateError, match="manifest identity changed"):
        asset_state.verify_frozen_asset_inputs(request, run_source=run_source)


def test_verified_run_closure_is_not_rehashed_as_request_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, run_source = _staged_asset_request(tmp_path)
    original_verify = asset_state._verify_binding
    labels: list[str] = []

    def observe(
        binding: ArtifactBinding,
        *,
        label: str,
        required_root: Path | None = None,
    ) -> None:
        labels.append(label)
        original_verify(binding, label=label, required_root=required_root)

    monkeypatch.setattr(asset_state, "_verify_binding", observe)

    asset_state.verify_frozen_asset_inputs(
        request,
        run_source=run_source,
        run_source_dependencies=[],
    )

    assert "staged source manifest" in labels
    assert not any(label.startswith("staged source artifact") for label in labels)


def test_v2_run_rejects_source_closure_outside_run_root(tmp_path: Path) -> None:
    request, _ = _staged_asset_request(tmp_path)
    assert request.source_staging is not None
    staging = request.source_staging.model_copy(
        update={
            "staged_source": request.source_staging.original_source,
            "staged_dependencies": request.source_staging.original_dependencies,
        }
    )
    request = request.model_copy(
        update={
            "source_asset": staging.staged_source.path,
            "source_staging": staging,
        }
    )
    request_path = Path(request.run_dir) / "request.json"
    request_path.write_text(
        request.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    asset_state.create_run(
        request.run_state,
        run_id=request.run_id,
        request_path=request_path,
        source_asset=staging.staged_source.path,
    )

    with pytest.raises(AssetCompositionStateError, match="must be inside"):
        load_verified_run(request.run_state)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("duplicate_paths", "dependency paths are not unique"),
        ("digest_set", "dependency digest set changed"),
    ],
)
def test_verify_frozen_asset_inputs_rejects_staging_identity_changes(
    tmp_path: Path,
    mutation: str,
    error: str,
) -> None:
    request, run_source = _staged_asset_request(tmp_path)
    assert request.source_staging is not None
    if mutation == "duplicate_paths":
        staging = request.source_staging.model_copy(
            update={
                "staged_dependencies": [
                    request.source_staging.staged_source,
                    request.source_staging.staged_source,
                ]
            }
        )
    else:
        staging = request.source_staging.model_copy(
            update={"dependency_digest_set_sha256": "f" * 64}
        )
    request = request.model_copy(update={"source_staging": staging})

    with pytest.raises(AssetCompositionStateError, match=error):
        asset_state.verify_frozen_asset_inputs(request, run_source=run_source)


@pytest.mark.parametrize(
    ("payload", "raw_text", "error"),
    [
        ({}, "{not-json", "manifest is invalid"),
        ({"files": "not-a-list"}, None, "manifest omitted dependency files"),
        ({"files": [{}]}, None, "manifest has malformed dependency identity"),
        (
            {
                "source_usd_path": "wrong",
                "source_sha256": "0" * 64,
                "staged_usd_path": "wrong",
                "dependency_digest_set_sha256": "0" * 64,
                "unresolved_dependencies": [],
                "files": [],
            },
            None,
            "manifest identity changed",
        ),
    ],
)
def test_verify_frozen_asset_inputs_rejects_invalid_staging_manifest(
    tmp_path: Path,
    payload: object,
    raw_text: str | None,
    error: str,
) -> None:
    request, run_source = _staged_asset_request(tmp_path)
    request = _replace_staging_manifest(request, payload, raw_text=raw_text)

    with pytest.raises(AssetCompositionStateError, match=error):
        asset_state.verify_frozen_asset_inputs(request, run_source=run_source)


def test_verify_frozen_asset_inputs_rejects_non_object_staging_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, run_source = _staged_asset_request(tmp_path)
    monkeypatch.setattr(asset_state, "load_json", lambda _path: [])

    with pytest.raises(
        AssetCompositionStateError, match="manifest is not a JSON object"
    ):
        asset_state.verify_frozen_asset_inputs(request, run_source=run_source)


def _plan_articulation_stage(state_path: Path) -> None:
    plan = state_path.parent / "articulation-plan-draft.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-plan-draft.v1"
                ),
                "stage": "articulation",
                "objective": "Inspect exact articulation evidence.",
                "steps": [
                    {
                        "stage": "articulation",
                        "objective": "Review the outer canonical graph.",
                        "acceptance_evidence": ["digest-bound canonical graph"],
                        "may_revisit": True,
                    }
                ],
                "evidence_paths": [str(state_path.parent / "request.json")],
                "revision_reason": "Exercise the supported coordinator gate.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_plan(state_path, plan_path=plan, actor="test-coordinator")


def _record_articulation_await_review(
    state_path: Path,
    *,
    candidates_path: Path,
) -> None:
    review = state_path.parent / "articulation-review-draft.json"
    review.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.asset-coordinator-review-draft.v1"
                ),
                "stage": "articulation",
                "output_asset_path": None,
                "evidence_paths": [str(candidates_path)],
                "findings": ["The exact outer graph requires human review."],
                "decision": "await_review",
                "target_stage": None,
                "decision_summary": "Pause at the exact Joint human gate.",
                "repair_scope": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    record_coordinator_evidence_review(
        state_path,
        review_path=review,
        actor="test-coordinator",
    )


def _stage_artifacts(state_path: Path, stage: str) -> tuple[Path, Path]:
    directory = stage_directory(state_path, stage)  # type: ignore[arg-type]
    directory.mkdir(parents=True, exist_ok=True)
    output = directory / "output.usda"
    output.write_text(f"#usda 1.0\n# {stage}\n", encoding="utf-8")
    evidence = directory / (
        "final_summary.json" if stage == "validation" else "evidence.json"
    )
    payload: dict[str, object] = {"stage": stage}
    if stage == "physics":
        payload = _physics_evidence_payload(output)
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return output, evidence


def _physics_evidence_payload(output: Path) -> dict[str, object]:
    evidence = physics_validation_evidence(
        asset=str(output.resolve()),
        target_runtime="fake",
        physics_properties_status="pass",
        runtime_loadability_status="pass",
        no_explosions_status="pass",
    )
    evidence.metadata["asset_sha256"] = file_sha256(output)
    return evidence.model_dump(mode="json")


def _refresh_physics_evidence(evidence: Path, output: Path) -> None:
    evidence.write_text(
        json.dumps(_physics_evidence_payload(output)) + "\n",
        encoding="utf-8",
    )


def _approve_articulation(state_path: Path) -> None:
    directory = stage_directory(state_path, "articulation")
    directory.mkdir(parents=True, exist_ok=True)
    candidates = directory / "candidates.json"
    candidates.write_text('{"joints":[]}\n', encoding="utf-8")
    require_review(state_path, candidates_path=candidates)
    decisions = state_path.parent / "operator-decisions.json"
    decisions.write_text('{"accepted":[]}\n', encoding="utf-8")
    record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="reviewer@example.com",
    )
    begin_stage(state_path, "articulation")


def _advance_to_physics(state_path: Path) -> None:
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    for stage, successor in (
        ("articulation", "material"),
        ("material", "texture"),
        ("texture", "physics"),
    ):
        output, evidence = _stage_artifacts(state_path, stage)
        complete_stage(
            state_path,
            stage,  # type: ignore[arg-type]
            output_asset=output,
            evidence_paths=[evidence],
            summary=f"Completed {stage}.",
        )
        begin_stage(state_path, successor)  # type: ignore[arg-type]


def _write_jointed_physics_output(
    output: Path,
    *,
    descendant_targets_under_one_body: bool,
) -> None:
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics

    output.unlink(missing_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.Xform.Define(stage, "/World")
    if descendant_targets_under_one_body:
        cabinet = UsdGeom.Xform.Define(stage, "/World/Cabinet").GetPrim()
        body_path = "/World/Cabinet/Body"
        drawer_path = "/World/Cabinet/Drawer"
        UsdPhysics.RigidBodyAPI.Apply(cabinet).CreateRigidBodyEnabledAttr(True)
    else:
        body_path = "/World/Cabinet"
        drawer_path = "/World/Drawer"
    body = UsdGeom.Xform.Define(stage, body_path).GetPrim()
    drawer = UsdGeom.Xform.Define(stage, drawer_path).GetPrim()
    if not descendant_targets_under_one_body:
        UsdPhysics.RigidBodyAPI.Apply(body).CreateRigidBodyEnabledAttr(True)
        UsdPhysics.RigidBodyAPI.Apply(drawer).CreateRigidBodyEnabledAttr(True)
    joint = UsdPhysics.PrismaticJoint.Define(stage, "/World/DrawerJoint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(body_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(drawer_path)])
    stage.GetRootLayer().Save()


def _complete_all(state_path: Path) -> None:
    for stage in LEGACY_STAGE_ORDER:
        begin_stage(state_path, stage)
        if stage == "articulation":
            _approve_articulation(state_path)
        output, evidence = _stage_artifacts(state_path, stage)
        evidence_paths = [evidence]
        if stage == "finalization":
            output.unlink()
            package_source = output.parent / "package-source"
            package_source.mkdir()
            package_root = package_source / "asset.usda"
            package_root.write_text("#usda 1.0\n", encoding="utf-8")
            output = output.with_suffix(".usdz")
            write_usdz_package_from_directory(
                package_source,
                Path("asset.usda"),
                output,
            )
            validation_summary = (
                load_verified_run(state_path).stages["validation"].evidence[0]
            )
            report_path = output.parent / "combined_report.json"
            report = build_combined_report(
                state_path,
                final_asset=output,
                validation_summary=validation_summary.path,
                output_path=report_path,
            )
            assert (
                build_combined_report(
                    state_path,
                    final_asset=output,
                    validation_summary=validation_summary.path,
                    output_path=report_path,
                )
                == report
            )
            evidence_paths.append(Path(report.path))
        complete_stage(
            state_path,
            stage,
            output_asset=output,
            evidence_paths=evidence_paths,
            summary=f"Completed {stage}.",
        )


def test_create_binds_request_and_source_and_starts_articulation(
    tmp_path: Path,
) -> None:
    state_path, source = _create(tmp_path)

    run = load_verified_run(state_path)

    assert run.current_stage == "articulation"
    assert run.stages["articulation"].status == "ready"
    assert run.stages["articulation"].input_asset == run.source_asset
    assert Path(run.source_asset.path) == source.resolve()
    assert all(
        run.stages[stage].status == "pending" for stage in LEGACY_STAGE_ORDER[1:]
    )


def test_write_run_revalidates_before_replacing_durable_state(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    run = load_verified_run(state_path)
    original = state_path.read_bytes()
    run.stages["articulation"].status = "completed"

    with pytest.raises(AssetCompositionStateError, match="invalid composed"):
        asset_state._write_run(state_path, run)

    assert state_path.read_bytes() == original


def test_begin_stage_creates_canonical_stage_directory(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)

    directory = stage_directory(state_path, "articulation")
    assert not directory.exists()

    begin_stage(state_path, "articulation")

    assert directory.is_dir()


def test_begin_stage_rejects_redirected_stages_root(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    (state_path.parent / "stages").symlink_to(redirected, target_is_directory=True)

    with pytest.raises(AssetCompositionStateError, match="stages root"):
        begin_stage(state_path, "articulation")


def test_articulation_review_blocks_authoring_until_exact_decisions_are_bound(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    directory = stage_directory(state_path, "articulation")
    directory.mkdir(parents=True, exist_ok=True)
    candidates = directory / "candidates.json"
    candidates.write_text('{"joints":[{"id":"drawer-1"}]}\n', encoding="utf-8")
    require_review(state_path, candidates_path=candidates)

    with pytest.raises(AssetCompositionStateError, match="expected running"):
        complete_stage(
            state_path,
            "articulation",
            output_asset=candidates,
            evidence_paths=[candidates],
            summary="Should not complete.",
        )

    decisions = tmp_path / "decisions.json"
    decision_bytes = b'{"accepted":["drawer-1"]}\n'
    decisions.write_bytes(decision_bytes)
    run = record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="joint-reviewer",
    )

    persisted = run.stages["articulation"].review_decisions
    assert persisted is not None
    assert Path(persisted.path).read_bytes() == decision_bytes
    assert run.stages["articulation"].status == "ready"


def test_review_decisions_persist_one_stably_captured_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    candidates = stage_directory(state_path, "articulation") / "candidates.json"
    candidates.write_text('{"joints":[{"id":"drawer-1"}]}\n', encoding="utf-8")
    require_review(state_path, candidates_path=candidates)
    decisions = tmp_path / "decisions.json"
    captured_bytes = b'{"accepted":["drawer-1"]}\n'
    decisions.write_bytes(captured_bytes)
    original_reader = asset_state._read_stable_regular_bytes

    def read_then_mutate(path: Path, *, label: str) -> bytes:
        payload = original_reader(path, label=label)
        path.write_text('{"accepted":[]}\n', encoding="utf-8")
        return payload

    monkeypatch.setattr(asset_state, "_read_stable_regular_bytes", read_then_mutate)

    run = record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="joint-reviewer",
    )

    persisted = run.stages["articulation"].review_decisions
    assert persisted is not None
    assert Path(persisted.path).read_bytes() == captured_bytes
    assert persisted.sha256 == file_sha256(Path(persisted.path))


def test_stable_prefix_reader_never_loads_the_remaining_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "large.usdc"
    source.write_bytes(b"PXR-USDC" + b"x" * (2 * 1024 * 1024))
    original_read = asset_state.os.read
    requested_sizes: list[int] = []

    def bounded_read(descriptor: int, size: int) -> bytes:
        requested_sizes.append(size)
        return original_read(descriptor, size)

    monkeypatch.setattr(asset_state.os, "read", bounded_read)

    assert (
        asset_state._read_stable_regular_prefix(
            source,
            label="test USDC",
            size=8,
        )
        == b"PXR-USDC"
    )
    assert requested_sizes == [8]


def test_embedded_articulation_human_gate_binds_exact_canonical_graph_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from content_agent_workflows.articulation import (
        EmbeddedArticulationCanonicalGraph,
        EmbeddedArticulationGroup,
        EmbeddedArticulationJoint,
        EmbeddedArticulationMembership,
    )
    from content_agent_workflows.common.artifacts import atomic_write_json
    from content_agent_workflows.common.embedded_domain_decision import (
        canonical_json_digest,
    )

    monkeypatch.setattr(
        asset_state,
        "_physics_validation_mode_from_run",
        lambda _run: "runtime_required",
    )
    monkeypatch.setattr(
        asset_state,
        "load_verified_asset_request",
        lambda *_args, **_kwargs: None,
    )
    state_path, _ = _create(tmp_path, coordinator_mode="single_reasoning_loop")
    _plan_articulation_stage(state_path)
    begin_stage(state_path, "articulation")
    graph = EmbeddedArticulationCanonicalGraph(
        graph_id="neutral-human-gate",
        source_sha256="1" * 64,
        source_dependency_bundle_sha256="2" * 64,
        source_member_prims=("/Structure/Root", "/Structure/MovingMember"),
        authoritative_owner_prims=("/Structure/Root", "/Structure/MovingMember"),
        candidate_ids=("motion_1",),
        groups=(
            EmbeddedArticulationGroup(
                group_id="root_group",
                authoritative_owner_prim="/Structure/Root",
                member_prims=("/Structure/Root",),
                role="base",
                role_state="known",
                evidence_ids=("inspection",),
            ),
            EmbeddedArticulationGroup(
                group_id="moving_group",
                authoritative_owner_prim="/Structure/MovingMember",
                member_prims=("/Structure/MovingMember",),
                role="moving_member",
                role_state="known",
                evidence_ids=("inspection",),
            ),
        ),
        memberships=(
            EmbeddedArticulationMembership(
                member_prim="/Structure/Root",
                authoritative_owner_prim="/Structure/Root",
                group_id="root_group",
                disposition="co_rigid",
                state="source_backed",
                evidence_ids=("inspection",),
            ),
            EmbeddedArticulationMembership(
                member_prim="/Structure/MovingMember",
                authoritative_owner_prim="/Structure/MovingMember",
                group_id="moving_group",
                disposition="independent_motion",
                state="source_backed",
                evidence_ids=("inspection",),
            ),
        ),
        joints=(
            EmbeddedArticulationJoint(
                joint_id="motion_1",
                body0_owner_prim="/Structure/Root",
                body1_owner_prim="/Structure/MovingMember",
                body0_role="base",
                body1_role="moving_member",
                role_state="known",
                joint_type="revolute",
                endpoint_state="source_backed",
                type_state="known",
                axis="z",
                axis_state="known",
                limit_unit="degrees",
                limit_state="known",
                frame_policy="body1_world_origin",
                frame_state="known",
                evidence_ids=("inspection",),
            ),
        ),
    )
    graph_path = stage_directory(state_path, "articulation") / "canonical_graph.json"
    atomic_write_json(graph_path, graph)
    with pytest.raises(
        AssetCompositionStateError,
        match="require a coordinator await_review decision",
    ):
        require_review(state_path, candidates_path=graph_path)
    _record_articulation_await_review(
        state_path,
        candidates_path=graph_path,
    )
    require_review(state_path, candidates_path=graph_path)
    decisions = tmp_path / "graph-decisions.json"
    decisions.write_text('{"motion_1":"revise"}\n', encoding="utf-8")
    record_review_decisions(
        state_path,
        decisions_path=decisions,
        reviewer="graph-reviewer",
    )

    acceptance = load_embedded_articulation_human_acceptance(
        state_path,
        canonical_graph=graph_path,
    )
    assert acceptance is not None
    assert acceptance.canonical_graph_sha256 == canonical_json_digest(graph)
    assert acceptance.decisions == {"motion_1": "revise"}
    assert acceptance.reviewer == "graph-reviewer"

    negative_root = tmp_path / "non-graph"
    negative_root.mkdir()
    negative_state, _ = _create(
        negative_root,
        coordinator_mode="single_reasoning_loop",
    )
    _plan_articulation_stage(negative_state)
    begin_stage(negative_state, "articulation")
    non_graph_candidates = (
        stage_directory(negative_state, "articulation") / "candidates.json"
    )
    atomic_write_json(
        non_graph_candidates,
        {
            "schema_version": "joint-agent-stage2-v0",
            "candidate_ids": ["motion_1"],
        },
    )
    mismatched_candidates = (
        stage_directory(negative_state, "articulation") / "mismatched-candidates.json"
    )
    atomic_write_json(
        mismatched_candidates,
        {
            "schema_version": "joint-agent-stage2-v0",
            "candidate_ids": ["other_motion"],
        },
    )
    _record_articulation_await_review(
        negative_state,
        candidates_path=non_graph_candidates,
    )
    with pytest.raises(
        AssetCompositionStateError,
        match="does not bind these Joint candidates",
    ):
        require_review(
            negative_state,
            candidates_path=mismatched_candidates,
        )
    require_review(negative_state, candidates_path=non_graph_candidates)
    negative_decisions = negative_root / "decisions.json"
    negative_decisions.write_text('{"motion_1":"revise"}\n', encoding="utf-8")
    with pytest.raises(AssetCompositionStateError, match="accept.*reject"):
        record_review_decisions(
            negative_state,
            decisions_path=negative_decisions,
            reviewer="graph-reviewer",
        )


def test_every_handoff_becomes_the_exact_successor_input_and_terminal_is_valid(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)

    _complete_all(state_path)

    run = load_verified_run(state_path)
    terminal = validate_terminal(state_path)
    assert terminal.valid
    assert run.terminal_status == "completed"
    assert run.current_stage is None
    for index, stage in enumerate(LEGACY_STAGE_ORDER[:-1]):
        successor = LEGACY_STAGE_ORDER[index + 1]
        assert run.stages[successor].input_asset == run.stages[stage].output_asset
    assert terminal.final_asset == run.stages["finalization"].output_asset


def test_finalization_rejects_mismatched_report_before_terminal_persistence(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    for stage in LEGACY_STAGE_ORDER[:-1]:
        begin_stage(state_path, stage)
        if stage == "articulation":
            _approve_articulation(state_path)
        output, evidence = _stage_artifacts(state_path, stage)
        complete_stage(
            state_path,
            stage,
            output_asset=output,
            evidence_paths=[evidence],
            summary=f"Completed {stage}.",
        )

    begin_stage(state_path, "finalization")
    final_dir = stage_directory(state_path, "finalization")
    package_source = final_dir / "package-source"
    package_source.mkdir()
    (package_source / "asset.usda").write_text("#usda 1.0\n", encoding="utf-8")
    output = final_dir / "asset.usdz"
    write_usdz_package_from_directory(package_source, Path("asset.usda"), output)
    validation_summary = load_verified_run(state_path).stages["validation"].evidence[0]
    report_path = final_dir / "combined_report.json"
    build_combined_report(
        state_path,
        final_asset=output,
        validation_summary=validation_summary.path,
        output_path=report_path,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    payload["run_id"] = "another-run"
    report_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="Combined final report does not match accepted workflow state",
    ):
        complete_stage(
            state_path,
            "finalization",
            output_asset=output,
            evidence_paths=[report_path],
            summary="Reject stale combined report.",
        )

    run = load_verified_run(state_path)
    assert run.terminal_status == "active"
    assert run.current_stage == "finalization"
    assert run.stages["finalization"].status == "running"


@pytest.mark.parametrize("target", ["request", "source", "output", "evidence"])
def test_mutated_accepted_artifact_fails_closed(tmp_path: Path, target: str) -> None:
    state_path, source = _create(tmp_path)
    run = begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    output, evidence = _stage_artifacts(state_path, "articulation")
    complete_stage(
        state_path,
        "articulation",
        output_asset=output,
        evidence_paths=[evidence],
        summary="Reviewed articulation output.",
    )
    if target == "request":
        path = Path(run.request.path)
    elif target == "source":
        path = source
    elif target == "output":
        path = output
    else:
        path = evidence
    path.write_bytes(path.read_bytes() + b"mutated")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_run(state_path)


def test_handoff_claims_are_revalidated_not_only_rehashed(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    output, evidence = _stage_artifacts(state_path, "articulation")
    run = complete_stage(
        state_path,
        "articulation",
        output_asset=output,
        evidence_paths=[evidence],
        summary="Reviewed articulation output.",
    )
    handoff = Path(run.stages["articulation"].handoff.path)  # type: ignore[union-attr]
    payload = json.loads(handoff.read_text(encoding="utf-8"))
    payload["stage"] = "material"
    handoff.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match="identity changed"):
        load_verified_run(state_path)


def test_outputs_are_confined_single_link_regular_files(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    external = tmp_path / "external.usda"
    external.write_text("#usda 1.0\n", encoding="utf-8")
    _, evidence = _stage_artifacts(state_path, "articulation")

    with pytest.raises(AssetCompositionStateError, match="inside"):
        complete_stage(
            state_path,
            "articulation",
            output_asset=external,
            evidence_paths=[evidence],
            summary="Invalid output.",
        )

    output, evidence = _stage_artifacts(state_path, "articulation")
    linked = state_path.parent / "linked-output.usda"
    os.link(output, linked)
    with pytest.raises(AssetCompositionStateError, match="hard-linked"):
        complete_stage(
            state_path,
            "articulation",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid output.",
        )


@pytest.mark.skipif(
    os.name == "nt",
    reason="native Windows uses confined handles instead of O_NOFOLLOW",
)
def test_binding_fails_closed_without_no_follow_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value":1}\n', encoding="utf-8")
    monkeypatch.delattr(asset_state.os, "O_NOFOLLOW", raising=False)

    with pytest.raises(AssetCompositionStateError, match="O_NOFOLLOW"):
        asset_state._binding(artifact, label="test artifact")


@pytest.mark.skipif(os.name != "nt", reason="native Windows confined-handle path")
def test_binding_uses_confined_windows_reader_without_o_nofollow(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value":1}\n', encoding="utf-8")

    binding = asset_state._binding(artifact, label="test artifact")

    assert binding.path == str(artifact.resolve())
    assert binding.size_bytes == artifact.stat().st_size


def test_binding_rejects_path_replacement_during_descriptor_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text('{"value":1}\n', encoding="utf-8")
    replacement = tmp_path / "replacement.json"
    replacement.write_text('{"value":2}\n', encoding="utf-8")
    moved = tmp_path / "artifact-original.json"
    real_read = asset_state.os.read
    replaced = False

    def replace_after_first_read(descriptor: int, size: int) -> bytes:
        nonlocal replaced
        chunk = real_read(descriptor, size)
        if not replaced:
            artifact.rename(moved)
            artifact.symlink_to(replacement)
            replaced = True
        return chunk

    monkeypatch.setattr(asset_state.os, "read", replace_after_first_read)

    with pytest.raises(AssetCompositionStateError, match="changed while being read"):
        asset_state._binding(artifact, label="test artifact")


def test_stage_completion_rejects_unresolved_usd_dependencies(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    output, evidence = _stage_artifacts(state_path, "articulation")
    output.write_text(
        '#usda 1.0\ndef Xform "World"\n{\n    custom asset missing = '
        "@missing.png@\n}\n",
        encoding="utf-8",
    )

    with pytest.raises(AssetCompositionStateError, match="unresolved USD dependencies"):
        complete_stage(
            state_path,
            "articulation",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid dependency closure.",
        )


def test_source_dependency_bytes_are_bound_and_reverified(tmp_path: Path) -> None:
    source = tmp_path / "source.usda"
    dependency = tmp_path / "geometry.usda"
    dependency.write_text('#usda 1.0\ndef Xform "Geometry" {}\n', encoding="utf-8")
    source.write_text(
        "#usda 1.0\n(\n    subLayers = [@geometry.usda@]\n)\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    request = run_dir / "request.json"
    request.write_text('{"prompt":"compose asset"}\n', encoding="utf-8")
    state_path = run_dir / "asset_run.json"

    run = create_run(
        state_path,
        run_id="dependency-test",
        request_path=request,
        source_asset=source,
    )

    assert [binding.path for binding in run.source_dependencies] == [
        str(dependency.resolve())
    ]
    dependency.write_text(
        '#usda 1.0\ndef Xform "ChangedGeometry" {}\n',
        encoding="utf-8",
    )
    with pytest.raises(AssetCompositionStateError, match="dependency closure"):
        load_verified_run(state_path)


def test_stage_output_dependency_bytes_are_bound_and_reverified(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)
    output, evidence = _stage_artifacts(state_path, "articulation")
    texture = output.parent / "texture.png"
    texture.write_bytes(b"texture-v1")
    output.write_text(
        '#usda 1.0\ndef Xform "World"\n{\n    custom asset texture = '
        "@texture.png@\n}\n",
        encoding="utf-8",
    )

    run = complete_stage(
        state_path,
        "articulation",
        output_asset=output,
        evidence_paths=[evidence],
        summary="Reviewed dependency-bound output.",
    )

    assert [
        binding.path for binding in run.stages["articulation"].output_dependencies
    ] == [str(texture.resolve())]
    texture.write_bytes(b"texture-v2")
    with pytest.raises(AssetCompositionStateError, match="dependency closure"):
        load_verified_run(state_path)


def test_physics_preserves_self_contained_usdz_handoff(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    _approve_articulation(state_path)

    for stage in ("articulation", "material"):
        output, evidence = _stage_artifacts(state_path, stage)
        complete_stage(
            state_path,
            stage,  # type: ignore[arg-type]
            output_asset=output,
            evidence_paths=[evidence],
            summary=f"Completed {stage}.",
        )
        begin_stage(state_path, "material" if stage == "articulation" else "texture")

    texture_dir = stage_directory(state_path, "texture")
    package_source = texture_dir / "package-source"
    package_source.mkdir()
    package_root = package_source / "asset.usda"
    package_root.write_text("#usda 1.0\n", encoding="utf-8")
    texture_output = texture_dir / "textured.usdz"
    write_usdz_package_from_directory(
        package_source,
        Path("asset.usda"),
        texture_output,
    )
    texture_evidence = texture_dir / "evidence.json"
    texture_evidence.write_text('{"stage":"texture"}\n', encoding="utf-8")
    complete_stage(
        state_path,
        "texture",
        output_asset=texture_output,
        evidence_paths=[texture_evidence],
        summary="Completed texture.",
    )
    begin_stage(state_path, "physics")
    physics_output, physics_evidence = _stage_artifacts(state_path, "physics")

    with pytest.raises(AssetCompositionStateError, match="must remain.*USDZ"):
        complete_stage(
            state_path,
            "physics",
            output_asset=physics_output,
            evidence_paths=[physics_evidence],
            summary="Invalid unpackaged Physics output.",
        )


def test_physics_rejects_conditional_native_runtime_evidence(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["sim_ready_status"] = "conditional"
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="native runtime validation must pass",
    ):
        complete_stage(
            state_path,
            "physics",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid conditional Physics output.",
        )


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ([], "has no checks"),
        (["not-a-check"], "native validation evidence is invalid"),
    ],
)
def test_physics_rejects_malformed_native_runtime_checks(
    tmp_path: Path,
    checks: list[object],
    message: str,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["checks"] = checks
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(AssetCompositionStateError, match=message):
        complete_stage(
            state_path,
            "physics",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid Physics checks.",
        )


def test_physics_rejects_native_evidence_for_another_output(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["asset"] = str((output.parent / "other.usda").resolve())
    evidence.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    with pytest.raises(
        AssetCompositionStateError,
        match="not bound to the accepted output",
    ):
        complete_stage(
            state_path,
            "physics",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid unrelated Physics evidence.",
        )


def test_physics_rejects_joint_targets_below_one_root_rigid_body(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    _write_jointed_physics_output(
        output,
        descendant_targets_under_one_body=True,
    )
    _refresh_physics_evidence(evidence, output)

    with pytest.raises(
        AssetCompositionStateError,
        match="exact enabled rigid-body prim",
    ):
        complete_stage(
            state_path,
            "physics",
            output_asset=output,
            evidence_paths=[evidence],
            summary="Invalid single-body articulation.",
        )


def test_physics_accepts_distinct_sibling_rigid_body_joint_targets(
    tmp_path: Path,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    _write_jointed_physics_output(
        output,
        descendant_targets_under_one_body=False,
    )
    _refresh_physics_evidence(evidence, output)

    run = complete_stage(
        state_path,
        "physics",
        output_asset=output,
        evidence_paths=[evidence],
        summary="Valid multi-body articulation.",
    )

    assert run.stages["physics"].status == "completed"
    assert run.current_stage == "validation"


def test_verified_load_rejects_legacy_invalid_physics_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_path, _ = _create(tmp_path)
    _advance_to_physics(state_path)
    output, evidence = _stage_artifacts(state_path, "physics")
    _write_jointed_physics_output(
        output,
        descendant_targets_under_one_body=True,
    )
    _refresh_physics_evidence(evidence, output)
    original_validator = asset_state._validate_articulated_physics_output
    monkeypatch.setattr(
        asset_state,
        "_validate_articulated_physics_output",
        lambda _path: None,
    )
    complete_stage(
        state_path,
        "physics",
        output_asset=output,
        evidence_paths=[evidence],
        summary="Legacy invalid Physics output.",
    )
    monkeypatch.setattr(
        asset_state,
        "_validate_articulated_physics_output",
        original_validator,
    )

    with pytest.raises(
        AssetCompositionStateError,
        match="exact enabled rigid-body prim",
    ):
        load_verified_run(state_path)


def test_failure_and_cancellation_require_explicit_recovery(tmp_path: Path) -> None:
    state_path, _ = _create(tmp_path)
    begin_stage(state_path, "articulation")
    failed = cancel_stage(
        state_path,
        "articulation",
        reason="Operator stopped before review.",
    )
    assert failed.terminal_status == "cancelled"
    assert not validate_terminal(state_path).valid

    with pytest.raises(AssetCompositionStateError, match="Cannot begin"):
        begin_stage(state_path, "articulation")

    recovered = recover_stage(
        state_path,
        "articulation",
        reason="Operator approved same-attempt resume.",
    )
    assert recovered.terminal_status == "active"
    assert recovered.stages["articulation"].attempt_count == 1
    restarted = begin_stage(state_path, "articulation")
    assert restarted.stages["articulation"].attempt_count == 1


def test_internal_cli_returns_nonzero_for_incomplete_terminal_state(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state_path, _ = _create(tmp_path)

    assert main(["status", "--run-state", str(state_path)]) == 0
    assert main(["validate-terminal", "--run-state", str(state_path)]) == 1

    output = capsys.readouterr().out
    assert '"valid": false' in output
