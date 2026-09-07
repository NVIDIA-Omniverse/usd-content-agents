# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import stat
import tomllib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pytest
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt
from world_understanding.utils.usd.package import write_usdz_package_from_directory

import content_agent_workflows.asset_composition as asset_composition_module
import content_agent_workflows.asset_composition.catalog as asset_catalog_module
import content_agent_workflows.asset_composition.state as asset_state_module
import content_agent_workflows.texture as texture_workflow
import content_agent_workflows.texture.asset_leaf_adapter as adapter_module
import content_agent_workflows.texture.capabilities as capabilities_module
import content_agent_workflows.texture.scene_validation as scene_validation_module
import content_agent_workflows.texture.uv_authoring as uv_authoring_module
from content_agent_workflows.asset_composition import (
    ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP,
    ArtifactBinding,
    AssetCompositionStateError,
    AssetExecutionGraph,
    AssetExecutionNode,
    AssetLeafCatalog,
    AssetLeafDescriptor,
    AssetLeafProjection,
    AssetLeafReceipt,
    AssetLeafRuntimeBundle,
    AssetRunRequest,
    AssetRuntimeRequest,
    AssetSoleCoordinatorIdentity,
    AssetSourceStaging,
    asset_model_schema_digest,
    begin_leaf,
    canonical_asset_digest,
    complete_leaf,
    compose_asset_leaf_runtime_bundles,
    create_run,
    fail_leaf,
    freeze_execution_graph,
    leaf_directory,
    recover_leaf,
)
from content_agent_workflows.asset_composition.catalog_adapters import (
    CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
    shared_asset_leaf_runtime_bundle,
)
from content_agent_workflows.common.artifacts import atomic_write_json, file_sha256
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.texture import (
    TEXTURE_UV_LEAF_ID,
    TEXTURE_UV_PROJECTION_FILENAME,
    TEXTURE_UV_PUBLICATION_FILENAME,
    TextureUvLeafInvocation,
    TextureUvLeafResult,
    build_texture_uv_leaf_invocation,
    project_texture_uv_verified_operation,
    run_texture_uv_leaf,
    texture_asset_leaf_catalog,
    texture_asset_leaf_descriptors,
    texture_asset_leaf_runtime_bindings,
    texture_asset_leaf_runtime_bundle,
)


def _artifact_binding(path: Path) -> ArtifactBinding:
    resolved = path.resolve()
    return ArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


def _execution_binding(path: Path) -> ExecutionArtifactBinding:
    resolved = path.resolve()
    return ExecutionArtifactBinding(
        path=str(resolved),
        sha256=file_sha256(resolved),
        size_bytes=resolved.stat().st_size,
    )


@pytest.fixture(autouse=True)
def _resolve_texture_runtime_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep graph-state tests bound to the exact local Texture v3 runtime."""

    runtime = compose_asset_leaf_runtime_bundles((texture_asset_leaf_runtime_bundle(),))

    def resolve(candidate: AssetLeafCatalog):  # type: ignore[no-untyped-def]
        if candidate != runtime.catalog:
            raise ValueError("test catalog does not resolve to Texture runtime")
        return runtime

    monkeypatch.setattr(
        asset_state_module,
        "resolve_repository_asset_leaf_catalog",
        resolve,
    )
    asset_catalog_module.repository_asset_leaf_runtime_catalog.cache_clear()
    yield
    asset_catalog_module.repository_asset_leaf_runtime_catalog.cache_clear()


def _build_stage(
    path: Path,
    *,
    with_uvs: bool,
    uv_interpolation: str = UsdGeom.Tokens.faceVarying,
    add_ready_mesh: bool = False,
) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
    mesh.CreatePointsAttr(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 0.0, 0.0),
                Gf.Vec3f(1.0, 1.0, 0.0),
                Gf.Vec3f(0.0, 1.0, 0.0),
            ]
        )
    )
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([4]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2, 3]))
    if with_uvs:
        primvar = UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            uv_interpolation,
        )
        primvar.Set(
            Vt.Vec2fArray(
                [Gf.Vec2f(0.5, 0.5)]
                if uv_interpolation == UsdGeom.Tokens.constant
                else [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(1.0, 1.0),
                    Gf.Vec2f(0.0, 1.0),
                ]
            )
        )
    if add_ready_mesh:
        ready = UsdGeom.Mesh.Define(stage, "/World/ReadyMesh")
        ready.CreatePointsAttr(mesh.GetPointsAttr().Get())
        ready.CreateFaceVertexCountsAttr(mesh.GetFaceVertexCountsAttr().Get())
        ready.CreateFaceVertexIndicesAttr(mesh.GetFaceVertexIndicesAttr().Get())
        ready_uv = UsdGeom.PrimvarsAPI(ready.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.faceVarying,
        )
        ready_uv.Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.0, 0.0),
                    Gf.Vec2f(1.0, 0.0),
                    Gf.Vec2f(1.0, 1.0),
                    Gf.Vec2f(0.0, 1.0),
                ]
            )
        )
    assert stage.GetRootLayer().Save()


def _build_variant_composed_stage(path: Path) -> None:
    stage = Usd.Stage.CreateNew(str(path))
    root = UsdGeom.Xform.Define(stage, "/World")
    variants = root.GetPrim().GetVariantSets().AddVariantSet("model")
    for selection, scale in (("A", 1.0), ("B", 2.0)):
        variants.AddVariant(selection)
        variants.SetVariantSelection(selection)
        with variants.GetVariantEditContext():
            mesh = UsdGeom.Mesh.Define(stage, "/World/Mesh")
            mesh.CreatePointsAttr(
                Vt.Vec3fArray(
                    [
                        Gf.Vec3f(0.0, 0.0, 0.0),
                        Gf.Vec3f(scale, 0.0, 0.0),
                        Gf.Vec3f(0.0, scale, 0.0),
                    ]
                )
            )
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([0, 1, 2]))
    variants.SetVariantSelection("A")
    assert stage.GetRootLayer().Save()


def _run_native_leaf(
    tmp_path: Path,
    *,
    with_uvs: bool,
    policy: Literal["inspect", "generate_missing"],
    uv_interpolation: str = UsdGeom.Tokens.faceVarying,
) -> tuple[Path, Path, TextureUvLeafResult]:
    source = tmp_path / "source.usda"
    _build_stage(
        source,
        with_uvs=with_uvs,
        uv_interpolation=uv_interpolation,
    )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy=policy,
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)
    return attempt_root, invocation_path, result


def _create_graph_run(
    tmp_path: Path,
    *,
    unsupported_and_ready_uvs: bool = False,
    stage_builder: Callable[[Path], object] | None = None,
    selected_leaf_ids: tuple[str, ...] = (TEXTURE_UV_LEAF_ID,),
    optional_leaf_ids: tuple[str, ...] = (),
) -> tuple[Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    original = tmp_path / "original.usda"
    staged = run_dir / "source.usda"
    if stage_builder is None:
        _build_stage(
            original,
            with_uvs=unsupported_and_ready_uvs,
            uv_interpolation=(
                UsdGeom.Tokens.constant
                if unsupported_and_ready_uvs
                else UsdGeom.Tokens.faceVarying
            ),
            add_ready_mesh=unsupported_and_ready_uvs,
        )
    else:
        stage_builder(original)
    shutil.copyfile(original, staged)
    original_binding = _artifact_binding(original)
    staged_binding = _artifact_binding(staged)
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
        manifest=_artifact_binding(manifest),
        dependency_digest_set_sha256=digest_set,
    )
    catalog = texture_asset_leaf_catalog()
    coordinator = AssetSoleCoordinatorIdentity.create(
        coordinator_id="asset-coordinator:texture-adapter-test",
        invocation_id="single-prompt:texture-adapter-test",
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
    prompt = "Prepare missing UVs without invoking a provider."
    configuration_digest = canonical_asset_digest(
        {
            "repository_root": str(tmp_path.resolve()),
            "runtime": runtime.model_dump(mode="json"),
            "requires_parent_resource_release": False,
        }
    )
    request = AssetRunRequest(
        created_at="2026-08-17T00:00:00+00:00",
        schema_version="content-agents.asset-composition-request.v3",
        selected_mode="agentic",
        run_id="texture-adapter-test",
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
        requires_parent_resource_release=False,
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
    descriptors = {item.leaf_id: item for item in catalog.descriptors}
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest=coordinator.identity_digest,
        prompt_digest=str(request.prompt_digest),
        source_digest=str(request.source_digest),
        configuration_digest=str(request.configuration_digest),
        reference_digest=str(request.reference_digest),
        leaf_catalog_digest=catalog.catalog_digest,
        nodes=[
            AssetExecutionNode(
                leaf_id=leaf_id,
                depends_on=descriptors[leaf_id].required_dependencies,
                requirement=(
                    "optional" if leaf_id in optional_leaf_ids else "required"
                ),
                descriptor_digest=descriptors[leaf_id].descriptor_digest,
                terminal_output=index == len(selected_leaf_ids) - 1,
            )
            for index, leaf_id in enumerate(selected_leaf_ids)
        ],
        omitted_leaf_ids=sorted(
            item.leaf_id
            for item in catalog.descriptors
            if item.leaf_id not in selected_leaf_ids
        ),
    )
    graph_path = run_dir / "execution_graph.json"
    atomic_write_json(graph_path, graph)
    freeze_execution_graph(state_path, graph_path=graph_path)
    begin_leaf(state_path, selected_leaf_ids[0])
    return state_path, staged


def test_catalog_registers_exact_versioned_texture_leaf_chain() -> None:
    bindings = texture_asset_leaf_runtime_bindings()
    descriptors = texture_asset_leaf_descriptors()
    catalog = texture_asset_leaf_catalog()

    assert isinstance(texture_asset_leaf_runtime_bundle(), AssetLeafRuntimeBundle)
    assert texture_asset_leaf_runtime_bundle().bundle_id == "texture"
    assert tuple(binding.descriptor for binding in bindings) == descriptors
    assert len(descriptors) == 6
    assert catalog.descriptors == list(descriptors)
    assert catalog.schema_version == "content-agents.asset-leaf-catalog.v3"
    assert {item.schema_version for item in descriptors} == {
        "content-agents.asset-leaf-descriptor.v3"
    }
    descriptor = next(
        item for item in descriptors if item.leaf_id == TEXTURE_UV_LEAF_ID
    )
    assert descriptor.leaf_id == TEXTURE_UV_LEAF_ID
    assert descriptor.invocation_schema_digest == asset_model_schema_digest(
        TextureUvLeafInvocation
    )
    assert descriptor.result_schema_digest == asset_model_schema_digest(
        TextureUvLeafResult
    )
    focused = {
        item.leaf_id: item for item in descriptors if item.leaf_id != TEXTURE_UV_LEAF_ID
    }
    assert tuple(focused) == (
        "texture.apply-provided.v1",
        "texture.evidence.v1",
        "texture.prepare.v1",
        "texture.publish.v1",
        "texture.review.v1",
    )
    assert {item.invocation_schema_digest for item in focused.values()} == {
        asset_model_schema_digest(texture_workflow.TextureFocusedLeafInvocation)
    }
    assert {item.result_schema_digest for item in focused.values()} == {
        asset_model_schema_digest(texture_workflow.TextureFocusedLeafResult)
    }
    assert {item.projection_schema_digest for item in descriptors} == {
        asset_model_schema_digest(AssetLeafProjection)
    }
    assert {
        leaf_id: item.required_dependencies for leaf_id, item in focused.items()
    } == {
        "texture.apply-provided.v1": ["texture.prepare.v1"],
        "texture.evidence.v1": ["texture.apply-provided.v1"],
        "texture.prepare.v1": ["texture.uv-prepare.v1"],
        "texture.publish.v1": ["texture.review.v1"],
        "texture.review.v1": ["texture.evidence.v1"],
    }
    assert {
        item.leaf_id: item.required_artifact_categories for item in descriptors
    } == {
        "texture.apply-provided.v1": ["evidence", "saved_stage_readback"],
        "texture.evidence.v1": [
            "evidence",
            "resource_release",
            "saved_stage_readback",
        ],
        "texture.prepare.v1": [
            "evidence",
            "resource_release",
            "saved_stage_readback",
        ],
        "texture.publish.v1": ["evidence", "saved_stage_readback"],
        "texture.review.v1": ["evidence", "saved_stage_readback"],
        "texture.uv-prepare.v1": ["evidence", "saved_stage_readback"],
    }
    assert {item.leaf_id: item.projector_id for item in descriptors} == {
        "texture.apply-provided.v1": "asset.projector.texture.apply-provided.v1",
        "texture.evidence.v1": "asset.projector.texture.evidence.v1",
        "texture.prepare.v1": "asset.projector.texture.prepare.v1",
        "texture.publish.v1": "asset.projector.texture.publish.v1",
        "texture.review.v1": "asset.projector.texture.review.v1",
        "texture.uv-prepare.v1": "asset.projector.texture-uv-prepare.v1",
    }
    assert catalog.catalog_digest == (
        "ac9bb24360568d85b0bf6ef6713ae3f35d664dcd9a83686d61d47a3edc44074a"
    )
    assert {
        item.leaf_id: (
            item.invocation_schema_digest,
            item.result_schema_digest,
            item.projection_schema_digest,
            item.projector_digest,
            item.descriptor_digest,
        )
        for item in descriptors
    } == {
        "texture.apply-provided.v1": (
            "348b47e433df68418f9c345ce4a1a9d1892462b58c3224e26dff070a3b3f9979",
            "5890993f875da5391c11ae6a28d171b8be6403a21b6a758a3e979fb60133be8c",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "a1a86862add58cedfdc1a7122b7862f48bbc86da7a414285da9a3f89c37b9459",
            "b9887872a76d5cb58c9f0c3176b498f7f0ee9688320ef549b81c019de69b4518",
        ),
        "texture.evidence.v1": (
            "348b47e433df68418f9c345ce4a1a9d1892462b58c3224e26dff070a3b3f9979",
            "5890993f875da5391c11ae6a28d171b8be6403a21b6a758a3e979fb60133be8c",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "24651448baa1ff28fba077ced75a59734193d5e9dd6539639a27e745928ac555",
            "0bb1b6df337242c76fce88e91c1a819fa91387937cac2a3430f2c5e5c665dc54",
        ),
        "texture.prepare.v1": (
            "348b47e433df68418f9c345ce4a1a9d1892462b58c3224e26dff070a3b3f9979",
            "5890993f875da5391c11ae6a28d171b8be6403a21b6a758a3e979fb60133be8c",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "a607672adc90e8c00491dc425695e212e9e9b2a39e2ca824116b43a94aa48e1d",
            "5e4beb13003af63edb2fa3c2aeabc6c290ac4b12dc57113ebe5e3ffe25dbdfb4",
        ),
        "texture.publish.v1": (
            "348b47e433df68418f9c345ce4a1a9d1892462b58c3224e26dff070a3b3f9979",
            "5890993f875da5391c11ae6a28d171b8be6403a21b6a758a3e979fb60133be8c",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "8963448c412d6699486e088031c2a1a8cc68b3890ea64698349c2ea1e424494e",
            "6e9dd43624d6df5780c4eb9aa1973a7ac0bbf714a5b743c20db327ad9f2c62e3",
        ),
        "texture.review.v1": (
            "348b47e433df68418f9c345ce4a1a9d1892462b58c3224e26dff070a3b3f9979",
            "5890993f875da5391c11ae6a28d171b8be6403a21b6a758a3e979fb60133be8c",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "e71ac65ec1ca57b0d422161399908aa324bd6d238faf2af0642dd6973adc48a9",
            "16102f89aedf803304b599cb3854feacb004e1f9f95b6f627ea02367f5d12ba2",
        ),
        "texture.uv-prepare.v1": (
            "fe07c250f34b724d6291fa7fd1cc71ff28bc55b27ad18b9510df3eed5528e4b7",
            "ed97fe57b8118ad1da8c8e9d7621792d6f42a2df17523fcab05ffaa85711d56a",
            "011ac5f66435850917349d044d51c482af6fce3c3b10a1b7fdb92dd31d91a47c",
            "79db69bd442b75549aaa236affe4bd1e95d81af90d78c865ebd1e46167164673",
            "7e8ef0565b471003b761775c3653039699fa607ebf7e52665b262421679301e5",
        ),
    }
    assert not hasattr(catalog, "selected_leaf_ids")


def test_texture_agent_owns_exact_bundle_entry_point_metadata() -> None:
    repository = Path(__file__).resolve().parents[4]
    texture_metadata = tomllib.loads(
        (repository / "apps/texture_agent/pyproject.toml").read_text(encoding="utf-8")
    )
    shared_metadata = tomllib.loads(
        (
            repository / "agentic/packages/content_agent_workflows/pyproject.toml"
        ).read_text(encoding="utf-8")
    )
    assert texture_metadata["project"]["entry-points"][
        ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP
    ] == {
        "texture": (
            "content_agent_workflows.texture.asset_leaf_adapter:"
            "texture_asset_leaf_runtime_bundle"
        )
    }
    assert (
        "texture"
        not in shared_metadata["project"]["entry-points"][
            ASSET_LEAF_BUNDLE_ENTRY_POINT_GROUP
        ]
    )


def test_texture_and_shared_bundles_compose_into_one_stable_catalog_and_graph() -> None:
    texture_bundle = texture_asset_leaf_runtime_bundle()
    shared_bundle = shared_asset_leaf_runtime_bundle()
    first = compose_asset_leaf_runtime_bundles((texture_bundle, shared_bundle))
    second = compose_asset_leaf_runtime_bundles((shared_bundle, texture_bundle))

    assert first.catalog == second.catalog
    assert first.catalog.catalog_digest == second.catalog.catalog_digest
    assert first.catalog == AssetLeafCatalog.create(
        [
            binding.descriptor
            for bundle in (texture_bundle, shared_bundle)
            for binding in bundle.bindings
        ]
    )
    assert tuple(first.bindings) == tuple(sorted(first.bindings))
    assert len(first.bindings) == 9

    descriptors = {item.leaf_id: item for item in first.catalog.descriptors}
    ordered_texture_ids = tuple(
        item.leaf_id for item in texture_asset_leaf_descriptors()
    )
    nodes = [
        AssetExecutionNode(
            leaf_id=leaf_id,
            depends_on=list(descriptors[leaf_id].required_dependencies),
            requirement="required",
            descriptor_digest=descriptors[leaf_id].descriptor_digest,
            terminal_output=False,
        )
        for leaf_id in ordered_texture_ids
    ]
    nodes.append(
        AssetExecutionNode(
            leaf_id=CANONICAL_OVRTX_EVIDENCE_LEAF_ID,
            depends_on=[texture_workflow.TEXTURE_PUBLISH_LEAF_ID],
            requirement="required",
            descriptor_digest=descriptors[
                CANONICAL_OVRTX_EVIDENCE_LEAF_ID
            ].descriptor_digest,
            terminal_output=True,
        )
    )
    identity = "a" * 64
    graph = AssetExecutionGraph.create(
        sole_coordinator_identity_digest=identity,
        prompt_digest="b" * 64,
        source_digest="c" * 64,
        configuration_digest="d" * 64,
        reference_digest="e" * 64,
        leaf_catalog_digest=first.catalog.catalog_digest,
        nodes=nodes,
        omitted_leaf_ids=[
            item.leaf_id
            for item in first.catalog.descriptors
            if item.leaf_id not in {node.leaf_id for node in nodes}
        ],
    )
    assert graph.selected_leaf_ids == sorted(
        (*ordered_texture_ids, CANONICAL_OVRTX_EVIDENCE_LEAF_ID)
    )
    canonical_node = next(
        item for item in graph.nodes if item.leaf_id == CANONICAL_OVRTX_EVIDENCE_LEAF_ID
    )
    assert canonical_node.depends_on == [texture_workflow.TEXTURE_PUBLISH_LEAF_ID]


def test_texture_catalog_rejects_mixed_version_and_duplicate_registration() -> None:
    legacy = AssetLeafDescriptor.create(
        leaf_id="legacy.texture-fixture.v1",
        entrypoint="legacy.texture.fixture",
        invocation_schema_digest="a" * 64,
        result_schema_digest="b" * 64,
    )
    with pytest.raises(ValueError, match="legacy descriptor"):
        AssetLeafCatalog.create([legacy, *texture_asset_leaf_descriptors()])

    bundle = texture_asset_leaf_runtime_bundle()
    with pytest.raises(ValueError, match="duplicate repository asset leaf bundle"):
        compose_asset_leaf_runtime_bundles((bundle, bundle))
    duplicate_leaf = AssetLeafRuntimeBundle.create(
        bundle_id="texture-substitute",
        bindings=(bundle.bindings[0],),
    )
    with pytest.raises(
        ValueError,
        match="duplicate repository asset leaf registration",
    ):
        compose_asset_leaf_runtime_bundles((bundle, duplicate_leaf))


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
    name: str
    entry_points: tuple[_EntryPoint, ...]
    version: str = "fixture"

    @property
    def metadata(self) -> dict[str, str]:
        return {"Name": self.name}


def _discovery_distributions(
    *,
    texture_provider: object = texture_asset_leaf_runtime_bundle,
) -> tuple[_Distribution, ...]:
    return (
        _Distribution(
            name="content-agent-workflows",
            version="0.1.0",
            entry_points=(
                _EntryPoint(
                    name="shared",
                    provider=shared_asset_leaf_runtime_bundle,
                ),
            ),
        ),
        _Distribution(
            name="texture-agent",
            version="0.5.3",
            entry_points=(_EntryPoint(name="texture", provider=texture_provider),),
        ),
    )


def test_texture_registrar_discovery_is_complete_and_source_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distributions = _discovery_distributions()
    monkeypatch.setattr(
        asset_catalog_module.importlib_metadata,
        "distributions",
        lambda: tuple(reversed(distributions)),
    )
    catalog = asset_catalog_module.discover_repository_asset_leaf_catalog()

    assert catalog.catalog_digest == (
        "4f0ccc16bed5ddf1c6f47d5b53f25a921b4b2f1f78013a71b40471d00b9e8458"
    )
    assert [item.registrar_id for item in catalog.registrars] == [
        "content-agent-workflows",
        "texture-agent",
    ]
    assert [
        bundle.bundle_id
        for registrar in catalog.registrars
        for bundle in registrar.bundles
    ] == ["shared", "texture"]
    assert len(catalog.descriptors) == 9
    texture_identity = catalog.registrars[1].bundles[0]
    assert texture_identity.implementation.endswith(
        ".texture_asset_leaf_runtime_bundle"
    )
    assert texture_identity.source_sha256 != "0" * 64
    assert texture_identity.leaf_descriptor_digests == {
        item.leaf_id: item.descriptor_digest
        for item in texture_asset_leaf_descriptors()
    }


def test_texture_registrar_discovery_fails_closed_for_broken_or_duplicate_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        asset_catalog_module.importlib_metadata,
        "distributions",
        lambda: _discovery_distributions(
            texture_provider=ImportError("Texture registrar unavailable")
        ),
    )
    with pytest.raises(RuntimeError, match="texture-agent/texture"):
        asset_catalog_module.discover_repository_asset_leaf_catalog()

    asset_catalog_module.repository_asset_leaf_runtime_catalog.cache_clear()
    distributions = _discovery_distributions()
    monkeypatch.setattr(
        asset_catalog_module.importlib_metadata,
        "distributions",
        lambda: (*distributions, distributions[1]),
    )
    with pytest.raises(RuntimeError, match="distribution is duplicated: texture-agent"):
        asset_catalog_module.discover_repository_asset_leaf_catalog()


def test_uv_runtime_projector_preserves_exact_context_and_native_disposition(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="inspect",
    )
    result_path = attempt_root / "texture_uv_leaf_result.json"
    binding = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }[TEXTURE_UV_LEAF_ID]
    before = _tree_identities(attempt_root)

    projection = binding.project(
        TextureUvLeafInvocation.model_validate_json(invocation_path.read_bytes()),
        result,
        invocation_artifact=_artifact_binding(invocation_path),
        result_artifact=_artifact_binding(result_path),
    )

    assert projection.payload.native_disposition == "not_evaluated"
    assert projection.payload.native_status == "not_evaluated"
    assert projection.payload.native_terminal_receipt == _artifact_binding(result_path)
    assert projection.payload.evidence == tuple(
        _artifact_binding(Path(item.path)) for item in result.evidence
    )
    assert projection.payload.saved_stage_readbacks == tuple(
        _artifact_binding(Path(item.path)) for item in result.saved_stage_readbacks
    )
    assert projection.context.invocation_artifact == _artifact_binding(invocation_path)
    assert projection.context.result_artifact == _artifact_binding(result_path)
    assert _tree_identities(attempt_root) == before


def test_uv_runtime_projector_reverifies_exact_package_predecessor(
    tmp_path: Path,
) -> None:
    package_source = tmp_path / "articulation-package"
    package_source.mkdir()
    root_layer = package_source / "rigged.usda"
    _build_stage(root_layer, with_uvs=False)
    package_stage = Usd.Stage.Open(str(root_layer))
    assert package_stage is not None
    UsdGeom.SetStageUpAxis(package_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(package_stage, 2.5)
    package_stage.GetRootLayer().documentation = "sealed Joint publication"
    package_stage.GetRootLayer().framePrecision = 4
    assert package_stage.GetRootLayer().Save()
    source = tmp_path / "rigged.usdz"
    write_usdz_package_from_directory(
        package_source,
        Path("rigged.usda"),
        source,
    )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)

    publication = project_texture_uv_verified_operation(
        attempt_root=attempt_root,
        invocation_path=invocation_path,
        native_result_path=attempt_root / "texture_uv_leaf_result.json",
    )

    assert publication.native_disposition == "passed"
    assert publication.output == result.output
    assert len(publication.output_dependencies) == 1
    predecessor = publication.output_dependencies[0]
    assert predecessor.path == invocation.source.path
    assert predecessor.sha256 == invocation.source.sha256
    assert predecessor.size_bytes == invocation.source.size_bytes
    projection = json.loads(
        (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).read_text(encoding="utf-8")
    )
    assert projection["native_disposition"] == "passed"
    reopened = Usd.Stage.Open(publication.output.path, load=Usd.Stage.LoadAll)
    assert reopened is not None
    assert UsdGeom.GetStageUpAxis(reopened) == UsdGeom.Tokens.z
    assert UsdGeom.GetStageMetersPerUnit(reopened) == 2.5
    assert reopened.GetRootLayer().documentation == "sealed Joint publication"
    assert reopened.GetRootLayer().framePrecision == 4


def test_uv_runtime_projector_reverifies_noop_self_contained_package(
    tmp_path: Path,
) -> None:
    package_source = tmp_path / "articulation-package"
    package_source.mkdir()
    texture_dir = package_source / "textures"
    texture_dir.mkdir()
    (texture_dir / "albedo.png").write_bytes(b"sealed-texture")
    root_layer = package_source / "rigged.usda"
    _build_stage(root_layer, with_uvs=True)
    package_stage = Usd.Stage.Open(str(root_layer))
    assert package_stage is not None
    material = UsdShade.Material.Define(package_stage, "/World/Looks/Material")
    texture = UsdShade.Shader.Define(
        package_stage,
        "/World/Looks/Material/Texture",
    )
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("textures/albedo.png")
    )
    UsdShade.MaterialBindingAPI.Apply(package_stage.GetPrimAtPath("/World/Mesh")).Bind(
        material
    )
    assert package_stage.GetRootLayer().Save()
    source = tmp_path / "rigged.usdz"
    write_usdz_package_from_directory(
        package_source,
        Path("rigged.usda"),
        source,
    )
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)
    assert result.native_disposition == "passed"
    assert result.output == invocation.source
    assert result.output_dependencies == ()

    publication = project_texture_uv_verified_operation(
        attempt_root=attempt_root,
        invocation_path=invocation_path,
        native_result_path=attempt_root / "texture_uv_leaf_result.json",
    )

    assert publication.native_disposition == "passed"
    assert publication.output == invocation.source
    assert publication.output_dependencies == ()


def test_uv_not_evaluated_cannot_complete_with_selected_dependent(
    tmp_path: Path,
) -> None:
    state_path, staged = _create_graph_run(
        tmp_path,
        selected_leaf_ids=(
            TEXTURE_UV_LEAF_ID,
            texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        ),
        optional_leaf_ids=(TEXTURE_UV_LEAF_ID,),
    )
    attempt_root = leaf_directory(state_path, TEXTURE_UV_LEAF_ID)
    invocation = build_texture_uv_leaf_invocation(
        staged,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="inspect",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)
    result_path = attempt_root / "texture_uv_leaf_result.json"
    assert result.native_disposition == "not_evaluated"

    with pytest.raises(
        AssetCompositionStateError,
        match=(
            "not_evaluated leaf texture.uv-prepare.v1 is required by another "
            "selected leaf"
        ),
    ):
        complete_leaf(
            state_path,
            TEXTURE_UV_LEAF_ID,
            invocation_path=invocation_path,
            result_path=result_path,
        )

    resumed = asset_state_module.load_verified_run(state_path)
    assert resumed.leaf_states[TEXTURE_UV_LEAF_ID].status == "running"
    assert resumed.leaf_states[TEXTURE_UV_LEAF_ID].receipt is None
    assert resumed.leaf_states[texture_workflow.TEXTURE_PREPARE_LEAF_ID].status == (
        "pending"
    )


@pytest.mark.parametrize("substitution", ["path", "sha256", "size_bytes"])
def test_uv_runtime_projector_rejects_context_substitution(
    tmp_path: Path,
    substitution: str,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    selected = _artifact_binding(invocation_path)
    if substitution == "path":
        substitute_path = attempt_root / "substituted_invocation.json"
        shutil.copyfile(invocation_path, substitute_path)
        substituted = _artifact_binding(substitute_path)
    elif substitution == "sha256":
        substituted = selected.model_copy(update={"sha256": "f" * 64})
    else:
        substituted = selected.model_copy(
            update={"size_bytes": selected.size_bytes + 1}
        )
    binding = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }[TEXTURE_UV_LEAF_ID]

    with pytest.raises(
        ValueError,
        match="selected artifacts|frozen projection context|exact invocation",
    ):
        binding.project(
            TextureUvLeafInvocation.model_validate_json(invocation_path.read_bytes()),
            result,
            invocation_artifact=substituted,
            result_artifact=_artifact_binding(
                attempt_root / "texture_uv_leaf_result.json"
            ),
        )


def test_projector_preserves_nonpassing_native_disposition(tmp_path: Path) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="inspect",
    )
    assert result.native_disposition == "not_evaluated"

    publication = project_texture_uv_verified_operation(
        attempt_root=attempt_root,
        invocation_path=invocation_path,
        native_result_path=attempt_root / "texture_uv_leaf_result.json",
    )

    assert publication.native_disposition == "not_evaluated"
    assert publication.graph_terminal_action == "complete_leaf"
    assert publication.output == result.output
    assert publication.graph_evidence_paths == tuple(
        item.path for item in result.evidence
    )
    assert publication.graph_saved_stage_readback_paths == tuple(
        item.path for item in result.saved_stage_readbacks
    )
    projection = json.loads(
        (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).read_text(encoding="utf-8")
    )
    assert projection["native_disposition"] == "not_evaluated"
    assert projection["projector_native_leaf_invoked"] is False
    assert projection["projector_graph_state_mutated"] is False


def test_projector_rejects_native_disposition_upgrade(tmp_path: Path) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="inspect",
    )
    result_path = attempt_root / "texture_uv_leaf_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["native_disposition"] = "passed"
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="refuses to upgrade or relabel"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_rejects_pending_detail_tampered_to_refusal(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="inspect",
    )
    assert result.detail == (
        "UV inspection completed but the selected stage requires authoring."
    )
    result_path = attempt_root / "texture_uv_leaf_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["detail"] = (
        "UV authoring refused this saved stage fail closed; saved-stage readback "
        "reasons: /World/Mesh: primvars:st is missing or empty."
    )
    atomic_write_json(result_path, payload)

    with pytest.raises(
        ValueError,
        match="native result detail differs from recomputed saved-stage facts",
    ):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_rejects_refusal_detail_tampered_to_pending(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="generate_missing",
        uv_interpolation=UsdGeom.Tokens.constant,
    )
    assert result.detail == (
        "UV authoring refused this saved stage fail closed; saved-stage readback "
        "reasons: /World/Mesh: unsupported UV interpolation constant."
    )
    result_path = attempt_root / "texture_uv_leaf_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["detail"] = (
        "UV inspection completed but the selected stage requires authoring."
    )
    payload["error"] = payload["detail"]
    atomic_write_json(result_path, payload)

    with pytest.raises(
        ValueError,
        match="native result detail differs from recomputed saved-stage facts",
    ):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_recomputes_stage_and_rejects_self_consistent_pass_tamper(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="inspect",
    )
    assert result.native_disposition == "not_evaluated"
    readback_path = Path(result.saved_stage_readbacks[0].path)
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    readback["all_uv_ready"] = True
    readback_without_identity = dict(readback)
    readback_without_identity.pop("readback_identity_sha256")
    readback["readback_identity_sha256"] = canonical_asset_digest(
        readback_without_identity
    )
    atomic_write_json(readback_path, readback)

    evidence_path = Path(result.evidence[0].path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["saved_stage_readback_identity_sha256"] = readback[
        "readback_identity_sha256"
    ]
    atomic_write_json(evidence_path, evidence)

    result_path = attempt_root / "texture_uv_leaf_result.json"
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload["native_disposition"] = "passed"
    result_payload["evidence"] = [
        ExecutionArtifactBinding(
            path=str(evidence_path),
            sha256=file_sha256(evidence_path),
            size_bytes=evidence_path.stat().st_size,
        ).model_dump(mode="json")
    ]
    result_payload["saved_stage_readbacks"] = [
        ExecutionArtifactBinding(
            path=str(readback_path),
            sha256=file_sha256(readback_path),
            size_bytes=readback_path.stat().st_size,
        ).model_dump(mode="json")
    ]
    atomic_write_json(result_path, result_payload)

    with pytest.raises(ValueError, match="differs from recomputed stage facts"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_rejects_self_consistent_non_uv_output_delta(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="generate_missing",
    )
    output_path = Path(result.output.path)
    stage = Usd.Stage.Open(str(output_path))
    stage.SetEditTarget(stage.GetRootLayer())
    UsdGeom.Xform.Define(stage, "/Injected")
    UsdGeom.Mesh(stage.GetPrimAtPath("/World/Mesh")).GetPointsAttr().Set(
        Vt.Vec3fArray(
            [
                Gf.Vec3f(0.0, 0.0, 0.0),
                Gf.Vec3f(4.0, 0.0, 0.0),
                Gf.Vec3f(4.0, 4.0, 0.0),
                Gf.Vec3f(0.0, 4.0, 0.0),
            ]
        )
    )
    assert stage.GetRootLayer().Save()
    output_binding = ExecutionArtifactBinding(
        path=str(output_path),
        sha256=file_sha256(output_path),
        size_bytes=output_path.stat().st_size,
    )

    readback_path = Path(result.saved_stage_readbacks[0].path)
    readback = json.loads(readback_path.read_text(encoding="utf-8"))
    readback["saved_stage"] = output_binding.model_dump(mode="json")
    readback_without_identity = dict(readback)
    readback_without_identity.pop("readback_identity_sha256")
    readback["readback_identity_sha256"] = canonical_asset_digest(
        readback_without_identity
    )
    atomic_write_json(readback_path, readback)
    readback_binding = ExecutionArtifactBinding(
        path=str(readback_path),
        sha256=file_sha256(readback_path),
        size_bytes=readback_path.stat().st_size,
    )

    evidence_path = Path(result.evidence[0].path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["saved_stage"] = output_binding.model_dump(mode="json")
    evidence["saved_stage_readback_identity_sha256"] = readback[
        "readback_identity_sha256"
    ]
    atomic_write_json(evidence_path, evidence)
    evidence_binding = ExecutionArtifactBinding(
        path=str(evidence_path),
        sha256=file_sha256(evidence_path),
        size_bytes=evidence_path.stat().st_size,
    )

    result_path = attempt_root / "texture_uv_leaf_result.json"
    result_payload = json.loads(result_path.read_text(encoding="utf-8"))
    result_payload["output"] = output_binding.model_dump(mode="json")
    result_payload["evidence"] = [evidence_binding.model_dump(mode="json")]
    result_payload["saved_stage_readbacks"] = [readback_binding.model_dump(mode="json")]
    atomic_write_json(result_path, result_payload)

    with pytest.raises(ValueError, match="non-UV or non-deterministic delta"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_missing_reopened_primvar_uses_explicit_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="generate_missing",
    )
    real_open = uv_authoring_module._stage_from_saved_snapshot

    def remove_reopened_st(document: bytes, **kwargs: object) -> Usd.Stage:
        stage = real_open(document, **kwargs)  # type: ignore[arg-type]
        stage.SetEditTarget(stage.GetRootLayer())
        prim = stage.GetPrimAtPath("/World/Mesh")
        prim.RemoveProperty("primvars:st")
        prim.RemoveProperty("primvars:st:indices")
        assert not UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
        return stage

    monkeypatch.setattr(
        uv_authoring_module,
        "_stage_from_saved_snapshot",
        remove_reopened_st,
    )

    with pytest.raises(ValueError) as failure:
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )

    assert str(failure.value) == (
        "Texture UV authored overlay does not contain the exact expected "
        "bounded box fallback values: /World/Mesh"
    )
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_refuses_publication_collision_without_partial_output(
    tmp_path: Path,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    publication_path = attempt_root / TEXTURE_UV_PUBLICATION_FILENAME
    publication_path.write_text("reserved", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refuses to replace"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )
    assert publication_path.read_text(encoding="utf-8") == "reserved"
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()


def test_projector_rejects_native_result_outside_attempt_root(tmp_path: Path) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    external_result = tmp_path / "external-result.json"
    shutil.copyfile(attempt_root / "texture_uv_leaf_result.json", external_result)

    with pytest.raises(ValueError, match="outside the workflow run"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=external_result,
        )


def test_projector_rejects_evidence_escape_and_readback_symlink(
    tmp_path: Path,
) -> None:
    escape_root = tmp_path / "escape"
    escape_root.mkdir()
    attempt_root, invocation_path, result = _run_native_leaf(
        escape_root,
        with_uvs=True,
        policy="inspect",
    )
    external_evidence = tmp_path / "external-evidence.json"
    shutil.copyfile(Path(result.evidence[0].path), external_evidence)
    result_path = attempt_root / "texture_uv_leaf_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["evidence"] = [
        ExecutionArtifactBinding(
            path=str(external_evidence.resolve()),
            sha256=file_sha256(external_evidence),
            size_bytes=external_evidence.stat().st_size,
        ).model_dump(mode="json")
    ]
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the workflow run"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
        )

    symlink_root = tmp_path / "symlink"
    symlink_root.mkdir()
    attempt_root, invocation_path, result = _run_native_leaf(
        symlink_root,
        with_uvs=True,
        policy="inspect",
    )
    readback_path = Path(result.saved_stage_readbacks[0].path)
    external_readback = tmp_path / "external-readback.json"
    shutil.copyfile(readback_path, external_readback)
    readback_path.unlink()
    readback_path.symlink_to(external_readback)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )


@pytest.mark.parametrize("drift_target", ["source", "output"])
def test_projector_rejects_source_or_output_dependency_drift(
    tmp_path: Path,
    drift_target: str,
) -> None:
    attempt_root, invocation_path, result = _run_native_leaf(
        tmp_path,
        with_uvs=False,
        policy="generate_missing",
    )
    assert result.output != result.source
    target = Path(
        result.source.path if drift_target == "source" else result.output.path
    )
    target.write_bytes(target.read_bytes() + b"\n# drift\n")

    with pytest.raises(ValueError, match="identity changed"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )


@pytest.mark.parametrize("swap_target", ["source", "dependency"])
def test_projector_consumes_frozen_source_across_path_swap_and_restore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    swap_target: str,
) -> None:
    dependency = tmp_path / "dependency.usda"
    _build_stage(dependency, with_uvs=False)
    source = tmp_path / "source.usda"
    root_layer = Sdf.Layer.CreateNew(str(source))
    root_layer.subLayerPaths = [dependency.name]
    assert root_layer.Save()
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)
    source = Path(result.source.path)
    target = source if swap_target == "source" else dependency
    competitor = tmp_path / f"competitor-{swap_target}.usda"
    _build_stage(competitor, with_uvs=True)
    competitor_stage = Usd.Stage.Open(str(competitor))
    UsdGeom.Xform.Define(competitor_stage, "/Injected")
    assert competitor_stage.GetRootLayer().Save()
    held_original = tmp_path / f"held-{swap_target}.usda"
    original_open_stage = uv_authoring_module._FrozenUsdSnapshot.open_stage
    swapped = False

    def swap_during_open(
        snapshot: uv_authoring_module._FrozenUsdSnapshot,
    ) -> Usd.Stage:
        nonlocal swapped
        if swapped:
            return original_open_stage(snapshot)
        swapped = True
        target.rename(held_original)
        competitor.rename(target)
        try:
            return original_open_stage(snapshot)
        finally:
            target.rename(competitor)
            held_original.rename(target)

    monkeypatch.setattr(
        uv_authoring_module._FrozenUsdSnapshot,
        "open_stage",
        swap_during_open,
    )

    publication = project_texture_uv_verified_operation(
        attempt_root=attempt_root,
        invocation_path=invocation_path,
        native_result_path=attempt_root / "texture_uv_leaf_result.json",
    )

    assert swapped is True
    assert publication.native_disposition == "passed"


def test_projector_rejects_binary_private_snapshot_byte_restoration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usdc"
    _build_stage(source, with_uvs=False)
    attempt_root = tmp_path / "attempt"
    attempt_root.mkdir()
    invocation = build_texture_uv_leaf_invocation(
        source,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    run_texture_uv_leaf(invocation_path)

    competitor = tmp_path / "competitor.usdc"
    _build_stage(competitor, with_uvs=True)
    competitor_stage = Usd.Stage.Open(str(competitor))
    UsdGeom.Xform.Define(competitor_stage, "/Injected")
    assert competitor_stage.GetRootLayer().Save()
    replacement_bytes = competitor.read_bytes()
    real_open = uv_authoring_module.Sdf.Layer.OpenAsAnonymous
    altered = False

    def alter_snapshot(path: str) -> Sdf.Layer:
        nonlocal altered
        snapshot_path = Path(path)
        if altered or snapshot_path.suffix != ".usdc":
            return real_open(path)
        altered = True
        original_bytes = snapshot_path.read_bytes()
        snapshot_path.write_bytes(replacement_bytes)
        try:
            return real_open(path)
        finally:
            snapshot_path.write_bytes(original_bytes)

    monkeypatch.setattr(
        uv_authoring_module.Sdf.Layer,
        "OpenAsAnonymous",
        alter_snapshot,
    )

    with pytest.raises(ValueError, match="namespace or bytes changed"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )

    assert altered is True
    assert not (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).exists()
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_race_cleanup_preserves_competing_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    real_write = adapter_module._HeldDirectory.write_json
    publication_path = attempt_root / TEXTURE_UV_PUBLICATION_FILENAME

    def race_publication(
        held: object,
        name: str,
        payload: object,
    ) -> tuple[int, int]:
        if name == publication_path.name:
            publication_path.write_text("competing publication", encoding="utf-8")
            raise FileExistsError("simulated concurrent publication")
        return real_write(held, name, payload)  # type: ignore[arg-type]

    monkeypatch.setattr(
        adapter_module._HeldDirectory,
        "write_json",
        race_publication,
    )
    with pytest.raises(FileExistsError, match="simulated concurrent"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )
    assert publication_path.read_text(encoding="utf-8") == "competing publication"
    assert (attempt_root / TEXTURE_UV_PROJECTION_FILENAME).is_file()


def test_projector_failure_never_unlinks_replacement_at_owned_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    projection_path = attempt_root / TEXTURE_UV_PROJECTION_FILENAME
    competitor_path = tmp_path / "competing-projection.json"
    competitor_path.write_text("competing projection", encoding="utf-8")
    real_binding = adapter_module._HeldDirectory.binding
    replaced = False

    def replace_before_binding(
        held: adapter_module._HeldDirectory,
        name: str,
        *,
        identity: tuple[int, int] | None = None,
    ) -> ExecutionArtifactBinding:
        nonlocal replaced
        if name == TEXTURE_UV_PROJECTION_FILENAME and not replaced:
            competitor_path.replace(projection_path)
            replaced = True
        return real_binding(held, name, identity=identity)

    monkeypatch.setattr(
        adapter_module._HeldDirectory,
        "binding",
        replace_before_binding,
    )
    with pytest.raises(ValueError, match="artifact identity changed"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )

    assert projection_path.read_text(encoding="utf-8") == "competing projection"
    assert not (attempt_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_rejects_attempt_root_replacement_without_path_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )
    moved_root = tmp_path / "moved-attempt"
    real_write = adapter_module._HeldDirectory.write_json
    replaced = False

    def replace_root_after_projection(
        held: object,
        name: str,
        payload: object,
    ) -> tuple[int, int]:
        nonlocal replaced
        identity = real_write(held, name, payload)  # type: ignore[arg-type]
        if name == TEXTURE_UV_PROJECTION_FILENAME and not replaced:
            attempt_root.rename(moved_root)
            attempt_root.mkdir()
            (attempt_root / "competitor.txt").write_text("preserve", encoding="utf-8")
            replaced = True
        return identity

    monkeypatch.setattr(
        adapter_module._HeldDirectory,
        "write_json",
        replace_root_after_projection,
    )
    with pytest.raises(ValueError, match="attempt root identity changed"):
        project_texture_uv_verified_operation(
            attempt_root=attempt_root,
            invocation_path=invocation_path,
            native_result_path=attempt_root / "texture_uv_leaf_result.json",
        )

    assert (attempt_root / "competitor.txt").read_text(encoding="utf-8") == "preserve"
    assert tuple(attempt_root.iterdir()) == (attempt_root / "competitor.txt",)
    assert (moved_root / TEXTURE_UV_PROJECTION_FILENAME).is_file()
    assert not (moved_root / TEXTURE_UV_PUBLICATION_FILENAME).exists()


def test_projector_invokes_no_leaf_graph_or_provider_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt_root, invocation_path, _result = _run_native_leaf(
        tmp_path,
        with_uvs=True,
        policy="inspect",
    )

    def tripwire(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("projector crossed its non-executing boundary")

    for module, names in (
        (uv_authoring_module, ("run_texture_uv_leaf",)),
        (
            asset_composition_module,
            ("begin_leaf", "complete_leaf", "fail_leaf", "cancel_leaf"),
        ),
        (
            capabilities_module,
            (
                "invoke_texture_generator",
                "request_texture_provider_proposal",
                "request_texture_critique",
            ),
        ),
        (
            scene_validation_module,
            ("LiveUsdCliTextureValidator", "VlmTextureVisualAssessor"),
        ),
    ):
        for name in names:
            monkeypatch.setattr(module, name, tripwire)

    publication = project_texture_uv_verified_operation(
        attempt_root=attempt_root,
        invocation_path=invocation_path,
        native_result_path=attempt_root / "texture_uv_leaf_result.json",
    )
    assert publication.native_disposition == "passed"


def test_projected_paths_complete_exact_generic_asset_receipt(tmp_path: Path) -> None:
    state_path, staged = _create_graph_run(tmp_path)
    attempt_root = leaf_directory(state_path, TEXTURE_UV_LEAF_ID)
    invocation = build_texture_uv_leaf_invocation(
        staged,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = run_texture_uv_leaf(invocation_path)
    assert result.native_disposition == "passed"
    result_path = attempt_root / "texture_uv_leaf_result.json"

    completed = complete_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        invocation_path=invocation_path,
        result_path=result_path,
        actor="test-outer-reasoner",
    )
    state = completed.leaf_states[TEXTURE_UV_LEAF_ID]
    assert state.receipt is not None
    receipt = AssetLeafReceipt.model_validate_json(
        Path(state.receipt.path).read_text(encoding="utf-8")
    )
    descriptor = {item.leaf_id: item for item in texture_asset_leaf_descriptors()}[
        TEXTURE_UV_LEAF_ID
    ]
    assert receipt.schema_version == "content-agent-workflows.asset-leaf-receipt.v2"
    assert receipt.native_disposition == "passed"
    assert receipt.native_status == "passed"
    assert receipt.descriptor_digest == descriptor.descriptor_digest
    assert receipt.projector_digest == descriptor.projector_digest
    assert receipt.invocation.path == str(invocation_path)
    assert receipt.result is not None
    assert receipt.result.path == str(result_path)
    assert receipt.projection is not None
    projection = AssetLeafProjection.model_validate_json(
        Path(receipt.projection.path).read_bytes()
    )
    assert projection.context.invocation_artifact == _artifact_binding(invocation_path)
    assert projection.context.result_artifact == _artifact_binding(result_path)
    assert tuple(item.path for item in receipt.evidence) == tuple(
        item.path for item in result.evidence
    )
    assert tuple(item.path for item in receipt.saved_stage_readbacks) == tuple(
        item.path for item in result.saved_stage_readbacks
    )
    assert receipt.native_terminal_receipt == _artifact_binding(result_path)
    assert receipt.operation_indexes == []
    assert receipt.evidence_indexes == []

    assert completed.coordinator.next_action == "finalize_receipts"
    resumed = asset_state_module.load_verified_run(state_path)
    resumed_state = resumed.leaf_states[TEXTURE_UV_LEAF_ID]
    assert resumed.execution_graph == completed.execution_graph
    assert resumed_state.receipt == state.receipt
    assert resumed_state.superseded_receipts == []


def test_uv_graph_v2_failure_recovery_supersedes_exact_native_chain(
    tmp_path: Path,
) -> None:
    state_path, staged = _create_graph_run(
        tmp_path,
        unsupported_and_ready_uvs=True,
    )
    initial = asset_state_module.load_verified_run(state_path)
    frozen_graph_binding = initial.execution_graph
    assert frozen_graph_binding is not None
    frozen_graph_bytes = Path(frozen_graph_binding.path).read_bytes()
    frozen_graph = AssetExecutionGraph.model_validate_json(frozen_graph_bytes)
    assert frozen_graph.schema_version == "content-agents.asset-execution-graph.v2"

    first_attempt = leaf_directory(state_path, TEXTURE_UV_LEAF_ID)
    first_invocation = build_texture_uv_leaf_invocation(
        staged,
        output_dir=first_attempt,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    first_invocation_path = first_attempt / "invocation.json"
    atomic_write_json(first_invocation_path, first_invocation)
    first_result = run_texture_uv_leaf(first_invocation_path)
    first_result_path = first_attempt / "texture_uv_leaf_result.json"
    assert first_result.native_disposition == "failed"
    assert first_result.error == first_result.detail
    assert first_result.error == (
        "UV authoring refused this saved stage fail closed; saved-stage readback "
        "reasons: /World/Mesh: unsupported UV interpolation constant."
    )
    assert first_result.output == first_result.source
    assert not (first_attempt / "prepared_texture_uvs.usda").exists()

    failed = fail_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        reason=first_result.error,
        invocation_path=first_invocation_path,
        result_path=first_result_path,
    )
    failed_state = failed.leaf_states[TEXTURE_UV_LEAF_ID]
    failed_receipt_binding = failed_state.receipt
    assert failed_receipt_binding is not None
    failed_receipt = AssetLeafReceipt.model_validate_json(
        Path(failed_receipt_binding.path).read_bytes()
    )
    assert failed_receipt.schema_version == (
        "content-agent-workflows.asset-leaf-receipt.v2"
    )
    assert failed_receipt.native_disposition == "failed"
    assert failed_receipt.native_status == "failed"
    assert failed_receipt.error == first_result.error
    assert failed_receipt.supersedes is None
    assert failed_receipt.projection is not None
    failed_projection = AssetLeafProjection.model_validate_json(
        Path(failed_receipt.projection.path).read_bytes()
    )
    assert failed_projection.payload.native_disposition == "failed"
    assert failed_projection.payload.error == first_result.error
    assert failed_projection.context.invocation_artifact == _artifact_binding(
        first_invocation_path
    )
    assert failed_projection.context.result_artifact == _artifact_binding(
        first_result_path
    )
    first_chain = {
        binding.path: Path(binding.path).read_bytes()
        for binding in (
            failed_receipt_binding,
            failed_receipt.invocation,
            failed_receipt.result,
            failed_receipt.projection,
            failed_receipt.native_terminal_receipt,
            *failed_receipt.evidence,
            *failed_receipt.saved_stage_readbacks,
        )
        if binding is not None
    }

    recovered = recover_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        reason="Retry the same frozen UV leaf against another selected ready scope.",
    )
    recovered_state = recovered.leaf_states[TEXTURE_UV_LEAF_ID]
    assert recovered.execution_graph == frozen_graph_binding
    assert recovered_state.status == "ready"
    assert recovered_state.receipt is None
    assert recovered_state.superseded_receipts == [failed_receipt_binding]
    assert Path(frozen_graph_binding.path).read_bytes() == frozen_graph_bytes
    assert {path: Path(path).read_bytes() for path in first_chain} == first_chain

    begin_leaf(state_path, TEXTURE_UV_LEAF_ID)
    second_attempt = leaf_directory(state_path, TEXTURE_UV_LEAF_ID)
    assert second_attempt != first_attempt
    second_invocation = build_texture_uv_leaf_invocation(
        staged,
        output_dir=second_attempt,
        target_prim_paths=("/World/ReadyMesh",),
        policy="generate_missing",
    )
    second_invocation_path = second_attempt / "invocation.json"
    atomic_write_json(second_invocation_path, second_invocation)
    second_result = run_texture_uv_leaf(second_invocation_path)
    second_result_path = second_attempt / "texture_uv_leaf_result.json"
    assert second_result.native_disposition == "passed"
    assert second_result.error is None
    completed = complete_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        invocation_path=second_invocation_path,
        result_path=second_result_path,
    )
    completed_state = completed.leaf_states[TEXTURE_UV_LEAF_ID]
    assert completed_state.receipt is not None
    assert completed_state.superseded_receipts == [failed_receipt_binding]
    completed_receipt = AssetLeafReceipt.model_validate_json(
        Path(completed_state.receipt.path).read_bytes()
    )
    assert completed_receipt.native_disposition == "passed"
    assert completed_receipt.error is None
    assert completed_receipt.supersedes == failed_receipt_binding
    assert completed_receipt.invocation == _artifact_binding(second_invocation_path)
    assert completed_receipt.result == _artifact_binding(second_result_path)
    assert completed_receipt.projection is not None
    completed_projection = AssetLeafProjection.model_validate_json(
        Path(completed_receipt.projection.path).read_bytes()
    )
    assert completed_projection.payload.native_disposition == "passed"
    assert completed_projection.payload.error is None
    assert completed_projection.context.invocation_artifact == (
        completed_receipt.invocation
    )
    assert completed_projection.context.result_artifact == completed_receipt.result
    assert completed.coordinator.next_action == "finalize_receipts"
    assert Path(frozen_graph_binding.path).read_bytes() == frozen_graph_bytes
    assert {path: Path(path).read_bytes() for path in first_chain} == first_chain

    resumed = asset_state_module.load_verified_run(state_path)
    resumed_state = resumed.leaf_states[TEXTURE_UV_LEAF_ID]
    assert resumed.execution_graph == frozen_graph_binding
    assert resumed_state.receipt == completed_state.receipt
    assert resumed_state.superseded_receipts == [failed_receipt_binding]


def test_uv_deterministic_preflight_rejection_seals_graph_v2_failure(
    tmp_path: Path,
) -> None:
    state_path, staged = _create_graph_run(
        tmp_path,
        stage_builder=_build_variant_composed_stage,
    )
    attempt_root = leaf_directory(state_path, TEXTURE_UV_LEAF_ID)
    invocation = build_texture_uv_leaf_invocation(
        staged,
        output_dir=attempt_root,
        target_prim_paths=("/World/Mesh",),
        policy="generate_missing",
    )
    invocation_path = attempt_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = run_texture_uv_leaf(invocation_path)
    result_path = attempt_root / "texture_uv_leaf_result.json"
    assert result.native_disposition == "failed"
    assert result.error is not None
    assert "variant-composed target meshes" in result.error

    failed = fail_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        reason=result.error,
        invocation_path=invocation_path,
        result_path=result_path,
    )
    failed_state = failed.leaf_states[TEXTURE_UV_LEAF_ID]
    assert failed_state.status == "failed"
    assert failed_state.receipt is not None
    receipt = AssetLeafReceipt.model_validate_json(
        Path(failed_state.receipt.path).read_bytes()
    )
    assert receipt.native_disposition == "failed"
    assert receipt.error == result.error
    assert receipt.invocation == _artifact_binding(invocation_path)
    assert receipt.result == _artifact_binding(result_path)

    recovered = recover_leaf(
        state_path,
        TEXTURE_UV_LEAF_ID,
        reason="Select a non-variant-composed scope in a fresh attempt.",
    )
    assert recovered.leaf_states[TEXTURE_UV_LEAF_ID].status == "ready"


def _focused_source(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "World"
)

def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Paint" {}
        def Material "Untouched" {}
    }
    def Mesh "Panel" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        rel material:binding = </World/Looks/Paint>
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }
    def Xform "Untouched" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        custom string releaseInvariant = "preserve topology physics and material"
    }
}
""",
        encoding="utf-8",
    )
    return path


class _FocusedInspector:
    def inspect(self, *, request: Any, plan: Any, output_dir: Path) -> Any:
        before = output_dir / "before.png"
        before.write_bytes(b"canonical-ovrtx-before")
        facts = output_dir / "inspection.json"
        facts.write_text('{"provider_invoked":false}\n', encoding="utf-8")
        return texture_workflow.TextureInspectionResult(
            source=_execution_binding(Path(request.source_asset)),
            proposal_plan_digest=texture_workflow.texture_plan_digest(plan),
            units=tuple(
                texture_workflow.TextureInspectionUnit(
                    unit_id=unit.unit_id,
                    material_prim_paths=unit.material_prim_paths,
                    member_prim_paths=unit.member_prim_paths,
                    member_subset_paths=unit.member_subset_paths,
                    uv_status="ready",
                    uv_facts={"interpolation": "vertex", "finite_values": True},
                    proposed_generator_inputs=texture_workflow.TextureGeneratorInputs(
                        backend="not_requested",
                        prompt="provider-free inspection",
                    ),
                )
                for unit in plan.selected_units
            ),
            before_render_artifacts=(_execution_binding(before),),
            inspection_artifacts=(_execution_binding(facts),),
            reference_artifacts=request.reference_artifacts,
            capability_constraints=("surface texturing only",),
            renderer_metadata={"renderer": "canonical-test-ovrtx"},
            tool_metadata={"provider": "not_requested"},
        )


class _FocusedCollector:
    def collect_candidate_evidence(self, **kwargs: Any) -> Any:
        output_dir = Path(kwargs["output_dir"])
        static = output_dir / "scope_invariants.json"
        static.write_text('{"passed":true}\n', encoding="utf-8")
        units: list[texture_workflow.TextureUnitRenderEvidence] = []
        for unit_id in kwargs["unit_ids"]:
            source = output_dir / f"{unit_id}_source.png"
            candidate = output_dir / f"{unit_id}_candidate.png"
            source.write_bytes(f"source:{unit_id}".encode())
            candidate.write_bytes(f"candidate:{unit_id}".encode())
            units.append(
                texture_workflow.TextureUnitRenderEvidence(
                    unit_id=unit_id,
                    source_images=(_execution_binding(source),),
                    candidate_images=(_execution_binding(candidate),),
                )
            )
        return (
            tuple(units),
            (_execution_binding(static),),
            {
                "renderer": "canonical-test-ovrtx",
                "semantic_assessment": "not_evaluated",
            },
        )


def _write_renderer_command_receipts(output_dir: Path) -> tuple[Path, Path]:
    raw = output_dir / "candidate_evidence" / "raw"
    raw.mkdir(parents=True)
    journal = raw / "usd_cli_command_receipts.jsonl"
    journal.write_text(
        json.dumps(
            {
                "schema_version": (
                    "content-agent-workflows.usd-cli-command-receipt.v1"
                ),
                "arguments": [
                    "render",
                    "--output",
                    str(output_dir / "candidate.png"),
                ],
                "status": "failed",
                "returncode": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint = raw / "usd_cli_command_receipts.checkpoint.json"
    atomic_write_json(
        checkpoint,
        {
            "schema_version": ("content-agent-workflows.usd-cli-receipt-checkpoint.v1"),
            "workflow": "texture-validation",
            "session_id": "failed-renderer-test",
            "receipt_device": journal.stat().st_dev,
            "receipt_inode": journal.stat().st_ino,
            "receipt_sha256": file_sha256(journal),
            "receipt_size_bytes": journal.stat().st_size,
        },
    )
    return journal, checkpoint


class _FailingRendererCollector:
    def collect_candidate_evidence(self, **kwargs: Any) -> Any:
        _write_renderer_command_receipts(Path(kwargs["output_dir"]))
        raise RuntimeError("canonical renderer failed after command receipt")


class _FailingRendererBeforeCheckpointCollector:
    def collect_candidate_evidence(self, **kwargs: Any) -> Any:
        _journal, checkpoint = _write_renderer_command_receipts(
            Path(kwargs["output_dir"])
        )
        checkpoint.unlink()
        raise RuntimeError("canonical renderer failed before receipt checkpoint")


def _tree_identities(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }


def _focused_full_chain(
    tmp_path: Path,
    *,
    stop_after_apply: bool = False,
    stop_after_review: bool = False,
    graph_state_path: Path | None = None,
    source_override: Path | None = None,
    review_disposition: Literal["accept", "reject", "revise"] = "accept",
) -> dict[str, Any]:
    if graph_state_path is not None and source_override is None:
        raise ValueError("Focused graph helper requires its exact staged source")
    source = source_override or _focused_source(tmp_path / "source.usda")

    def attempt_root(leaf_id: str, name: str) -> Path:
        if graph_state_path is not None:
            if leaf_id != TEXTURE_UV_LEAF_ID:
                begin_leaf(graph_state_path, leaf_id)
            return leaf_directory(graph_state_path, leaf_id)
        root = tmp_path / name
        root.mkdir()
        return root

    def complete_graph_leaf(
        leaf_id: str,
        invocation_path: Path,
        result_path: Path,
    ) -> None:
        if graph_state_path is not None:
            complete_leaf(
                graph_state_path,
                leaf_id,
                invocation_path=invocation_path,
                result_path=result_path,
            )

    reference = tmp_path / "reference.png"
    provided_path = tmp_path / "provided.png"
    Image.new("RGB", (64, 64), (20, 30, 40)).save(reference)
    Image.new("RGB", (64, 64), (220, 80, 25)).save(provided_path)
    operations = texture_workflow.TextureOperationSelection(
        generate="requested",
        evidence="requested",
        review="requested",
        publish="requested",
    )
    uv_root = attempt_root(TEXTURE_UV_LEAF_ID, "00-uv")
    uv_invocation = texture_workflow.build_texture_uv_leaf_invocation(
        source,
        output_dir=uv_root,
        target_prim_paths=("/World/Panel",),
        policy="generate_missing",
    )
    uv_invocation_path = uv_root / "invocation.json"
    atomic_write_json(uv_invocation_path, uv_invocation)
    uv_result = texture_workflow.run_texture_uv_leaf(uv_invocation_path)
    uv_result_path = uv_root / "texture_uv_leaf_result.json"
    uv_result_binding = _execution_binding(uv_result_path)
    complete_graph_leaf(
        TEXTURE_UV_LEAF_ID,
        uv_invocation_path,
        uv_result_path,
    )
    prepare_root = attempt_root(texture_workflow.TEXTURE_PREPARE_LEAF_ID, "01-prepare")
    request = texture_workflow.build_texture_capability_request(
        source_asset=source,
        output_dir=prepare_root / "native",
        intent="Apply the exact outer-provided image.",
        material_prim_paths=("/World/Looks/Paint",),
        reference_artifacts=(("appearance_reference", reference),),
        operations=operations,
        texture_size=64,
    )
    prepare_invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        attempt_root=str(prepare_root),
        request=request,
        uv_result=uv_result_binding,
    )
    prepare_path = prepare_root / "invocation.json"
    atomic_write_json(prepare_path, prepare_invocation)
    prior_uv = _tree_identities(uv_root)
    prepare_result = texture_workflow.run_texture_prepare_asset_leaf(
        prepare_path,
        inspector=_FocusedInspector(),
    )
    complete_graph_leaf(
        texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        prepare_path,
        prepare_root / "texture_prepare_leaf_result.json",
    )
    assert _tree_identities(uv_root) == prior_uv
    preparation_binding = prepare_result.native_packet
    preparation = texture_workflow.TexturePreparationPacket.model_validate_json(
        Path(preparation_binding.path).read_bytes()
    )
    request_binding = _execution_binding(
        prepare_root / "native/capability_request.json"
    )
    unit = preparation.inspection.units[0]
    provided = texture_workflow.TextureProvidedImageArtifact(
        unit_id=unit.unit_id,
        channel="albedo",
        artifact=_execution_binding(provided_path),
        producer=texture_workflow.TextureProvidedImageProducer(
            provider="outer-image-tool",
            capability="image.generate.v1",
            invocation_id="exact-outer-invocation",
        ),
    )
    apply_root = attempt_root(
        texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        "02-apply",
    )
    outer_plan = texture_workflow.TextureOuterPlan(
        preparation=preparation_binding,
        source=request.source,
        scope_plan_digest=preparation.scope_plan_digest,
        reference_artifacts=request.reference_artifacts,
        operations=operations,
        targets=(
            texture_workflow.TexturePlanTarget(
                unit_id=unit.unit_id,
                material_prim_paths=unit.material_prim_paths,
                member_prim_paths=unit.member_prim_paths,
                member_subset_paths=unit.member_subset_paths,
                requested_appearance="Use the exact supplied orange image.",
                generator_inputs=texture_workflow.TextureGeneratorInputs(
                    execution_mode="apply_provided",
                    backend="outer_provided_image_apply",
                    prompt="outer-authored exact orange image",
                    texture_size=64,
                    reference_artifacts=tuple(
                        item.artifact for item in request.reference_artifacts
                    ),
                    provided_images=(provided,),
                ),
            ),
        ),
        preservation=texture_workflow.TexturePreservationConstraints(),
        acceptance=texture_workflow.TextureAcceptanceCriteria(
            appearance_requirements=("show the supplied orange image",),
        ),
        capability_constraints=preparation.inspection.capability_constraints,
        stop_policy="Publish only after exact outer acceptance.",
    )
    outer_plan_path = apply_root / "outer_plan.json"
    atomic_write_json(outer_plan_path, outer_plan)
    apply_invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        attempt_root=str(apply_root),
        preparation=preparation_binding,
        outer_plan=_execution_binding(outer_plan_path),
    )
    apply_path = apply_root / "invocation.json"
    atomic_write_json(apply_path, apply_invocation)
    prior_prepare = _tree_identities(prepare_root)
    apply_result = texture_workflow.run_texture_apply_provided_asset_leaf(apply_path)
    complete_graph_leaf(
        texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        apply_path,
        apply_root / "texture_apply_provided_leaf_result.json",
    )
    assert _tree_identities(prepare_root) == prior_prepare
    if stop_after_apply:
        return {
            "source": source,
            "uv_root": uv_root,
            "uv_invocation_path": uv_invocation_path,
            "uv_result": uv_result,
            "uv_result_binding": uv_result_binding,
            "request": request,
            "outer_plan_path": outer_plan_path,
            "apply_result": apply_result,
            "unit_id": unit.unit_id,
            "attempts": (
                (prepare_root, prepare_path, prepare_result),
                (apply_root, apply_path, apply_result),
            ),
        }

    evidence_root = attempt_root(
        texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        "03-evidence",
    )
    evidence_invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        attempt_root=str(evidence_root),
        preparation=preparation_binding,
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
    )
    evidence_path = evidence_root / "invocation.json"
    atomic_write_json(evidence_path, evidence_invocation)
    prior_apply = _tree_identities(apply_root)
    evidence_result = texture_workflow.run_texture_evidence_asset_leaf(
        evidence_path,
        collector=_FocusedCollector(),
    )
    complete_graph_leaf(
        texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        evidence_path,
        evidence_root / "texture_evidence_leaf_result.json",
    )
    assert _tree_identities(apply_root) == prior_apply
    evidence = texture_workflow.TextureCandidateEvidencePacket.model_validate_json(
        Path(evidence_result.native_packet.path).read_bytes()
    )

    review_root = attempt_root(
        texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        "04-review",
    )
    visuals = (
        *(item.artifact for item in request.reference_artifacts),
        *(item.artifact for item in outer_plan.provided_images),
        *(
            binding
            for item in evidence.unit_evidence
            for binding in (*item.source_images, *item.candidate_images)
        ),
    )
    review_input = texture_workflow.TextureOuterReviewInput(
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
        candidate_evidence=evidence_result.native_packet,
        candidate=apply_result.output,
        reference_artifacts=request.reference_artifacts,
        provided_images=outer_plan.provided_images,
        unit_reviews=(
            texture_workflow.TextureOuterUnitReview(
                unit_id=unit.unit_id,
                disposition=review_disposition,
                rationale=(
                    "Exact outer inspection accepted every digest-bound view."
                    if review_disposition == "accept"
                    else "Exact outer inspection requires a fresh candidate."
                ),
            ),
        ),
        inspected_visual_artifacts=visuals,
        findings=(
            (
                "Accepted exact candidate and evidence bytes."
                if review_disposition == "accept"
                else "Publication is blocked pending a fresh candidate."
            ),
        ),
    )
    review_input_path = review_root / "review_input.json"
    atomic_write_json(review_input_path, review_input)
    review_invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        attempt_root=str(review_root),
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
        candidate_evidence=evidence_result.native_packet,
        review_input=_execution_binding(review_input_path),
    )
    review_path = review_root / "invocation.json"
    atomic_write_json(review_path, review_invocation)
    prior_evidence = _tree_identities(evidence_root)
    review_result = texture_workflow.run_texture_review_asset_leaf(review_path)
    assert _tree_identities(evidence_root) == prior_evidence
    if review_disposition == "accept":
        complete_graph_leaf(
            texture_workflow.TEXTURE_REVIEW_LEAF_ID,
            review_path,
            review_root / "texture_review_leaf_result.json",
        )
    if stop_after_review:
        return {
            "source": source,
            "uv_root": uv_root,
            "uv_invocation_path": uv_invocation_path,
            "uv_result": uv_result,
            "uv_result_binding": uv_result_binding,
            "request": request,
            "outer_plan_path": outer_plan_path,
            "apply_result": apply_result,
            "evidence_result": evidence_result,
            "review_result": review_result,
            "unit_id": unit.unit_id,
            "attempts": (
                (prepare_root, prepare_path, prepare_result),
                (apply_root, apply_path, apply_result),
                (evidence_root, evidence_path, evidence_result),
                (review_root, review_path, review_result),
            ),
        }
    if review_disposition != "accept":
        raise ValueError("Nonaccepted focused review cannot proceed to publication")

    publish_root = attempt_root(
        texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
        "05-publish",
    )
    publish_invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
        attempt_root=str(publish_root),
        request_binding=request_binding,
        preparation=preparation_binding,
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
        candidate_evidence=evidence_result.native_packet,
        outer_review=review_result.native_packet,
        output_asset_path=str(publish_root / "native/published.usdz"),
    )
    publish_path = publish_root / "invocation.json"
    atomic_write_json(publish_path, publish_invocation)
    prior_review = _tree_identities(review_root)
    publish_result = texture_workflow.run_texture_publish_asset_leaf(publish_path)
    complete_graph_leaf(
        texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
        publish_path,
        publish_root / "texture_publish_leaf_result.json",
    )
    assert _tree_identities(review_root) == prior_review
    return {
        "source": source,
        "uv_root": uv_root,
        "uv_invocation_path": uv_invocation_path,
        "uv_result": uv_result,
        "uv_result_binding": uv_result_binding,
        "request": request,
        "outer_plan_path": outer_plan_path,
        "apply_result": apply_result,
        "evidence_result": evidence_result,
        "unit_id": unit.unit_id,
        "attempts": (
            (prepare_root, prepare_path, prepare_result),
            (apply_root, apply_path, apply_result),
            (evidence_root, evidence_path, evidence_result),
            (review_root, review_path, review_result),
            (publish_root, publish_path, publish_result),
        ),
    }


def test_focused_full_chain_is_attempt_confined_and_reverified(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    uv_publication = texture_workflow.project_texture_uv_verified_operation(
        attempt_root=chain["uv_root"],
        invocation_path=chain["uv_invocation_path"],
        native_result_path=chain["uv_result_binding"].path,
    )
    assert uv_publication.native_disposition == "passed"
    source_before = Path(chain["source"]).read_bytes()
    for root, invocation_path, result in chain["attempts"]:
        operation = (
            result.leaf_id.removeprefix("texture.")
            .removesuffix(".v1")
            .replace("-", "_")
        )
        publication = texture_workflow.project_texture_focused_verified_operation(
            attempt_root=root,
            invocation_path=invocation_path,
            native_result_path=root / f"texture_{operation}_leaf_result.json",
            native_terminal_receipt_path=(
                root / f"texture_{operation}_terminal_receipt.json"
            ),
        )
        assert publication.native_disposition == "passed"
        assert publication.graph_terminal_action == "complete_leaf"
        assert publication.graph_native_terminal_receipt_path.startswith(str(root))
        assert publication.graph_saved_stage_readback_paths
        assert publication.output == result.output
        if result.renderer_invoked:
            assert publication.resource_claims == (
                f"texture.renderer:{result.leaf_id}",
            )
            assert publication.graph_resource_release_paths == (
                publication.graph_native_terminal_receipt_path,
            )
        else:
            assert publication.resource_claims == ()
            assert publication.graph_resource_release_paths == ()
        projection = json.loads(Path(publication.projection.path).read_text())
        assert projection["provider_status"] == "not_requested"
        assert projection["projector_leaf_invoked"] is False
        assert projection["projector_graph_state_mutated"] is False
        assert all(
            Path(path).is_relative_to(root)
            for path in (
                publication.graph_invocation_path,
                publication.graph_result_path,
                publication.graph_native_terminal_receipt_path,
                *publication.graph_operation_index_paths,
                *publication.graph_evidence_index_paths,
                *publication.graph_evidence_paths,
                *publication.graph_saved_stage_readback_paths,
                *publication.graph_resource_release_paths,
            )
        )
    assert Path(chain["source"]).read_bytes() == source_before


def test_focused_runtime_projectors_preserve_receipts_readbacks_and_releases(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    bindings = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }
    for root, invocation_path, result in chain["attempts"]:
        operation = (
            result.leaf_id.removeprefix("texture.")
            .removesuffix(".v1")
            .replace("-", "_")
        )
        result_path = root / f"texture_{operation}_leaf_result.json"
        receipt_path = root / f"texture_{operation}_terminal_receipt.json"
        before = _tree_identities(root)
        projection = bindings[result.leaf_id].project(
            texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
                invocation_path.read_bytes()
            ),
            result,
            invocation_artifact=_artifact_binding(invocation_path),
            result_artifact=_artifact_binding(result_path),
        )

        payload = projection.payload
        assert payload.native_disposition == result.native_disposition
        assert payload.native_status == result.native_disposition
        assert payload.native_terminal_receipt == _artifact_binding(receipt_path)
        assert payload.evidence == tuple(
            _artifact_binding(Path(item.path)) for item in result.evidence
        )
        assert payload.saved_stage_readbacks == tuple(
            _artifact_binding(Path(item.path)) for item in result.saved_stage_readbacks
        )
        if result.renderer_invoked:
            assert payload.resource_claims == (f"texture.renderer:{result.leaf_id}",)
            assert payload.resource_release_receipts == (
                _artifact_binding(receipt_path),
            )
        else:
            assert payload.resource_claims == ()
            assert payload.resource_release_receipts == ()
        assert _tree_identities(root) == before


def test_focused_runtime_projector_rejects_result_context_substitution(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    root, invocation_path, result = chain["attempts"][0]
    result_path = root / "texture_prepare_leaf_result.json"
    substituted_path = root / "substituted_result.json"
    shutil.copyfile(result_path, substituted_path)
    binding = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }[result.leaf_id]

    with pytest.raises(
        ValueError,
        match="frozen projection context|selected artifacts|exact invocation",
    ):
        binding.project(
            texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
                invocation_path.read_bytes()
            ),
            result,
            invocation_artifact=_artifact_binding(invocation_path),
            result_artifact=_artifact_binding(substituted_path),
        )


def test_focused_runtime_projector_rejects_leaf_identity_substitution(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    publish_root, publish_invocation_path, publish_result = chain["attempts"][-1]
    prepare_binding = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }[texture_workflow.TEXTURE_PREPARE_LEAF_ID]

    with pytest.raises(ValueError, match="frozen projection context"):
        prepare_binding.project(
            texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
                publish_invocation_path.read_bytes()
            ),
            publish_result,
            invocation_artifact=_artifact_binding(publish_invocation_path),
            result_artifact=_artifact_binding(
                publish_root / "texture_publish_leaf_result.json"
            ),
        )


def test_focused_prepare_rejects_substituted_uv_output_source(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    substituted = tmp_path / "substituted-source.usda"
    substituted.write_bytes(Path(chain["source"]).read_bytes())
    attempt = tmp_path / "substituted-prepare"
    attempt.mkdir()
    request = texture_workflow.build_texture_capability_request(
        source_asset=substituted,
        output_dir=attempt / "native",
        intent="Reject a source not produced by the bound UV result.",
        material_prim_paths=("/World/Looks/Paint",),
        operations=chain["request"].operations,
        texture_size=64,
    )
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        attempt_root=str(attempt),
        request=request,
        uv_result=chain["uv_result_binding"],
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    with pytest.raises(ValueError, match="differ from UV output"):
        texture_workflow.run_texture_prepare_asset_leaf(
            invocation_path,
            inspector=_FocusedInspector(),
        )
    assert not (attempt / "native").exists()


def test_focused_runners_reject_symlinked_invocation_before_execution(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-invocation.json"
    outside.write_text("{}\n", encoding="utf-8")
    attempt = tmp_path / "symlink-runner-attempt"
    attempt.mkdir()
    invocation_path = attempt / "invocation.json"
    invocation_path.symlink_to(outside)
    runners: tuple[tuple[Any, dict[str, Any]], ...] = (
        (
            texture_workflow.run_texture_prepare_asset_leaf,
            {"inspector": object()},
        ),
        (texture_workflow.run_texture_apply_provided_asset_leaf, {}),
        (
            texture_workflow.run_texture_evidence_asset_leaf,
            {"collector": object()},
        ),
        (texture_workflow.run_texture_review_asset_leaf, {}),
        (texture_workflow.run_texture_publish_asset_leaf, {}),
    )
    for runner, kwargs in runners:
        with pytest.raises(ValueError, match="symlink|not a regular"):
            runner(invocation_path, **kwargs)
    assert not (attempt / "native").exists()


def test_focused_apply_rejects_native_symlink_precreated_after_initial_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _focused_full_chain(tmp_path)
    generation = texture_workflow.TextureGenerationPacket.model_validate_json(
        Path(chain["apply_result"].native_packet.path).read_bytes()
    )
    attempt = tmp_path / "06-raced-apply"
    attempt.mkdir()
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        attempt_root=str(attempt),
        preparation=generation.preparation,
        outer_plan=_execution_binding(chain["outer_plan_path"]),
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    outside = tmp_path / "outside-native"
    outside.mkdir()
    outside_mode = outside.stat().st_mode
    native_root = attempt / "native"
    real_read_preparation = adapter_module._read_preparation
    injected = False

    def inject_symlink(
        binding: ExecutionArtifactBinding,
    ) -> texture_workflow.TexturePreparationPacket:
        nonlocal injected
        preparation = real_read_preparation(binding)
        native_root.symlink_to(outside, target_is_directory=True)
        injected = True
        return preparation

    def forbidden_generate(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("generator followed a precreated native symlink")

    monkeypatch.setattr(adapter_module, "_read_preparation", inject_symlink)
    monkeypatch.setattr(
        "content_agent_workflows.texture.provided_image_apply."
        "ProvidedImageTextureApplyLeaf.generate",
        forbidden_generate,
    )

    with pytest.raises(
        FileExistsError,
        match="child directory already exists|operation root already exists",
    ):
        texture_workflow.run_texture_apply_provided_asset_leaf(invocation_path)

    assert injected is True
    assert native_root.is_symlink()
    assert outside.stat().st_mode == outside_mode
    assert list(outside.iterdir()) == []


def test_focused_native_claim_cannot_escape_replaced_attempt_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _focused_full_chain(tmp_path, stop_after_apply=True)
    generation = texture_workflow.TextureGenerationPacket.model_validate_json(
        Path(chain["apply_result"].native_packet.path).read_bytes()
    )
    attempt = tmp_path / "06-replaced-attempt"
    moved_attempt = tmp_path / "06-held-attempt"
    attempt.mkdir()
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        attempt_root=str(attempt),
        preparation=generation.preparation,
        outer_plan=_execution_binding(chain["outer_plan_path"]),
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    real_read_preparation = adapter_module._read_preparation

    def replace_attempt(
        binding: ExecutionArtifactBinding,
    ) -> texture_workflow.TexturePreparationPacket:
        preparation = real_read_preparation(binding)
        attempt.rename(moved_attempt)
        attempt.mkdir()
        (attempt / "competitor.txt").write_text("preserve", encoding="utf-8")
        return preparation

    monkeypatch.setattr(adapter_module, "_read_preparation", replace_attempt)

    with pytest.raises(ValueError, match="renamed or removed|identity changed"):
        texture_workflow.run_texture_apply_provided_asset_leaf(invocation_path)

    assert (moved_attempt / "native").is_dir()
    assert stat.S_IMODE((moved_attempt / "native").stat().st_mode) == 0o700
    assert not (attempt / "native").exists()
    assert (attempt / "competitor.txt").read_text(encoding="utf-8") == "preserve"


@pytest.mark.parametrize("operation", ("evidence", "review", "publish"))
@pytest.mark.parametrize("precreated_kind", ("directory", "symlink"))
def test_focused_operations_reject_precreated_explicit_native_root(
    tmp_path: Path,
    operation: str,
    precreated_kind: str,
) -> None:
    chain = _focused_full_chain(tmp_path)
    prepare_result = chain["attempts"][0][2]
    evidence_result = chain["attempts"][2][2]
    _review_root, review_invocation_path, review_result = chain["attempts"][3]
    preparation_binding = prepare_result.native_packet
    preparation = texture_workflow.TexturePreparationPacket.model_validate_json(
        Path(preparation_binding.path).read_bytes()
    )
    outer_plan_binding = _execution_binding(chain["outer_plan_path"])
    outer_plan = texture_workflow.TextureOuterPlan.model_validate_json(
        Path(outer_plan_binding.path).read_bytes()
    )
    generation_binding = chain["apply_result"].native_packet
    generation = texture_workflow.TextureGenerationPacket.model_validate_json(
        Path(generation_binding.path).read_bytes()
    )
    evidence_binding = evidence_result.native_packet
    evidence = texture_workflow.TextureCandidateEvidencePacket.model_validate_json(
        Path(evidence_binding.path).read_bytes()
    )
    review_binding = review_result.native_packet
    review = texture_workflow.TextureOuterReviewPacket.model_validate_json(
        Path(review_binding.path).read_bytes()
    )
    review_invocation = (
        texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
            review_invocation_path.read_bytes()
        )
    )
    assert review_invocation.review_input is not None
    review_input = texture_workflow.TextureOuterReviewInput.model_validate_json(
        Path(review_invocation.review_input.path).read_bytes()
    )

    operation_root = tmp_path / f"precreated-{operation}-native"
    outside = tmp_path / f"outside-{operation}-native"
    outside.mkdir()
    if precreated_kind == "directory":
        operation_root.mkdir()
    else:
        operation_root.symlink_to(outside, target_is_directory=True)
    publication_path = tmp_path / "new-publication" / "published.usdz"

    class ForbiddenCollector:
        def collect_candidate_evidence(self, **_kwargs: Any) -> Any:
            raise AssertionError("evidence collector entered a precreated output root")

    with pytest.raises(
        FileExistsError,
        match="Texture graph operation root already exists",
    ):
        if operation == "evidence":
            texture_workflow.collect_texture_candidate_evidence(
                outer_plan,
                generation,
                preparation=preparation,
                preparation_binding=preparation_binding,
                outer_plan_binding=outer_plan_binding,
                generation_binding=generation_binding,
                collector=ForbiddenCollector(),
                output_dir=operation_root,
            )
        elif operation == "review":
            texture_workflow.record_texture_outer_review(
                review_input,
                outer_plan=outer_plan,
                generation=generation,
                evidence=evidence,
                output_dir=operation_root,
            )
        else:
            texture_workflow.publish_texture_candidate(
                review,
                review_binding=review_binding,
                request_binding=_execution_binding(
                    Path(preparation.request.output_dir) / "capability_request.json"
                ),
                preparation=preparation,
                preparation_binding=preparation_binding,
                outer_plan=outer_plan,
                outer_plan_binding=outer_plan_binding,
                generation=generation,
                generation_binding=generation_binding,
                evidence=evidence,
                evidence_binding=evidence_binding,
                publication_path=publication_path,
                output_dir=operation_root,
            )

    assert list(outside.iterdir()) == []
    if precreated_kind == "directory":
        assert list(operation_root.iterdir()) == []
    else:
        assert operation_root.is_symlink()
    assert not publication_path.parent.exists()


def test_focused_projector_rejects_stale_and_false_native_claims(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    publish_root, publish_invocation, publish_result = chain["attempts"][-1]
    result_path = publish_root / "texture_publish_leaf_result.json"
    receipt_path = publish_root / "texture_publish_terminal_receipt.json"
    Path(publish_result.output.path).write_bytes(b"substituted-published-stage")
    with pytest.raises(ValueError, match="identity changed|scope readback"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=publish_root,
            invocation_path=publish_invocation,
            native_result_path=result_path,
            native_terminal_receipt_path=receipt_path,
        )
    assert not (publish_root / "texture_publish_verified_projection.json").exists()

    apply_root, apply_invocation, _apply_result = chain["attempts"][1]
    apply_result_path = apply_root / "texture_apply_provided_leaf_result.json"
    apply_receipt_path = apply_root / "texture_apply_provided_terminal_receipt.json"
    result_payload = json.loads(apply_result_path.read_text(encoding="utf-8"))
    result_payload["provider_invoked"] = True
    atomic_write_json(apply_result_path, result_payload)
    with pytest.raises(ValueError, match="Invalid Texture focused native result"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=apply_root,
            invocation_path=apply_invocation,
            native_result_path=apply_result_path,
            native_terminal_receipt_path=apply_receipt_path,
        )


def test_focused_projector_rejects_cross_attempt_escape_and_symlink(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    evidence_root, evidence_invocation, _result = chain["attempts"][2]
    result_path = evidence_root / "texture_evidence_leaf_result.json"
    receipt_path = evidence_root / "texture_evidence_terminal_receipt.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["evidence"][0]["path"] = str(
        chain["attempts"][1][0] / "texture_apply_provided_leaf_result.json"
    )
    atomic_write_json(result_path, payload)
    with pytest.raises(ValueError, match="differs from recomputed facts"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=evidence_root,
            invocation_path=evidence_invocation,
            native_result_path=result_path,
            native_terminal_receipt_path=receipt_path,
        )
    outside_binding = _execution_binding(
        chain["attempts"][1][0] / "texture_apply_provided_leaf_result.json"
    )
    with adapter_module._HeldDirectory.open(evidence_root) as held:
        with pytest.raises(ValueError, match="escapes its attempt"):
            adapter_module._HeldAttemptTree(held).verify_binding(
                outside_binding,
                label="cross-attempt evidence",
            )

    symlink_root = tmp_path / "symlink-attempt"
    symlink_root.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")
    (symlink_root / "invocation.json").symlink_to(outside)
    with pytest.raises(ValueError, match="must not contain symlinks|not a regular"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=symlink_root,
            invocation_path=symlink_root / "invocation.json",
            native_result_path=symlink_root / "missing-result.json",
            native_terminal_receipt_path=symlink_root / "missing-receipt.json",
        )


def test_focused_invocation_rejects_missing_unknown_duplicate_and_mixed_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve() / "attempt"
    with pytest.raises(ValueError, match="invocation fields differ"):
        texture_workflow.TextureFocusedLeafInvocation(
            leaf_id=texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
            attempt_root=str(root),
        )
    with pytest.raises(ValueError, match="extra_forbidden"):
        texture_workflow.TextureFocusedLeafInvocation.model_validate(
            {
                "leaf_id": texture_workflow.TEXTURE_PREPARE_LEAF_ID,
                "attempt_root": str(root),
                "request": {},
                "unknown_packet": {},
            }
        )
    with pytest.raises(ValueError, match="invocation fields differ"):
        texture_workflow.TextureFocusedLeafInvocation(
            leaf_id=texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
            attempt_root=str(root),
            preparation=ExecutionArtifactBinding(
                path="/missing/preparation.json",
                sha256="0" * 64,
                size_bytes=0,
            ),
            outer_plan=ExecutionArtifactBinding(
                path="/missing/plan.json",
                sha256="1" * 64,
                size_bytes=0,
            ),
            review_input=ExecutionArtifactBinding(
                path="/mixed/legacy-agent-step.json",
                sha256="2" * 64,
                size_bytes=0,
            ),
        )
    with pytest.raises(ValueError, match="extra_forbidden"):
        texture_workflow.TextureFocusedLeafInvocation.model_validate(
            {
                "leaf_id": texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
                "attempt_root": str(root),
                "preparation": {
                    "path": "/missing/preparation.json",
                    "sha256": "0" * 64,
                    "size_bytes": 0,
                },
                "outer_plan": {
                    "path": "/missing/plan.json",
                    "sha256": "1" * 64,
                    "size_bytes": 0,
                },
                "preparation_duplicate": {
                    "path": "/substituted/preparation.json",
                    "sha256": "2" * 64,
                    "size_bytes": 0,
                },
            }
        )


def test_focused_scope_rejection_seals_failed_result_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_validate = adapter_module.validate_texture_scope_invariants

    def reject_scope(**kwargs: Any) -> Any:
        report = real_validate(**kwargs)
        return type(report).model_validate(
            {
                **report.model_dump(mode="json"),
                "passed": False,
                "geometry_unchanged": False,
                "violations": [
                    {
                        "code": "test_topology_rejection",
                        "prim_path": "/World/Untouched",
                        "summary": "deterministic validation rejected topology",
                    }
                ],
            }
        )

    monkeypatch.setattr(
        adapter_module,
        "validate_texture_scope_invariants",
        reject_scope,
    )
    chain = _focused_full_chain(tmp_path, stop_after_apply=True)
    root, invocation_path, result = chain["attempts"][-1]
    result_path = root / "texture_apply_provided_leaf_result.json"
    receipt_path = root / "texture_apply_provided_terminal_receipt.json"

    assert result.native_disposition == "failed"
    assert result.error is not None
    assert "test_topology_rejection" in result.error
    receipt = texture_workflow.TextureFocusedTerminalReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    assert receipt.native_disposition == "failed"
    assert receipt.error == result.error
    readback = texture_workflow.TextureFocusedSavedStageReadback.model_validate_json(
        Path(result.saved_stage_readbacks[0].path).read_bytes()
    )
    assert readback.scope_invariant_report.passed is False
    assert readback.non_target_topology_preserved is False
    assert readback.non_target_physics_preserved is False

    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=root,
        invocation_path=invocation_path,
        native_result_path=result_path,
        native_terminal_receipt_path=receipt_path,
    )
    assert publication.native_disposition == "failed"
    assert publication.error == result.error
    assert publication.graph_terminal_action == "fail_leaf"

    binding = {
        item.descriptor.leaf_id: item for item in texture_asset_leaf_runtime_bindings()
    }[texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID]
    projection = binding.project(
        texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
            invocation_path.read_bytes()
        ),
        result,
        invocation_artifact=_artifact_binding(invocation_path),
        result_artifact=_artifact_binding(result_path),
    )
    assert projection.payload.native_disposition == "failed"
    assert projection.payload.error == result.error


def test_focused_prepare_exception_seals_failed_result_and_receipt(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path, stop_after_apply=True)
    attempt = tmp_path / "06-failed-prepare"
    attempt.mkdir()
    request = texture_workflow.build_texture_capability_request(
        source_asset=chain["source"],
        output_dir=attempt / "native",
        intent="Seal an exact focused preparation failure.",
        material_prim_paths=("/World/Looks/Paint",),
        reference_artifacts=tuple(
            (item.role, item.artifact.path)
            for item in chain["request"].reference_artifacts
        ),
        operations=chain["request"].operations,
        texture_size=64,
    )
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        attempt_root=str(attempt),
        request=request,
        uv_result=chain["uv_result_binding"],
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    class FailingInspector:
        def inspect(self, **_kwargs: Any) -> Any:
            raise RuntimeError("focused inspector failed after native claim")

    result = texture_workflow.run_texture_prepare_asset_leaf(
        invocation_path,
        inspector=FailingInspector(),
    )
    result_path = attempt / "texture_prepare_leaf_result.json"
    receipt_path = attempt / "texture_prepare_terminal_receipt.json"

    assert result.native_disposition == "failed"
    assert result.error == ("RuntimeError: focused inspector failed after native claim")
    assert result.output == result.source
    assert result.renderer_invoked is False
    assert stat.S_IMODE((attempt / "native").stat().st_mode) == 0o700
    native_payload = json.loads(
        Path(result.native_packet.path).read_text(encoding="utf-8")
    )
    assert native_payload["schema_version"] == (
        "content-agent-workflows.texture-focused-native-failure.v1"
    )
    readback = adapter_module.TextureFocusedFailureReadback.model_validate_json(
        Path(result.saved_stage_readbacks[0].path).read_bytes()
    )
    assert readback.error == result.error
    receipt = texture_workflow.TextureFocusedTerminalReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    assert receipt.resources_released is True
    assert receipt.error == result.error
    assert receipt.renderer_invoked is False
    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=attempt,
        invocation_path=invocation_path,
        native_result_path=result_path,
        native_terminal_receipt_path=receipt_path,
    )
    assert publication.graph_terminal_action == "fail_leaf"
    assert publication.error == result.error
    assert publication.resource_claims == (
        f"texture.renderer:{texture_workflow.TEXTURE_PREPARE_LEAF_ID}",
    )
    assert publication.graph_resource_release_paths == (str(receipt_path),)

    binding = {
        item.descriptor.leaf_id: item
        for item in texture_workflow.texture_asset_leaf_runtime_bindings()
    }[texture_workflow.TEXTURE_PREPARE_LEAF_ID]
    projection = binding.project(
        invocation,
        result,
        invocation_artifact=_artifact_binding(invocation_path),
        result_artifact=_artifact_binding(result_path),
    )
    assert projection.payload.native_disposition == "failed"
    assert projection.payload.resource_claims == (
        f"texture.renderer:{texture_workflow.TEXTURE_PREPARE_LEAF_ID}",
    )
    assert projection.payload.resource_release_receipts == (
        _artifact_binding(receipt_path),
    )


def test_focused_evidence_renderer_failure_projects_release_and_fails_graph(
    tmp_path: Path,
) -> None:
    selected_leaf_ids = (
        TEXTURE_UV_LEAF_ID,
        texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
    )
    state_path, staged = _create_graph_run(
        tmp_path,
        stage_builder=_focused_source,
        selected_leaf_ids=selected_leaf_ids,
    )
    chain = _focused_full_chain(
        tmp_path,
        graph_state_path=state_path,
        source_override=staged,
        stop_after_apply=True,
    )
    apply_result = chain["apply_result"]
    generation = texture_workflow.TextureGenerationPacket.model_validate_json(
        Path(apply_result.native_packet.path).read_bytes()
    )
    begin_leaf(state_path, texture_workflow.TEXTURE_EVIDENCE_LEAF_ID)
    attempt = leaf_directory(state_path, texture_workflow.TEXTURE_EVIDENCE_LEAF_ID)
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        attempt_root=str(attempt),
        preparation=generation.preparation,
        outer_plan=_execution_binding(chain["outer_plan_path"]),
        generation=apply_result.native_packet,
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = texture_workflow.run_texture_evidence_asset_leaf(
        invocation_path,
        collector=_FailingRendererCollector(),
    )
    result_path = attempt / "texture_evidence_leaf_result.json"
    receipt_path = attempt / "texture_evidence_terminal_receipt.json"
    native_failure = adapter_module.TextureFocusedNativeFailure.model_validate_json(
        Path(result.native_packet.path).read_bytes()
    )

    assert result.native_disposition == "failed"
    assert result.error == (
        "RuntimeError: canonical renderer failed after command receipt"
    )
    assert result.output == apply_result.output
    assert result.renderer_invoked is True
    assert native_failure.packet_chain == (
        invocation.preparation,
        invocation.outer_plan,
        invocation.generation,
    )
    assert native_failure.renderer_evidence == result.evidence[1:]
    assert tuple(Path(item.path).name for item in result.evidence[1:]) == (
        "usd_cli_command_receipts.jsonl",
        "usd_cli_command_receipts.checkpoint.json",
    )
    receipt = texture_workflow.TextureFocusedTerminalReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    assert receipt.renderer_invoked is True
    assert receipt.resources_released is True

    binding = {
        item.descriptor.leaf_id: item
        for item in texture_workflow.texture_asset_leaf_runtime_bindings()
    }[texture_workflow.TEXTURE_EVIDENCE_LEAF_ID]
    projection = binding.project(
        invocation,
        result,
        invocation_artifact=_artifact_binding(invocation_path),
        result_artifact=_artifact_binding(result_path),
    )
    assert projection.payload.native_disposition == "failed"
    assert projection.payload.evidence == tuple(
        _artifact_binding(Path(item.path)) for item in result.evidence
    )
    assert projection.payload.resource_claims == (
        f"texture.renderer:{texture_workflow.TEXTURE_EVIDENCE_LEAF_ID}",
    )
    assert projection.payload.resource_release_receipts == (
        _artifact_binding(receipt_path),
    )

    failed = fail_leaf(
        state_path,
        texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        reason=result.error,
        invocation_path=invocation_path,
        result_path=result_path,
    )
    failed_state = failed.leaf_states[texture_workflow.TEXTURE_EVIDENCE_LEAF_ID]
    assert failed_state.status == "failed"
    assert failed_state.receipt is not None
    assert failed.leaf_states[texture_workflow.TEXTURE_REVIEW_LEAF_ID].status == (
        "pending"
    )
    graph_receipt = AssetLeafReceipt.model_validate_json(
        Path(failed_state.receipt.path).read_bytes()
    )
    assert graph_receipt.native_disposition == "failed"
    assert "resource_release" in graph_receipt.required_artifact_categories
    assert graph_receipt.resource_release_receipts == [
        _artifact_binding(receipt_path),
    ]

    journal_path = Path(result.evidence[1].path)
    journal_path.write_text("substituted renderer receipt\n", encoding="utf-8")
    with pytest.raises(
        ValueError,
        match=(
            "renderer receipt custody changed|invalid JSON|"
            "checkpoint does not bind its journal"
        ),
    ):
        binding.project(
            invocation,
            result,
            invocation_artifact=_artifact_binding(invocation_path),
            result_artifact=_artifact_binding(result_path),
        )


def test_focused_evidence_failure_seals_interrupted_renderer_receipt_pair(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path, stop_after_apply=True)
    apply_result = chain["apply_result"]
    generation = texture_workflow.TextureGenerationPacket.model_validate_json(
        Path(apply_result.native_packet.path).read_bytes()
    )
    attempt = tmp_path / "06-interrupted-renderer-receipt"
    attempt.mkdir()
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        attempt_root=str(attempt),
        preparation=generation.preparation,
        outer_plan=_execution_binding(chain["outer_plan_path"]),
        generation=apply_result.native_packet,
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    result = texture_workflow.run_texture_evidence_asset_leaf(
        invocation_path,
        collector=_FailingRendererBeforeCheckpointCollector(),
    )
    result_path = attempt / "texture_evidence_leaf_result.json"
    receipt_path = attempt / "texture_evidence_terminal_receipt.json"
    native_failure = adapter_module.TextureFocusedNativeFailure.model_validate_json(
        Path(result.native_packet.path).read_bytes()
    )

    assert result.native_disposition == "failed"
    assert result.renderer_invoked is True
    assert result.error is not None
    assert "failed before receipt checkpoint" in result.error
    assert "journal and checkpoint are unpaired" in result.error
    assert native_failure.renderer_receipt_custody_error is not None
    assert "journal and checkpoint are unpaired" in (
        native_failure.renderer_receipt_custody_error
    )
    assert tuple(Path(item.path).name for item in native_failure.renderer_evidence) == (
        "usd_cli_command_receipts.jsonl",
    )
    receipt = texture_workflow.TextureFocusedTerminalReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    assert receipt.native_disposition == "failed"
    assert receipt.error == result.error
    assert receipt.renderer_invoked is True
    assert receipt.resources_released is True
    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=attempt,
        invocation_path=invocation_path,
        native_result_path=result_path,
        native_terminal_receipt_path=receipt_path,
    )
    assert publication.graph_terminal_action == "fail_leaf"
    assert publication.error == result.error
    assert publication.graph_resource_release_paths == (str(receipt_path),)


@pytest.mark.parametrize(
    ("leaf_id", "runner", "capability_name", "chain_fields"),
    (
        (
            texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
            texture_workflow.run_texture_apply_provided_asset_leaf,
            "invoke_texture_generator",
            ("preparation", "outer_plan"),
        ),
        (
            texture_workflow.TEXTURE_REVIEW_LEAF_ID,
            texture_workflow.run_texture_review_asset_leaf,
            "record_texture_outer_review",
            ("outer_plan", "generation", "candidate_evidence", "review_input"),
        ),
        (
            texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
            texture_workflow.run_texture_publish_asset_leaf,
            "publish_texture_candidate",
            (
                "request_binding",
                "preparation",
                "outer_plan",
                "generation",
                "candidate_evidence",
                "outer_review",
            ),
        ),
    ),
)
def test_focused_exception_failures_recompute_every_invocation_chain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    leaf_id: str,
    runner: Callable[..., Any],
    capability_name: str,
    chain_fields: tuple[str, ...],
) -> None:
    chain = _focused_full_chain(tmp_path)
    _source_root, source_invocation_path, _source_result = next(
        item
        for item in chain["attempts"]
        if texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
            item[1].read_bytes()
        ).leaf_id
        == leaf_id
    )
    source_invocation = (
        texture_workflow.TextureFocusedLeafInvocation.model_validate_json(
            source_invocation_path.read_bytes()
        )
    )
    attempt = tmp_path / f"06-failed-{leaf_id}"
    attempt.mkdir()
    invocation_payload = source_invocation.model_dump(mode="python")
    invocation_payload["attempt_root"] = str(attempt)
    if leaf_id == texture_workflow.TEXTURE_PUBLISH_LEAF_ID:
        invocation_payload["output_asset_path"] = str(attempt / "native/published.usdz")
    invocation = texture_workflow.TextureFocusedLeafInvocation.model_validate(
        invocation_payload
    )
    invocation_path = attempt / "invocation.json"
    atomic_write_json(invocation_path, invocation)

    def fail_operation(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError(f"{leaf_id} failed after native claim")

    monkeypatch.setattr(capabilities_module, capability_name, fail_operation)
    result = runner(invocation_path)
    native_failure_path = Path(result.native_packet.path)
    native_failure = adapter_module.TextureFocusedNativeFailure.model_validate_json(
        native_failure_path.read_bytes()
    )
    expected_chain = tuple(getattr(invocation, name) for name in chain_fields)

    assert result.native_disposition == "failed"
    assert result.output == result.source
    assert native_failure.packet_chain == expected_chain
    assert all(item is not None for item in expected_chain)
    assert result.error == f"RuntimeError: {leaf_id} failed after native claim"
    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=attempt,
        invocation_path=invocation_path,
        native_result_path=attempt
        / f"texture_{leaf_id[8:-3].replace('-', '_')}_leaf_result.json",
        native_terminal_receipt_path=(
            attempt / f"texture_{leaf_id[8:-3].replace('-', '_')}_terminal_receipt.json"
        ),
    )
    assert publication.graph_terminal_action == "fail_leaf"

    failure_values = native_failure.model_dump(
        mode="python",
        exclude={"schema_version", "failure_identity_sha256"},
    )
    failure_values["packet_chain"] = (native_failure.invocation,)
    substituted = adapter_module.TextureFocusedNativeFailure.create(**failure_values)
    atomic_write_json(native_failure_path, substituted)
    with adapter_module._HeldDirectory.open(attempt) as held_attempt:
        with pytest.raises(ValueError, match="changed its exact source, output"):
            adapter_module._focused_native_contract(
                held_attempt,
                _execution_binding(invocation_path),
                invocation,
                _execution_binding(native_failure_path),
            )


def test_focused_review_retains_review_and_scope_rejections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_detail = adapter_module._scope_rejection_detail
    call_count = 0

    def reject_review_scope(report: Any) -> str | None:
        nonlocal call_count
        call_count += 1
        detail = real_detail(report)
        if call_count == 4:
            assert detail is None
            return "Texture focused saved-stage scope readback rejected: injected"
        return detail

    monkeypatch.setattr(
        adapter_module,
        "_scope_rejection_detail",
        reject_review_scope,
    )
    chain = _focused_full_chain(
        tmp_path,
        stop_after_review=True,
        review_disposition="revise",
    )
    result = chain["attempts"][-1][2]

    assert result.native_disposition == "failed"
    assert result.error is not None
    assert "Texture outer review did not accept every unit" in result.error
    assert "saved-stage scope readback rejected: injected" in result.error


def test_focused_review_projector_preserves_failure_without_upgrade(
    tmp_path: Path,
) -> None:
    chain = _focused_full_chain(tmp_path)
    request = chain["request"]
    outer_plan_path = chain["outer_plan_path"]
    apply_result = chain["apply_result"]
    evidence_result = chain["evidence_result"]
    evidence = texture_workflow.TextureCandidateEvidencePacket.model_validate_json(
        Path(evidence_result.native_packet.path).read_bytes()
    )
    review_root = tmp_path / "06-rejected-review"
    review_root.mkdir()
    review_input = texture_workflow.TextureOuterReviewInput(
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
        candidate_evidence=evidence_result.native_packet,
        candidate=apply_result.output,
        reference_artifacts=request.reference_artifacts,
        provided_images=evidence.provided_images,
        unit_reviews=(
            texture_workflow.TextureOuterUnitReview(
                unit_id=chain["unit_id"],
                disposition="revise",
                rationale="Exact outer review requires a fresh candidate.",
            ),
        ),
        inspected_visual_artifacts=(
            *(item.artifact for item in request.reference_artifacts),
            *(item.artifact for item in evidence.provided_images),
            *(
                binding
                for item in evidence.unit_evidence
                for binding in (*item.source_images, *item.candidate_images)
            ),
        ),
        findings=("Revision is required; no publication authority is granted.",),
    )
    review_input_path = review_root / "review_input.json"
    atomic_write_json(review_input_path, review_input)
    invocation = texture_workflow.TextureFocusedLeafInvocation(
        leaf_id=texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        attempt_root=str(review_root),
        outer_plan=_execution_binding(outer_plan_path),
        generation=apply_result.native_packet,
        candidate_evidence=evidence_result.native_packet,
        review_input=_execution_binding(review_input_path),
    )
    invocation_path = review_root / "invocation.json"
    atomic_write_json(invocation_path, invocation)
    result = texture_workflow.run_texture_review_asset_leaf(invocation_path)
    expected_error = (
        "Texture outer review did not accept every unit: "
        f"{chain['unit_id']}: revise: Exact outer review requires a fresh candidate."
    )
    assert result.native_disposition == "failed"
    assert result.error == expected_error
    result_path = review_root / "texture_review_leaf_result.json"
    receipt_path = review_root / "texture_review_terminal_receipt.json"
    receipt = texture_workflow.TextureFocusedTerminalReceipt.model_validate_json(
        receipt_path.read_bytes()
    )
    assert receipt.native_disposition == "failed"
    assert receipt.error == expected_error
    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=review_root,
        invocation_path=invocation_path,
        native_result_path=result_path,
        native_terminal_receipt_path=receipt_path,
    )
    assert publication.native_disposition == "failed"
    assert publication.error == expected_error
    assert publication.graph_terminal_action == "fail_leaf"

    binding = {
        item.descriptor.leaf_id: item
        for item in texture_workflow.texture_asset_leaf_runtime_bindings()
    }[texture_workflow.TEXTURE_REVIEW_LEAF_ID]
    projection = binding.project(
        invocation,
        result,
        invocation_artifact=_artifact_binding(invocation_path),
        result_artifact=_artifact_binding(result_path),
    )
    assert projection.payload.native_disposition == "failed"
    assert projection.payload.native_status == "failed"
    assert projection.payload.error == expected_error
    assert projection.payload.native_terminal_receipt == _artifact_binding(receipt_path)

    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["native_disposition"] = "passed"
    payload["error"] = None
    atomic_write_json(result_path, payload)
    with pytest.raises(ValueError, match="differs from recomputed facts"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=review_root,
            invocation_path=invocation_path,
            native_result_path=result_path,
            native_terminal_receipt_path=receipt_path,
        )


def test_focused_graph_review_failure_recovers_without_unblocking_publish(
    tmp_path: Path,
) -> None:
    selected_leaf_ids = (
        TEXTURE_UV_LEAF_ID,
        texture_workflow.TEXTURE_PREPARE_LEAF_ID,
        texture_workflow.TEXTURE_APPLY_PROVIDED_LEAF_ID,
        texture_workflow.TEXTURE_EVIDENCE_LEAF_ID,
        texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        texture_workflow.TEXTURE_PUBLISH_LEAF_ID,
    )
    state_path, staged = _create_graph_run(
        tmp_path,
        stage_builder=_focused_source,
        selected_leaf_ids=selected_leaf_ids,
    )
    chain = _focused_full_chain(
        tmp_path,
        graph_state_path=state_path,
        source_override=staged,
        review_disposition="revise",
        stop_after_review=True,
    )
    review_root, invocation_path, result = chain["attempts"][-1]
    result_path = review_root / "texture_review_leaf_result.json"
    expected_error = (
        "Texture outer review did not accept every unit: "
        f"{chain['unit_id']}: revise: "
        "Exact outer inspection requires a fresh candidate."
    )
    assert result.native_disposition == "failed"
    assert result.error == expected_error

    failed = fail_leaf(
        state_path,
        texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        reason=expected_error,
        invocation_path=invocation_path,
        result_path=result_path,
    )
    failed_state = failed.leaf_states[texture_workflow.TEXTURE_REVIEW_LEAF_ID]
    failed_receipt_binding = failed_state.receipt
    assert failed_state.status == "failed"
    assert failed_receipt_binding is not None
    assert failed.leaf_states[texture_workflow.TEXTURE_PUBLISH_LEAF_ID].status == (
        "pending"
    )
    failed_receipt = AssetLeafReceipt.model_validate_json(
        Path(failed_receipt_binding.path).read_bytes()
    )
    assert failed_receipt.native_disposition == "failed"
    assert failed_receipt.native_status == "failed"
    assert failed_receipt.error == expected_error
    assert failed_receipt.invocation == _artifact_binding(invocation_path)
    assert failed_receipt.result == _artifact_binding(result_path)
    assert failed_receipt.projection is not None
    failed_projection = AssetLeafProjection.model_validate_json(
        Path(failed_receipt.projection.path).read_bytes()
    )
    assert failed_projection.payload.native_disposition == "failed"
    assert failed_projection.payload.error == expected_error

    recovered = recover_leaf(
        state_path,
        texture_workflow.TEXTURE_REVIEW_LEAF_ID,
        reason="Revise the rejected candidate in a fresh graph attempt.",
    )
    recovered_state = recovered.leaf_states[texture_workflow.TEXTURE_REVIEW_LEAF_ID]
    assert recovered_state.status == "ready"
    assert recovered_state.receipt is None
    assert recovered_state.superseded_receipts == [failed_receipt_binding]
    assert recovered.leaf_states[texture_workflow.TEXTURE_PUBLISH_LEAF_ID].status == (
        "pending"
    )


def test_focused_projector_detects_nested_artifact_replacement_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _focused_full_chain(tmp_path)
    prepare_root, invocation_path, result = chain["attempts"][0]
    native_packet = Path(result.native_packet.path)
    competitor = native_packet.with_name("competing-preparation.json")
    competitor.write_bytes(native_packet.read_bytes())
    real_open = adapter_module._HeldAttemptTree._open
    opens = 0

    def replace_between_reads(
        tree: adapter_module._HeldAttemptTree,
        path: str | Path,
    ) -> int:
        nonlocal opens
        if Path(path) == native_packet:
            if opens == 1:
                competitor.replace(native_packet)
            opens += 1
        return real_open(tree, path)

    monkeypatch.setattr(
        adapter_module._HeldAttemptTree,
        "_open",
        replace_between_reads,
    )
    with pytest.raises(ValueError, match="identity changed after read"):
        texture_workflow.project_texture_focused_verified_operation(
            attempt_root=prepare_root,
            invocation_path=invocation_path,
            native_result_path=prepare_root / "texture_prepare_leaf_result.json",
            native_terminal_receipt_path=(
                prepare_root / "texture_prepare_terminal_receipt.json"
            ),
        )
    assert not (prepare_root / "texture_prepare_verified_projection.json").exists()


def test_focused_projector_invokes_no_provider_controller_or_graph_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chain = _focused_full_chain(tmp_path)
    apply_root, invocation_path, _result = chain["attempts"][1]

    def tripwire(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("focused projector crossed its non-executing boundary")

    for name in (
        "run_texture_prepare_asset_leaf",
        "run_texture_apply_provided_asset_leaf",
        "run_texture_evidence_asset_leaf",
        "run_texture_review_asset_leaf",
        "run_texture_publish_asset_leaf",
    ):
        monkeypatch.setattr(adapter_module, name, tripwire)
    for name in ("begin_leaf", "complete_leaf", "fail_leaf", "cancel_leaf"):
        monkeypatch.setattr(asset_composition_module, name, tripwire)
    monkeypatch.setattr(
        texture_workflow.ProvidedImageTextureApplyLeaf,
        "generate",
        tripwire,
    )
    monkeypatch.setattr(
        texture_workflow.TextureAgentServiceClient,
        "__init__",
        tripwire,
    )
    publication = texture_workflow.project_texture_focused_verified_operation(
        attempt_root=apply_root,
        invocation_path=invocation_path,
        native_result_path=apply_root / "texture_apply_provided_leaf_result.json",
        native_terminal_receipt_path=(
            apply_root / "texture_apply_provided_terminal_receipt.json"
        ),
    )
    assert publication.native_disposition == "passed"
