# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scene decomposition workflow implementation."""

from __future__ import annotations

import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from content_agent_workflows.common.artifacts import (
    artifact_set_digest,
    atomic_write_json,
    file_sha256,
    seal_phase_result,
)

from .adapter_material_agent_scene import convert_material_agent_manifest
from .manifest import (
    ArtifactReference,
    DecompositionPhaseResult,
    DecompositionPolicy,
    ManifestCatalog,
    ManifestCatalogEntry,
    SceneDecompositionRequest,
    SceneDecompositionResult,
    write_manifest_json,
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    return atomic_write_json(path, payload)


def _load_material_agent_scene_functions() -> tuple[
    Callable[..., Any], Callable[..., Any]
]:
    """Load Material Agent scene functions lazily.

    The generic workflow package uses Material Agent scene decomposition as an
    adapter, but the public contracts in this package remain domain-neutral.
    """

    try:
        from material_agent.scene.analyze import analyze_scene
        from material_agent.scene.extract import extract_all
    except Exception as exc:  # pragma: no cover - exercised via caller error path
        raise RuntimeError(
            "Scene decomposition currently requires the optional Material Agent "
            "scene package to be importable."
        ) from exc
    return analyze_scene, extract_all


def _decomposition_policy(request: SceneDecompositionRequest) -> DecompositionPolicy:
    include_paths = list(request.include_paths)
    if request.root_prim_path and request.root_prim_path not in include_paths:
        include_paths.insert(0, request.root_prim_path)

    return DecompositionPolicy(
        refinement_mode="external_llm" if request.enable_llm_refinement else "none",
        decomposition_intent=request.decomposition_intent,
        root_prim_path=request.root_prim_path,
        include_paths=include_paths,
        exclude_paths=list(request.exclude_paths),
        min_mesh_count=request.min_mesh_count,
        exclude_invisible_assets=request.exclude_invisible_assets,
        detect_structural_duplicates=request.detect_structural_duplicates,
        detect_payload_groups=request.detect_payload_groups,
        detect_native_prototypes=request.detect_native_prototypes,
        extract_large_payload_representatives=request.extract_large_payload_representatives,
        extract_assets=request.extract_assets,
        flatten_extracts=request.flatten_extracts,
        skip_geometry=request.skip_geometry,
        llm_refinement_enabled=request.enable_llm_refinement,
    )


def _material_agent_filters(policy: DecompositionPolicy) -> dict[str, Any]:
    return {
        "include_paths": policy.include_paths,
        "exclude_paths": policy.exclude_paths,
        "min_mesh_count": policy.min_mesh_count,
        "exclude_invisible_assets": policy.exclude_invisible_assets,
        "detect_structural_duplicates": policy.detect_structural_duplicates,
    }


def run_scene_decomposition(
    request: SceneDecompositionRequest,
    *,
    input_digest: str | None = None,
) -> SceneDecompositionResult:
    """Run scene decomposition and write canonical artifacts."""

    output_dir = request.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_payload = request.model_dump(mode="json")
    request_path = _write_json(
        output_dir / "scene_decomposition_request.json",
        request_payload,
    )
    policy = _decomposition_policy(request)
    phase_result_path = output_dir / "decomposition_result.json"
    source_path = request.usd_path.resolve()
    source_identity_digest = "unavailable"
    effective_input_digest = input_digest or "unavailable"

    try:
        source_identity_digest = artifact_set_digest([source_path])
        if input_digest is None:
            digest_request_payload = request.model_dump(
                mode="json",
                exclude={"output_dir", "usd_path"},
            )
            if not request.enable_llm_refinement:
                digest_request_payload["llm_config"] = None
            effective_input_digest = artifact_set_digest(
                [source_path],
                metadata={
                    "schema": "content-agent-workflows.standalone-decomposition-input.v1",
                    "manifest_id": request.manifest_id,
                    "decomposition_intent": request.decomposition_intent,
                    "request": digest_request_payload,
                },
            )
        analyze_scene, extract_all = _load_material_agent_scene_functions()
        analysis_working_dir = output_dir / "analysis_working"
        material_manifest = analyze_scene(
            scene_usd_path=source_path,
            skip_geometry=request.skip_geometry,
            building_block_min_reuse=request.building_block_min_reuse,
            filters=_material_agent_filters(policy),
            llm_config=request.llm_config if request.enable_llm_refinement else None,
            working_dir=analysis_working_dir,
            detect_payload_groups=request.detect_payload_groups,
            detect_native_prototypes=request.detect_native_prototypes,
            extract_large_payload_representatives=(
                request.extract_large_payload_representatives
            ),
        )

        extracted_dir = output_dir / "extracted"
        if request.extract_assets:
            material_manifest = extract_all(
                scene_usd_path=source_path,
                manifest=material_manifest,
                output_dir=extracted_dir,
                names_filter=request.asset_filter or None,
                flatten=request.flatten_extracts,
                max_workers=request.extract_workers,
            )

        material_manifest_path: Path | None = None
        mapping_artifacts = [
            ArtifactReference(
                path=str(request_path),
                kind="request",
                description="Scene decomposition request.",
            )
        ]
        if request.write_material_agent_manifest:
            material_manifest_path = output_dir / "material_agent_scene_manifest.json"
            material_manifest.save(material_manifest_path)
            mapping_artifacts.append(
                ArtifactReference(
                    path=str(material_manifest_path),
                    kind="source_manifest",
                    description="Raw Material Agent scene manifest used by the adapter.",
                )
            )
        if request.extract_assets:
            mapping_artifacts.append(
                ArtifactReference(
                    path=str(extracted_dir),
                    kind="extracted_assets_dir",
                    description="Directory containing extracted per-asset USD files.",
                )
            )

        manifest = convert_material_agent_manifest(
            material_manifest,
            policy=policy,
            mapping_artifacts=mapping_artifacts,
        )
        manifest.scene_id = source_path.stem
        manifest.original_usd_path = str(source_path)
        manifest_path = write_manifest_json(
            manifest, output_dir / "scene_manifest.json"
        )
        manifest_digest = file_sha256(manifest_path)
        catalog = ManifestCatalog(
            original_usd_path=str(source_path),
            source_identity_digest=source_identity_digest,
            structural_analysis_id=manifest_digest,
            manifests=[
                ManifestCatalogEntry(
                    manifest_id=request.manifest_id,
                    intent=request.decomposition_intent,
                    path=str(manifest_path),
                    finalized=True,
                    manifest_digest=manifest_digest,
                )
            ],
        )
        catalog_path = atomic_write_json(output_dir / "manifest_catalog.json", catalog)

        extracted_asset_paths = sorted(
            {
                str(Path(asset.working_usd_path).expanduser().resolve())
                for asset in manifest.assets
                if asset.working_usd_path
            }
        )
        unresolved_issues: list[str] = []
        if request.extract_assets:
            for asset in manifest.processable_assets:
                if not asset.working_usd_path:
                    unresolved_issues.append(
                        f"Processable asset {asset.asset_id} has no extracted USD."
                    )
                    continue
                extracted_path = Path(asset.working_usd_path).expanduser().resolve()
                if not extracted_path.is_file() or extracted_path.stat().st_size == 0:
                    unresolved_issues.append(
                        f"Extracted USD is missing or empty for {asset.asset_id}: "
                        f"{extracted_path}"
                    )

        artifact_paths = [str(request_path), str(manifest_path), str(catalog_path)]
        if material_manifest_path:
            artifact_paths.append(str(material_manifest_path))
        artifact_paths.extend(extracted_asset_paths)
        phase_result = seal_phase_result(
            DecompositionPhaseResult(
                success=not unresolved_issues,
                input_digest=effective_input_digest,
                source_scene=str(source_path),
                source_identity_digest=source_identity_digest,
                manifest_catalog_path=str(catalog_path),
                manifest_paths=[str(manifest_path)],
                extracted_asset_paths=extracted_asset_paths,
                artifact_paths=sorted(set(artifact_paths)),
                completion_policy_satisfied=not unresolved_issues,
                unresolved_issues=unresolved_issues,
                error="; ".join(unresolved_issues) if unresolved_issues else None,
            ),
            phase_result_path,
        )

        return SceneDecompositionResult(
            success=phase_result.success,
            output_dir=str(output_dir),
            manifest_path=str(manifest_path),
            manifest_catalog_path=str(catalog_path),
            material_agent_manifest_path=str(material_manifest_path)
            if material_manifest_path
            else None,
            phase_result_path=str(phase_result_path),
            input_digest=effective_input_digest,
            output_digest=phase_result.output_digest,
            asset_count=len(manifest.assets),
            processable_asset_count=len(manifest.processable_assets),
            instance_group_count=len(manifest.instance_groups),
            payload_group_count=len(manifest.payload_groups),
            prototype_group_count=len(manifest.prototype_groups),
            error=phase_result.error,
        )
    except Exception as exc:
        diagnostic = {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_result = seal_phase_result(
            DecompositionPhaseResult(
                success=False,
                input_digest=effective_input_digest,
                source_scene=str(source_path),
                source_identity_digest=source_identity_digest,
                artifact_paths=[str(request_path)],
                completion_policy_satisfied=False,
                unresolved_issues=[str(exc)],
                diagnostics=[diagnostic],
                error=str(exc),
            ),
            phase_result_path,
        )
        return SceneDecompositionResult(
            success=False,
            output_dir=str(output_dir),
            phase_result_path=str(phase_result_path),
            input_digest=effective_input_digest,
            output_digest=failure_result.output_digest,
            error=str(exc),
        )


decompose_scene = run_scene_decomposition
