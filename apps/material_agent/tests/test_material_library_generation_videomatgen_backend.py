# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""WP8 tests for the VideoMatGen material-creation adapter."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from material_agent.material_library_generation import (
    VIDEOMATGEN_ADAPTER_REVISION,
    VIDEOMATGEN_BACKEND_NAME,
    VIDEOMATGEN_UNAVAILABLE_REVISION,
    CreateMaterialRequest,
    IntendedPart,
    MaterialColorSpace,
    MaterialConditioningArtifact,
    MaterialConditioningKind,
    MaterialCreationError,
    MaterialCreationErrorCode,
    MaterialCreationMode,
    MaterialRecipe,
    PBRHints,
    PreparedMaterialConditioning,
    VideoMatGenBackendConfig,
    VideoMatGenMaterialCreationBackend,
)


def _source_usd(tmp_path: Path) -> Path:
    source = tmp_path / "asset.usda"
    source.write_text("#usda 1.0\n", encoding="utf-8")
    return source


def _recipe() -> MaterialRecipe:
    return MaterialRecipe(
        id="wp8_painted_metal",
        name="WP8 Painted Metal",
        description="Satin red painted metal for a machine housing.",
        appearance_prompt="satin red painted metal with subtle edge wear",
        color="red",
        material="painted metal",
        finish="satin",
        pbr_hints=PBRHints(metallic=0.6, roughness=0.38),
        intended_parts=(
            IntendedPart(
                semantic_label="housing",
                evidence="Planner selected the painted housing.",
                prim_path_hints=("/World/Asset/Housing",),
            ),
        ),
    )


def _request(
    tmp_path: Path,
    *,
    creation_mode: MaterialCreationMode = MaterialCreationMode.ASSET_UV,
) -> CreateMaterialRequest:
    return CreateMaterialRequest(
        source_usd=_source_usd(tmp_path),
        target_prim_paths=("/World/Asset/Housing",),
        recipe=_recipe(),
        creation_mode=creation_mode,
        texture_size=64,
        backend=VIDEOMATGEN_BACKEND_NAME,
        seed=8,
        source_usd_sha256="0" * 64,
    )


def _conditioning(
    request: CreateMaterialRequest,
    tmp_path: Path,
    *,
    complete: bool = True,
) -> PreparedMaterialConditioning:
    artifacts = [
        MaterialConditioningArtifact(
            kind=MaterialConditioningKind.SCOPED_USD,
            uri=(tmp_path / "conditioning" / "scoped.usda").as_posix(),
            sha256="1" * 64,
        )
    ]
    if complete:
        artifacts.extend(
            (
                MaterialConditioningArtifact(
                    kind=MaterialConditioningKind.UV_LAYOUT,
                    uri=(tmp_path / "conditioning" / "uv_layout.png").as_posix(),
                    color_space=MaterialColorSpace.RAW,
                    sha256="2" * 64,
                ),
                MaterialConditioningArtifact(
                    kind=MaterialConditioningKind.UV_MASK,
                    uri=(tmp_path / "conditioning" / "uv_mask.png").as_posix(),
                    color_space=MaterialColorSpace.RAW,
                    sha256="3" * 64,
                ),
                MaterialConditioningArtifact(
                    kind=MaterialConditioningKind.NORMAL,
                    uri=(tmp_path / "conditioning" / "normal.png").as_posix(),
                    color_space=MaterialColorSpace.RAW,
                    sha256="4" * 64,
                ),
                MaterialConditioningArtifact(
                    kind=MaterialConditioningKind.DEPTH,
                    uri=(tmp_path / "conditioning" / "depth.png").as_posix(),
                    color_space=MaterialColorSpace.RAW,
                    sha256="5" * 64,
                ),
                MaterialConditioningArtifact(
                    kind=MaterialConditioningKind.RENDER,
                    uri=(tmp_path / "conditioning" / "render.png").as_posix(),
                    color_space=MaterialColorSpace.SRGB,
                    sha256="6" * 64,
                ),
            )
        )
    return PreparedMaterialConditioning.for_request(
        request,
        artifacts=tuple(artifacts),
    )


def _create_and_capture(
    backend: VideoMatGenMaterialCreationBackend,
    request: CreateMaterialRequest,
    tmp_path: Path,
    conditioning: PreparedMaterialConditioning | None,
) -> MaterialCreationError:
    with pytest.raises(MaterialCreationError) as error_info:
        backend.create(
            request,
            output_dir=tmp_path / "package",
            conditioning=conditioning,
        )
    return error_info.value


def test_config_normalizes_optional_paths_and_metadata(tmp_path: Path) -> None:
    config = VideoMatGenBackendConfig(
        source_root=str(tmp_path / "source"),
        entrypoint=tmp_path / "source" / "infer.py",
        checkpoint_path=str(tmp_path / "model.ckpt"),
        source_revision="  abc123  ",
        rights_evidence_uri="  evidence://videomatgen-rights  ",
        model_revision="  videomatgen-model-v1  ",
        min_gpu_count=4,
    )
    blank_config = VideoMatGenBackendConfig(
        source_root="",
        entrypoint="   ",
        checkpoint_path="",
        source_revision="   ",
        rights_evidence_uri="",
        model_revision=None,
    )

    assert config.source_root == tmp_path / "source"
    assert config.entrypoint == tmp_path / "source" / "infer.py"
    assert config.checkpoint_path == tmp_path / "model.ckpt"
    assert config.source_revision == "abc123"
    assert config.rights_evidence_uri == "evidence://videomatgen-rights"
    assert config.effective_model_revision == "videomatgen-model-v1"
    assert blank_config.source_root is None
    assert blank_config.entrypoint is None
    assert blank_config.checkpoint_path is None
    assert blank_config.source_revision is None
    assert blank_config.rights_evidence_uri is None
    assert blank_config.model_revision is None
    assert blank_config.effective_model_revision == VIDEOMATGEN_UNAVAILABLE_REVISION
    with pytest.raises(ValueError, match="min_gpu_count"):
        VideoMatGenBackendConfig(min_gpu_count=0)


def test_backend_metadata_is_stable() -> None:
    backend = VideoMatGenMaterialCreationBackend()

    assert backend.name == VIDEOMATGEN_BACKEND_NAME
    assert backend.revision == VIDEOMATGEN_ADAPTER_REVISION


def test_backend_reports_wp8_hard_blockers_when_runtime_is_unconfigured(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend()

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert error.backend == VIDEOMATGEN_BACKEND_NAME
    assert error.retryable is False
    assert {diagnostic.code for diagnostic in error.diagnostics} == {
        "VIDEOMATGEN_SOURCE_UNCONFIGURED",
        "VIDEOMATGEN_ENTRYPOINT_UNCONFIGURED",
        "VIDEOMATGEN_SOURCE_REVISION_UNPINNED",
        "VIDEOMATGEN_MODEL_REVISION_UNPINNED",
        "VIDEOMATGEN_CHECKPOINT_UNCONFIGURED",
        "VIDEOMATGEN_RIGHTS_UNVERIFIED",
    }
    assert error.to_dict()["diagnostics"][0]["severity"] == "error"


def test_backend_reports_missing_configured_source_and_entrypoint(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend(
        VideoMatGenBackendConfig(
            source_root=tmp_path / "missing-source",
            entrypoint=tmp_path / "missing-source" / "infer.py",
            checkpoint_path=tmp_path / "missing.ckpt",
            source_revision="abc123",
            rights_evidence_uri="evidence://videomatgen-rights",
            model_revision="videomatgen-model-v1",
        )
    )

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert {diagnostic.code for diagnostic in error.diagnostics} == {
        "VIDEOMATGEN_SOURCE_MISSING",
        "VIDEOMATGEN_ENTRYPOINT_MISSING",
        "VIDEOMATGEN_CHECKPOINT_MISSING",
    }


def test_backend_rejects_wrong_runtime_path_kinds(tmp_path: Path) -> None:
    source_root = tmp_path / "videomatgen-source-file"
    source_root.write_text("not a directory\n", encoding="utf-8")
    entrypoint = tmp_path / "entrypoint-dir"
    entrypoint.mkdir()
    checkpoint_path = tmp_path / "checkpoint-dir"
    checkpoint_path.mkdir()
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend(
        VideoMatGenBackendConfig(
            source_root=source_root,
            entrypoint=entrypoint,
            checkpoint_path=checkpoint_path,
            source_revision="abc123",
            rights_evidence_uri="evidence://videomatgen-rights",
            model_revision="videomatgen-model-v1",
        )
    )

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_SOURCE_MISSING",
        "VIDEOMATGEN_ENTRYPOINT_MISSING",
        "VIDEOMATGEN_CHECKPOINT_MISSING",
    ]
    assert "not a directory" in error.diagnostics[0].message
    assert "not a file" in error.diagnostics[1].message
    assert "not a file" in error.diagnostics[2].message


def test_backend_uses_missing_checkpoint_code_for_checkpoint_only_blocker(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "videomatgen"
    source_root.mkdir()
    entrypoint = source_root / "infer.py"
    entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend(
        VideoMatGenBackendConfig(
            source_root=source_root,
            entrypoint=entrypoint,
            checkpoint_path=tmp_path / "missing.ckpt",
            source_revision="abc123",
            rights_evidence_uri="evidence://videomatgen-rights",
            model_revision="videomatgen-model-v1",
        )
    )

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.MISSING_CHECKPOINT
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_CHECKPOINT_MISSING"
    ]


def test_backend_reports_unpinned_model_revision(tmp_path: Path) -> None:
    source_root = tmp_path / "videomatgen"
    source_root.mkdir()
    entrypoint = source_root / "infer.py"
    entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend(
        VideoMatGenBackendConfig(
            source_root=source_root,
            entrypoint=entrypoint,
            checkpoint_path=checkpoint_path,
            source_revision="abc123",
            rights_evidence_uri="evidence://videomatgen-rights",
            model_revision=None,
        )
    )

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_MODEL_REVISION_UNPINNED"
    ]


def test_backend_reports_unavailable_runtime_mapping_when_probe_is_clean(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "videomatgen"
    source_root.mkdir()
    entrypoint = source_root / "infer.py"
    entrypoint.write_text("raise SystemExit(0)\n", encoding="utf-8")
    checkpoint_path = tmp_path / "model.ckpt"
    checkpoint_path.write_bytes(b"checkpoint")
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend(
        VideoMatGenBackendConfig(
            source_root=source_root,
            entrypoint=entrypoint,
            checkpoint_path=checkpoint_path,
            source_revision="abc123",
            rights_evidence_uri="evidence://videomatgen-rights",
            model_revision="videomatgen-model-v1",
            min_gpu_count=4,
        )
    )

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.BACKEND_UNAVAILABLE
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_RUNTIME_CONTRACT_UNAVAILABLE"
    ]
    assert error.diagnostics[0].details["source_revision"] == "abc123"
    assert error.diagnostics[0].details["model_revision"] == "videomatgen-model-v1"
    assert error.diagnostics[0].details["min_gpu_count"] == "4"


def test_backend_rejects_tileable_requests_before_runtime_probe(tmp_path: Path) -> None:
    request = _request(tmp_path, creation_mode=MaterialCreationMode.TILEABLE)
    backend = VideoMatGenMaterialCreationBackend()

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path),
    )

    assert error.code is MaterialCreationErrorCode.UNSUPPORTED_MATERIAL
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_TILEABLE_UNSUPPORTED"
    ]
    assert error.diagnostics[0].details == {"creation_mode": "tileable"}


def test_backend_requires_wp4_conditioning(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend()

    error = _create_and_capture(backend, request, tmp_path, None)

    assert error.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_CONDITIONING_REQUIRED"
    ]


def test_backend_wraps_conditioning_scope_mismatch(tmp_path: Path) -> None:
    request = _request(tmp_path)
    mismatched = replace(
        _conditioning(request, tmp_path),
        target_prim_paths=("/World/Asset/Other",),
    )
    backend = VideoMatGenMaterialCreationBackend()

    error = _create_and_capture(backend, request, tmp_path, mismatched)

    assert error.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_CONDITIONING_MISMATCH"
    ]


def test_backend_requires_complete_conditioning_kinds(tmp_path: Path) -> None:
    request = _request(tmp_path)
    backend = VideoMatGenMaterialCreationBackend()

    error = _create_and_capture(
        backend,
        request,
        tmp_path,
        _conditioning(request, tmp_path, complete=False),
    )

    assert error.code is MaterialCreationErrorCode.INVALID_REQUEST
    assert [diagnostic.code for diagnostic in error.diagnostics] == [
        "VIDEOMATGEN_CONDITIONING_INCOMPLETE"
    ]
    assert error.diagnostics[0].details["missing_kinds"] == (
        "uv_layout,uv_mask,normal,depth,render"
    )


def test_backend_honors_cancellation_before_runtime_probe(tmp_path: Path) -> None:
    request = _request(tmp_path)
    cancel_event = threading.Event()
    cancel_event.set()
    backend = VideoMatGenMaterialCreationBackend()

    with pytest.raises(MaterialCreationError) as error_info:
        backend.create(
            request,
            output_dir=tmp_path / "package",
            conditioning=_conditioning(request, tmp_path),
            cancel_event=cancel_event,
        )

    assert error_info.value.code is MaterialCreationErrorCode.CANCELLED
    assert error_info.value.diagnostics == ()
