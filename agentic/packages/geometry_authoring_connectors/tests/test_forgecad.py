# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from geometry_authoring_connectors import (
    ForgeCadArtifactAdapter,
    ForgeCadArtifactManifest,
    ForgeCadExecutionUnavailableError,
    ForgeCadManifestArtifact,
    InvalidProviderResponseError,
    UnsafeArtifactError,
    UnsupportedCapabilityError,
)

FORGE_SOURCE = b"export default function makePart() { return cube([10, 20, 30]); }\n"
STEP_BYTES = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "part.forge.js"
    step = tmp_path / "part.step"
    source.write_bytes(FORGE_SOURCE)
    step.write_bytes(STEP_BYTES)
    return source, step


def _manifest(source: Path, step: Path) -> ForgeCadArtifactManifest:
    return ForgeCadArtifactManifest(
        provider_version="forgecad-exporter-1",
        source_revision="forgecad-revision-1",
        units="millimeter",
        up_axis="Z",
        rights_assertion="Caller is authorized to process these exported artifacts.",
        artifacts=(
            ForgeCadManifestArtifact(
                filename=source.name,
                role="native_source",
                media_type="text/javascript",
                sha256=hashlib.sha256(FORGE_SOURCE).hexdigest(),
                size_bytes=len(FORGE_SOURCE),
            ),
            ForgeCadManifestArtifact(
                filename=step.name,
                role="cad_geometry",
                media_type="model/step",
                sha256=hashlib.sha256(STEP_BYTES).hexdigest(),
                size_bytes=len(STEP_BYTES),
            ),
        ),
    )


def test_imports_inert_source_export_and_exact_manifest(tmp_path: Path) -> None:
    source, step = _inputs(tmp_path)
    manifest = _manifest(source, step)
    manifest_path = tmp_path / "forgecad-manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    bundle = ForgeCadArtifactAdapter().import_artifacts(
        forge_source=source,
        exports=(step,),
        manifest_path=manifest_path,
        output_dir=tmp_path / "result",
    )

    assert bundle.source_revision == "forgecad-revision-1"
    assert bundle.metadata == {
        "source_system": "forgecad",
        "execution_invoked": False,
        "artifact_only": True,
        "manifest_verified": True,
        "rights_assertion": "Caller is authorized to process these exported artifacts.",
    }
    assert {artifact.filename for artifact in bundle.artifacts} == {
        source.name,
        step.name,
        manifest_path.name,
    }
    assert (tmp_path / "result" / source.name).read_bytes() == FORGE_SOURCE
    assert (tmp_path / "result" / step.name).read_bytes() == STEP_BYTES


def test_source_without_existing_export_returns_typed_unavailable(tmp_path: Path) -> None:
    source, _ = _inputs(tmp_path)
    with pytest.raises(ForgeCadExecutionUnavailableError, match="requires an existing"):
        ForgeCadArtifactAdapter().import_artifacts(
            forge_source=source,
            exports=(),
            output_dir=tmp_path / "result",
        )


def test_adapter_never_claims_generation_revision_or_export() -> None:
    adapter = ForgeCadArtifactAdapter()
    with pytest.raises(ForgeCadExecutionUnavailableError):
        adapter.generate()
    with pytest.raises(ForgeCadExecutionUnavailableError):
        adapter.revise()
    with pytest.raises(ForgeCadExecutionUnavailableError):
        adapter.export()
    assert adapter.capabilities().export is False


def test_rejects_non_forge_native_source_suffix(tmp_path: Path) -> None:
    source = tmp_path / "part.js"
    source.write_bytes(FORGE_SOURCE)
    step = tmp_path / "part.step"
    step.write_bytes(STEP_BYTES)
    with pytest.raises(UnsupportedCapabilityError, match=".forge.js"):
        ForgeCadArtifactAdapter().import_artifacts(
            forge_source=source,
            exports=(step,),
            output_dir=tmp_path / "result",
        )


def test_rejects_unsupported_geometry_format(tmp_path: Path) -> None:
    unsupported = tmp_path / "part.sat"
    unsupported.write_bytes(b"ACIS")
    with pytest.raises(UnsupportedCapabilityError, match=".sat"):
        ForgeCadArtifactAdapter().import_artifacts(
            exports=(unsupported,),
            output_dir=tmp_path / "result",
        )


def test_rejects_malformed_step_before_materialization(tmp_path: Path) -> None:
    malformed = tmp_path / "broken.step"
    malformed.write_bytes(b"not a STEP file")
    with pytest.raises(InvalidProviderResponseError, match="STEP"):
        ForgeCadArtifactAdapter().import_artifacts(
            exports=(malformed,),
            output_dir=tmp_path / "result",
        )
    assert not (tmp_path / "result").exists()


def test_rejects_manifest_digest_substitution(tmp_path: Path) -> None:
    source, step = _inputs(tmp_path)
    manifest_payload = _manifest(source, step).model_dump(mode="json")
    manifest_payload["artifacts"][1]["sha256"] = "0" * 64
    manifest_path = tmp_path / "forgecad-manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
    with pytest.raises(InvalidProviderResponseError, match="binding differs"):
        ForgeCadArtifactAdapter().import_artifacts(
            forge_source=source,
            exports=(step,),
            manifest_path=manifest_path,
            output_dir=tmp_path / "result",
        )


def test_rejects_manifest_inventory_omission(tmp_path: Path) -> None:
    source, step = _inputs(tmp_path)
    manifest_payload = _manifest(source, step).model_copy(
        update={"artifacts": (_manifest(source, step).artifacts[1],)}
    )
    manifest_path = tmp_path / "forgecad-manifest.json"
    manifest_path.write_text(manifest_payload.model_dump_json(), encoding="utf-8")
    with pytest.raises(InvalidProviderResponseError, match="inventory differs"):
        ForgeCadArtifactAdapter().import_artifacts(
            forge_source=source,
            exports=(step,),
            manifest_path=manifest_path,
            output_dir=tmp_path / "result",
        )


def test_rejects_symlinked_input(tmp_path: Path) -> None:
    _, step = _inputs(tmp_path)
    symlink = tmp_path / "alias.step"
    symlink.symlink_to(step)
    with pytest.raises(UnsafeArtifactError, match="symlink"):
        ForgeCadArtifactAdapter().import_artifacts(
            exports=(symlink,),
            output_dir=tmp_path / "result",
        )


def test_production_adapter_contains_no_forgecad_invocation() -> None:
    source = (
        (Path(__file__).parents[1] / "geometry_authoring_connectors" / "forgecad.py")
        .read_text(encoding="utf-8")
        .lower()
    )
    assert "import subprocess" not in source
    assert "import forgecad" not in source
    assert "forgecad install" not in source
    assert "forgecad run" not in source
