# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

from texture_agent.functions import artifact_manifest as am
from texture_agent.functions.texture_generation import GeneratedTextures


@dataclass
class _Thing:
    path: Path
    token: str


def test_manifest_write_uses_atomic_sibling_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = {"working_dir": str(tmp_path)}
    replacements: list[tuple[Path, Path]] = []
    real_replace = am.os.replace

    def _record_replace(source: str | Path, target: str | Path) -> None:
        replacements.append((Path(source), Path(target)))
        real_replace(source, target)

    monkeypatch.setattr(am.os, "replace", _record_replace)

    manifest_path = am.write_artifacts_manifest(
        context,
        payload={"schema_version": am.ARTIFACTS_MANIFEST_SCHEMA_VERSION},
    )

    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "schema_version": am.ARTIFACTS_MANIFEST_SCHEMA_VERSION
    }
    assert len(replacements) == 1
    source, target = replacements[0]
    assert source.parent == target.parent == tmp_path
    assert source.name.startswith(".artifacts_manifest.json.")
    assert target == manifest_path
    assert not source.exists()


def test_manifest_private_path_redaction_and_json_edges(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()
    png = root / "image.png"
    Image.new("RGB", (2, 3), (1, 2, 3)).save(png)
    text = root / "not_image.png"
    text.write_text("not an image", encoding="utf-8")

    assert am._display_path(None, root) is None
    assert am._display_path("", root) == ""
    assert am._display_path("https://x.test/a.png?token=SECRET", root) == (
        "https://x.test/a.png?token=<redacted>"
    )
    assert am._path_entry(None, root) is None
    assert am._path_entry("s3://bucket/key.png", root)["exists"] is False
    assert am._image_info(None, root) is None

    image_info = am._image_info(png, root)
    assert image_info["width"] == 2
    assert image_info["height"] == 3
    assert image_info["nonblank"] is True
    assert "open_error" in am._image_info(text, root)

    payload = am._jsonable(
        {
            "dataclass": _Thing(Path("a"), "b"),
            "set": {"x"},
            "object": object(),
        }
    )
    assert payload["dataclass"]["path"] == "a"
    assert payload["set"] == ["x"]
    assert isinstance(payload["object"], str)

    redacted = am.redact_sensitive(
        {
            "path": Path("secret.txt"),
            "endpoint": "https://configured",
            "tuple": ("Bearer nvapi-FAKESECRET12345678",),
            "nested": {"password": "secret"},
        }
    )
    assert redacted["endpoint"] == "<configured>"
    assert redacted["nested"]["password"] == "<redacted>"
    assert redacted["tuple"] == ["Bearer <redacted>"]

    assert am._read_uv_report(None) is None
    bad_json = root / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    assert am._read_uv_report(str(bad_json)) is None
    list_json = root / "list.json"
    list_json.write_text("[]", encoding="utf-8")
    assert am._read_uv_report(str(list_json)) is None

    entry = am._artifact_map_entry("file:///tmp/a.png", root)
    assert entry["uri"] == "file:///tmp/a.png"
    assert entry["path"]["exists"] is False


def test_manifest_path_entry_handles_os_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    with monkeypatch.context() as scoped:
        scoped.setattr(
            am.os.path,
            "relpath",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("bad path")),
        )
        assert am._display_path(tmp_path / "image.png", tmp_path) == str(
            tmp_path / "image.png"
        )

    def raise_os_error(self: Path) -> bool:
        raise OSError("cannot stat")

    monkeypatch.setattr(Path, "exists", raise_os_error)

    entry = am._path_entry(tmp_path / "image.png", tmp_path)

    assert entry["exists"] is False


def test_projection_state_and_schema_error_edges(tmp_path: Path) -> None:
    generated = GeneratedTextures(albedo="", normal="normal.png", orm="orm.png")
    state = am._projection_channel_state(
        {
            "maps": {"roughness": {"uri": "r.png"}},
            "degraded_channels": ["mask"],
            "diagnostics": [
                {"details": {"missing_maps": ["normal", "orm", "metalness"]}},
                "malformed",
            ],
        },
        generated,
    )
    assert state["albedo"] == "missing"
    assert state["normal"] == "synthesized_neutral"
    assert state["orm"] == "packed_from_channels_or_constants"
    assert state["mask"] == "absent"

    entries = am._projection_backend_entries(
        {
            "generated_textures": {"Steel": generated},
            "projection_backend_results": {
                "bad": "not-a-record",
                "Steel": {
                    "maps": {"roughness": "file:///roughness.png"},
                    "metadata": {"degraded_channels": ["normal"]},
                    "diagnostics": [{"severity": "warning", "code": "W"}],
                    "variant_asset_uri": "s3://bucket/variant.usd",
                },
            },
        },
        tmp_path,
    )
    assert set(entries) == {"Steel"}
    assert entries["Steel"]["warnings"][0]["code"] == "W"
    assert entries["Steel"]["variant_asset"]["exists"] is False

    assert am._projection_warning_entries(
        {"diagnostics": ["bad", {"severity": "info"}, {"severity": "warning"}]}
    ) == [{"severity": "warning"}]
    summary = am._projection_backend_summary(
        {"projection_backend_results": {"bad": "record", "Steel": {"metadata": {}}}}
    )
    assert summary["unit_count"] == 2
    assert set(summary["metadata"]) == {"Steel"}

    errors = am.validate_artifacts_manifest_schema(
        {
            "schema_version": "wrong",
            "outputs": {"portability": {}},
            "textures": {
                "generated": {},
                "blended": [],
                "projection_backend": {
                    "Steel": {"channel_state": {}},
                    "Bad": "not-an-object",
                },
            },
            "backend": {"projection": {}},
            "status": {"diagnostics": ["bad", {"schema_version": "wrong"}]},
        }
    )
    assert "schema_version must be texture-agent-artifacts.v1" in errors
    assert "textures.blended must be present" in errors
    assert "textures.projection_backend.Bad must be an object" in errors
    assert any(
        "textures.projection_backend.Steel.maps is required" in e for e in errors
    )
    assert any("status.diagnostics[0] must be an object" in e for e in errors)
    assert any("status.diagnostics[1].code is required" in e for e in errors)


def test_output_texture_references_and_portability_edges(tmp_path: Path) -> None:
    from pxr import Sdf, Usd, UsdShade

    output_usd = tmp_path / "bundle" / "output" / "scene.usda"
    output_usd.parent.mkdir(parents=True)
    stage = Usd.Stage.CreateNew(str(output_usd))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("../textures/missing.png"))
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Steel/Shader")
    shader.CreateInput("opacity_texture", Sdf.ValueTypeNames.String).Set(
        "../textures/opacity.png"
    )
    stage.GetRootLayer().Save()

    refs = am._output_texture_references(output_usd)
    assert {ref["value_type"] for ref in refs} == {"asset", "string"}

    result = am.validate_output_texture_portability(output_usd)
    assert result["portable"] is False
    assert sorted(result["missing_texture_paths"]) == [
        "../textures/missing.png",
        "../textures/opacity.png",
    ]

    absolute_usd = tmp_path / "bundle" / "output" / "absolute.usda"
    stage = Usd.Stage.CreateNew(str(absolute_usd))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Absolute")
    material.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file",
        Sdf.ValueTypeNames.Asset,
    ).Set(Sdf.AssetPath("s3://bucket/texture.png"))
    stage.GetRootLayer().Save()
    absolute_result = am.validate_output_texture_portability(absolute_usd)
    assert absolute_result["non_relative_texture_paths"] == ["s3://bucket/texture.png"]

    assert (
        am.validate_output_texture_portability(tmp_path / "missing.usda")[
            "diagnostics"
        ][0]["code"]
        == "PACKAGE_MISSING_ARTIFACT"
    )

    bad_manifest = tmp_path / "bad.usda"
    bad_manifest.write_text("not usd", encoding="utf-8")
    assert am.validate_output_texture_portability(bad_manifest)["portable"] is False

    original_open = Usd.Stage.Open
    try:
        Usd.Stage.Open = staticmethod(lambda *args, **kwargs: None)
        assert am._output_texture_references(output_usd) is None
    finally:
        Usd.Stage.Open = original_open

    deduped = am._dedupe_diagnostics([{"a": 1}, {"a": 1}, {"a": 2}])
    assert deduped == [{"a": 1}, {"a": 2}]
