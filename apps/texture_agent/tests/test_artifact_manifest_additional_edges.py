# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

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

    preflight = {"verdict": "go"}
    assert (
        am._external_authoring_preflight({"external_authoring_preflight": preflight})
        == preflight
    )
    assert am._external_authoring_preflight({}) is None

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

    invalid_external = am.validate_artifacts_manifest_schema(
        {"backend": {"external_authoring": "bad"}}
    )
    assert "backend.external_authoring must be an object" in invalid_external
    incomplete_external = am.validate_artifacts_manifest_schema(
        {"backend": {"external_authoring": {}}}
    )
    assert "backend.external_authoring.schema_version is required" in (
        incomplete_external
    )
    assert "backend.external_authoring.verdict must be go or no_go" in (
        incomplete_external
    )


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
    shader.CreateInput("normal_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("source.usdz[textures/normal.png]")
    )
    shader.CreateInput("specular_texture", Sdf.ValueTypeNames.String).Set(
        "source.usdz[textures/specular.jpg]"
    )
    shader.CreateInput("roughness_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("source.usdz[textures/roughness.jpeg]")
    )
    stage.GetRootLayer().Save()

    refs = am._output_texture_references(output_usd)
    assert {ref["value_type"] for ref in refs} == {"asset", "string"}
    assert {ref["path"] for ref in refs} == {
        "../textures/missing.png",
        "../textures/opacity.png",
        "source.usdz[textures/normal.png]",
        "source.usdz[textures/roughness.jpeg]",
        "source.usdz[textures/specular.jpg]",
    }

    result = am.validate_output_texture_portability(output_usd)
    assert result["portable"] is False
    assert sorted(result["missing_texture_paths"]) == [
        "../textures/missing.png",
        "../textures/opacity.png",
    ]
    assert sorted(result["non_relative_texture_paths"]) == [
        "source.usdz[textures/normal.png]",
        "source.usdz[textures/roughness.jpeg]",
        "source.usdz[textures/specular.jpg]",
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


def test_portability_checks_string_textures_in_unselected_variants(
    tmp_path: Path,
) -> None:
    """A dormant variant cannot hide a missing String texture dependency."""
    from pxr import Sdf, Usd, UsdShade

    output_usd = tmp_path / "bundle" / "output" / "scene.usda"
    textures_dir = output_usd.parent.parent / "textures"
    output_usd.parent.mkdir(parents=True)
    textures_dir.mkdir(parents=True)
    (textures_dir / "active.png").write_bytes(b"active")

    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("look")
    for name, texture in (
        ("active", "../textures/active.png"),
        ("dormant", "../textures/missing.png"),
    ):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            shader = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput(
                "diffuse_texture",
                Sdf.ValueTypeNames.String,
            ).Set(texture)
    variants.SetVariantSelection("active")
    stage.GetRootLayer().Save()

    refs = am._output_texture_references(output_usd)
    assert refs is not None
    assert {ref["path"] for ref in refs} == {
        "../textures/active.png",
        "../textures/missing.png",
    }
    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=output_usd.parent.parent,
    )
    assert portability["portable"] is False
    assert portability["missing_texture_paths"] == ["../textures/missing.png"]


def test_portability_ignores_variant_opinions_masked_by_stronger_override(
    tmp_path: Path,
) -> None:
    """A dead variant opinion cannot reject a valid generated override."""
    from pxr import Sdf, Usd, UsdShade

    (tmp_path / "generated.png").write_bytes(b"generated")
    output_usd = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    stage.SetDefaultPrim(stage.DefinePrim("/Root", "Xform"))
    material = UsdShade.Material.Define(stage, "/Root/Looks/Paint")
    shader = UsdShade.Shader.Define(stage, "/Root/Looks/Paint/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")

    variants = material.GetPrim().GetVariantSets().AddVariantSet("finish")
    for name, texture in (("a", "missing-A.png"), ("b", "missing-B.png")):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            stage.OverridePrim("/Root/Looks/Paint/Surface").CreateAttribute(
                "inputs:diffuse_texture",
                Sdf.ValueTypeNames.String,
            ).Set(texture)
    variants.SetVariantSelection("a")
    # A direct opinion outranks every variant arc on the same prim.
    shader.GetPrim().CreateAttribute(
        "inputs:diffuse_texture",
        Sdf.ValueTypeNames.String,
    ).Set("generated.png")
    stage.GetRootLayer().Save()

    # Neither selection can surface the obsolete variant values.
    opened = Usd.Stage.Open(str(output_usd))
    variant_set = opened.GetPrimAtPath("/Root/Looks/Paint").GetVariantSet("finish")
    for selection in ("a", "b"):
        variant_set.SetVariantSelection(selection)
        composed = (
            opened.GetPrimAtPath("/Root/Looks/Paint/Surface")
            .GetAttribute("inputs:diffuse_texture")
            .Get()
        )
        assert composed == "generated.png"

    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=tmp_path,
    )
    assert portability["portable"] is True
    assert portability["missing_texture_paths"] == []
    assert portability["diagnostics"] == []


def test_portability_checks_textures_in_dormant_variant_layers(
    tmp_path: Path,
) -> None:
    """A layer referenced only by a dormant variant cannot hide a miss."""
    from pxr import Sdf, Usd, UsdShade

    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "Textures").mkdir()
    (bundle / "Textures" / "ok.png").write_bytes(b"ok")

    for name, texture in (("A", "Textures/ok.png"), ("B", "Textures/missing.png")):
        branch = Usd.Stage.CreateNew(str(bundle / f"{name}.usda"))
        shader = UsdShade.Shader.Define(branch, f"/{name}/Looks/Paint/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(texture)
        branch.SetDefaultPrim(branch.GetPrimAtPath(f"/{name}"))
        branch.GetRootLayer().Save()

    output_usd = bundle / "scene.usda"
    stage = Usd.Stage.CreateNew(str(output_usd))
    root = stage.DefinePrim("/Root", "Xform")
    stage.SetDefaultPrim(root)
    variants = root.GetVariantSets().AddVariantSet("look")
    for name, branch_layer in (("active", "./A.usda"), ("dormant", "./B.usda")):
        variants.AddVariant(name)
        variants.SetVariantSelection(name)
        with variants.GetVariantEditContext():
            model = stage.DefinePrim("/Root/Model", "Xform")
            model.GetReferences().AddReference(branch_layer)
    variants.SetVariantSelection("active")
    stage.GetRootLayer().Save()

    # B.usda contributes nothing to the active composition, so it is absent
    # from the stage's used layers.
    opened = Usd.Stage.Open(str(output_usd))
    used = {
        Path(layer.realPath).name for layer in opened.GetUsedLayers() if layer.realPath
    }
    assert "B.usda" not in used

    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=bundle,
    )
    assert portability["portable"] is False
    assert portability["missing_texture_paths"] == ["Textures/missing.png"]
    assert [item["code"] for item in portability["diagnostics"]] == [
        "PACKAGE_MISSING_ARTIFACT"
    ]


def _build_instanceable_shader_stage(
    tmp_path: Path,
    texture: str,
    *,
    instances: tuple[str, ...],
) -> Path:
    """Author a root layer whose shader is reachable only via instance proxies."""
    from pxr import Sdf, Usd, UsdShade

    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    model = bundle / "Model.usda"
    model_stage = Usd.Stage.CreateNew(str(model))
    shader = UsdShade.Shader.Define(model_stage, "/Model/Looks/Paint/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.String).Set(texture)
    model_stage.SetDefaultPrim(model_stage.GetPrimAtPath("/Model"))
    model_stage.GetRootLayer().Save()

    root_usd = bundle / "scene.usda"
    root_stage = Usd.Stage.CreateNew(str(root_usd))
    root = root_stage.DefinePrim("/Root", "Xform")
    root_stage.SetDefaultPrim(root)
    for name in instances:
        instance = root_stage.DefinePrim(f"/Root/{name}", "Xform")
        instance.GetReferences().AddReference("./Model.usda")
        instance.SetInstanceable(True)
    root_stage.GetRootLayer().Save()
    return root_usd


def test_portability_detects_missing_instance_proxy_string_texture(
    tmp_path: Path,
) -> None:
    """An instanceable reference cannot hide a missing String texture."""
    output_usd = _build_instanceable_shader_stage(
        tmp_path,
        "Textures/missing.png",
        instances=("A",),
    )

    refs = am._output_texture_references(output_usd)
    assert refs is not None
    # A default traversal skips instance proxies and collects nothing here.
    assert [ref["prim_path"] for ref in refs] == ["/Root/A/Looks/Paint/Shader"]

    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=output_usd.parent,
    )
    assert portability["portable"] is False
    assert portability["missing_texture_paths"] == ["Textures/missing.png"]
    assert [item["code"] for item in portability["diagnostics"]] == [
        "PACKAGE_MISSING_ARTIFACT"
    ]


def test_portability_dedupes_shared_instance_proxy_texture_specs(
    tmp_path: Path,
) -> None:
    """Instances sharing one prototype spec report that texture once."""
    output_usd = _build_instanceable_shader_stage(
        tmp_path,
        "Textures/missing.png",
        instances=("A", "B"),
    )

    refs = am._output_texture_references(output_usd)
    assert refs is not None
    assert len(refs) == 1

    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=output_usd.parent,
    )
    assert portability["portable"] is False
    assert portability["texture_reference_count"] == 1
    assert portability["missing_texture_paths"] == ["Textures/missing.png"]


def test_portability_accepts_resolvable_instance_proxy_string_texture(
    tmp_path: Path,
) -> None:
    """A present instance-proxy texture stays portable and raises no diagnostic."""
    output_usd = _build_instanceable_shader_stage(
        tmp_path,
        "Textures/ok.png",
        instances=("A", "B"),
    )
    textures_dir = output_usd.parent / "Textures"
    textures_dir.mkdir(parents=True, exist_ok=True)
    (textures_dir / "ok.png").write_bytes(b"ok")

    portability = am.validate_output_texture_portability(
        output_usd,
        bundle_root=output_usd.parent,
    )
    assert portability["portable"] is True
    assert portability["texture_reference_count"] == 1
    assert portability["diagnostics"] == []


def test_output_string_texture_reference_uses_matching_authoring_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Sdf

    class BrokenSpec:
        @property
        def default(self):
            raise AttributeError("missing default")

    class FakeAttr:
        def __init__(self, name: str, value: str, specs: list[object]) -> None:
            self._name = name
            self._value = value
            self._specs = specs

        def Get(self) -> str:
            return self._value

        def GetName(self) -> str:
            return self._name

        def GetPropertyStack(self) -> list[object]:
            return self._specs

    attrs = [
        FakeAttr(
            "inputs:albedo_texture",
            "albedo.png",
            [
                SimpleNamespace(
                    default=Sdf.AssetPath("albedo.png"),
                    layer=object(),
                )
            ],
        ),
        FakeAttr(
            "inputs:normal_texture",
            "normal.png",
            [
                SimpleNamespace(default="other.png", layer=object()),
                SimpleNamespace(default="normal.png", layer=object()),
            ],
        ),
        FakeAttr(
            "inputs:roughness_texture",
            "roughness.png",
            [BrokenSpec()],
        ),
    ]
    prim = SimpleNamespace(
        IsA=lambda _schema: True,
        GetAttributes=lambda: attrs,
        GetPath=lambda: "/Root/Looks/Shader",
        IsInstanceProxy=lambda: False,
    )
    stage = SimpleNamespace(Traverse=lambda _predicate=None: [prim])
    monkeypatch.setattr(am.Usd.Stage, "Open", staticmethod(lambda _path: stage))
    monkeypatch.setattr(
        am.Sdf,
        "ComputeAssetPathRelativeToLayer",
        lambda _layer, path: f"/resolved/{path}",
    )

    refs = am._output_texture_references(Path("scene.usda"))

    assert refs == [
        {
            "prim_path": "/Root/Looks/Shader",
            "attribute": "inputs:albedo_texture",
            "value_type": "string",
            "path": "albedo.png",
            "resolved_path": "/resolved/albedo.png",
        },
        {
            "prim_path": "/Root/Looks/Shader",
            "attribute": "inputs:normal_texture",
            "value_type": "string",
            "path": "normal.png",
            "resolved_path": "/resolved/normal.png",
        },
        {
            "prim_path": "/Root/Looks/Shader",
            "attribute": "inputs:roughness_texture",
            "value_type": "string",
            "path": "roughness.png",
            "resolved_path": "",
        },
    ]
