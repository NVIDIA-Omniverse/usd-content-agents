# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused edge coverage for layered USDZ reconstruction and packaging."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from ...service.workers.executor import _package_usdz, _prepare_source_usdz_stage


def _layered_paths(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    session_dir = tmp_path / "session"
    input_dir = session_dir / "input"
    prepared_dir = session_dir / "cache" / "prepared"
    output_dir = session_dir / "cache" / "output"
    input_dir.mkdir(parents=True)
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return session_dir, input_dir, prepared_dir, output_dir


def _write_instanced_uv_usdz(
    tmp_path: Path,
    destination: Path,
    *,
    instance_names: tuple[str, ...],
) -> Path:
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    root_path = tmp_path / f"{destination.stem}-root.usda"
    stage = Usd.Stage.CreateNew(str(root_path))
    stage.CreateClassPrim("/Body/Prototypes")
    UsdGeom.Xform.Define(stage, "/Body/Prototypes/Body")

    def _tiny_quad(path: str) -> None:
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(
            [
                Gf.Vec3f(0, 0, 0),
                Gf.Vec3f(1, 0, 0),
                Gf.Vec3f(1, 1, 0),
                Gf.Vec3f(0, 1, 0),
            ]
        )
        mesh.CreateFaceVertexCountsAttr([4])
        mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdGeom.PrimvarsAPI(mesh.GetPrim()).CreatePrimvar(
            "st",
            Sdf.ValueTypeNames.TexCoord2fArray,
            UsdGeom.Tokens.vertex,
        ).Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.5, 0.5),
                    Gf.Vec2f(0.5015, 0.5),
                    Gf.Vec2f(0.5015, 0.5015),
                    Gf.Vec2f(0.5, 0.5015),
                ]
            )
        )

    _tiny_quad("/Body/Prototypes/Body/Part/Mesh")
    _tiny_quad("/Body/Unselected/Mesh")
    for name in instance_names:
        instance = UsdGeom.Xform.Define(stage, f"/Body/{name}").GetPrim()
        instance.GetReferences().AddInternalReference("/Body/Prototypes/Body")
        instance.SetInstanceable(True)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Body"))
    assert stage.GetRootLayer().Save()
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as package:
        package.write(root_path, "Scene/root.usda")
    return destination


def _uv_u_span(stage: Any, path: str) -> float:
    from pxr import UsdGeom

    values = UsdGeom.PrimvarsAPI(stage.GetPrimAtPath(path)).GetPrimvar("st").Get()
    return max(value[0] for value in values) - min(value[0] for value in values)


@pytest.mark.parametrize(
    "extra_flatten_count",
    [0, 1, 2],
    ids=["prepared", "second", "third"],
)
def test_instanced_uv_delta_reconstructs_and_packages_without_synthetic_roots(
    tmp_path: Path,
    extra_flatten_count: int,
) -> None:
    """A scoped synthetic-prototype UV edit returns to its source class spec."""
    pytest.importorskip("pxr")
    from pxr import Usd
    from texture_agent.tasks.prepare_uvs import PrepareUVsTask

    session_dir, input_dir, _prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = _write_instanced_uv_usdz(
        tmp_path,
        input_dir / "instanced.usdz",
        instance_names=("Body",),
    )
    context: dict[str, Any] = {
        "usd_path": str(source_usdz),
        "working_dir": str(session_dir / "cache"),
        "texture_config": {
            "uv_policy": "preserve_or_fix",
            "uv_scope": "target_prims",
            "uv_target_prim_paths": ["/Body/Body/Part/Mesh"],
        },
    }
    PrepareUVsTask().run(context)
    prepared_path = Path(context["usd_path"])
    assert context["uv_preparation"]["degenerate_repaired"] == 1

    output_path = prepared_path
    retained_stages_and_layers = []
    for generation in range(1, extra_flatten_count + 1):
        stage = Usd.Stage.Open(str(output_path))
        flattened_layer = stage.Flatten(addSourceFileComment=False)
        retained_stages_and_layers.append((stage, flattened_layer))
        output_path = output_dir / f"textured_output_f{generation + 1}.usda"
        assert flattened_layer.Export(str(output_path))
    context["output_usd_paths"] = [str(output_path)]

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert _uv_u_span(reconstructed, "/Body/Body/Part/Mesh") == pytest.approx(0.95)
    assert _uv_u_span(reconstructed, "/Body/Unselected/Mesh") == pytest.approx(
        0.0015,
        abs=1e-6,
    )
    assert not any(
        spec.name.startswith("Flattened_Prototype_")
        for spec in reconstructed.GetRootLayer().rootPrims
    )

    packaged_path = _package_usdz(context, session_dir)
    assert packaged_path is not None, context.get("usdz_packaging_error")
    packaged = Usd.Stage.Open(packaged_path)
    assert packaged.GetPrimAtPath("/Body/Body").IsInstanceable()
    assert _uv_u_span(packaged, "/Body/Body/Part/Mesh") == pytest.approx(0.95)
    assert _uv_u_span(packaged, "/Body/Unselected/Mesh") == pytest.approx(
        0.0015,
        abs=1e-6,
    )
    assert not any(
        spec.name.startswith("Flattened_Prototype_")
        for spec in packaged.GetRootLayer().rootPrims
    )
    with zipfile.ZipFile(packaged_path) as package:
        authored_text = b"\n".join(
            package.read(member)
            for member in package.namelist()
            if member.endswith((".usd", ".usda"))
        )
    assert b"Flattened_Prototype_" not in authored_text


@pytest.mark.parametrize("edited", [False, True], ids=["untouched", "touched"])
def test_shared_flattened_prototype_rejects_only_a_real_delta(
    tmp_path: Path,
    edited: bool,
) -> None:
    """Shared synthetic backing specs remain safe until an edit targets them."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, Vt

    session_dir, input_dir, prepared_dir, _output_dir = _layered_paths(tmp_path)
    source_usdz = _write_instanced_uv_usdz(
        tmp_path,
        input_dir / "shared.usdz",
        instance_names=("A", "B"),
    )
    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_path = prepared_dir / "prepared.usda"
    assert source_stage.Flatten().Export(str(prepared_path))
    if edited:
        prepared_stage = Usd.Stage.Open(str(prepared_path))
        proxy = prepared_stage.GetPrimAtPath("/Body/A/Part/Mesh")
        backing_paths = {
            spec.path
            for spec in proxy.GetPrimStack()
            if spec.layer == prepared_stage.GetRootLayer()
            and spec.path != proxy.GetPath()
        }
        assert len(backing_paths) == 1
        backing = prepared_stage.GetPrimAtPath(next(iter(backing_paths)))
        UsdGeom.PrimvarsAPI(backing).GetPrimvar("st").Set(
            Vt.Vec2fArray(
                [
                    Gf.Vec2f(0.1, 0.1),
                    Gf.Vec2f(0.9, 0.1),
                    Gf.Vec2f(0.9, 0.9),
                    Gf.Vec2f(0.1, 0.9),
                ]
            )
        )
        prepared_stage.GetRootLayer().Save()

    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_path),
        "output_usd_paths": [str(prepared_path)],
    }
    staged_root = _prepare_source_usdz_stage(context, session_dir)

    if edited:
        assert staged_root is None
        assert "multiple composed paths" in context["usdz_packaging_error"]
    else:
        assert staged_root is not None, context.get("usdz_packaging_error")
        assert "usdz_packaging_error" not in context


def test_layered_package_rejects_absolute_authored_dependency(
    tmp_path: Path,
) -> None:
    """A worker-local absolute reference must never escape in the USDZ."""
    pytest.importorskip("pxr")
    from pxr import Sdf

    session_dir, input_dir, _prepared_dir, output_dir = _layered_paths(tmp_path)
    root_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "Referenced" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
"""
    source_usdz = input_dir / "absolute-reference.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr(
            "Scene/Model/model.usda",
            '#usda 1.0\ndef Xform "Model" {}\n',
        )
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_text, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }
    staged_root = _prepare_source_usdz_stage(context, session_dir)
    assert staged_root is not None
    extract_root = Path(context["source_usdz_extract_root"])
    model_path = extract_root / "Scene" / "Model" / "model.usda"
    root_layer = Sdf.Layer.FindOrOpen(str(staged_root))
    assert root_layer is not None
    referenced = root_layer.GetPrimAtPath("/Root/Referenced")
    assert referenced is not None
    referenced.referenceList.prependedItems = [Sdf.Reference(str(model_path), "/Model")]
    root_layer.Save()

    assert _package_usdz(context, session_dir) is None
    # The failure manifest must not claim the package is portable.
    portability = context["usdz_source_portability"]
    assert portability["portable"] is False
    assert "PACKAGE_NON_RELATIVE_PATH" in [
        item["code"] for item in portability["diagnostics"]
    ]
    assert context["usdz_absolute_dependency_count"] == 1
    assert context["usdz_absolute_dependencies"] == [
        {
            "layer": str(staged_root),
            "path": str(model_path),
        }
    ]
    assert "absolute authored dependencies" in context["usdz_packaging_error"]


_DELETION_ROOT_USDA = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
                float inputs:roughness = 0.5
            }
        }
    }
}
"""


_WEAKER_DELETION_ROOT_USDA = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World" (
    prepend references = @Model/model.usda@</World>
)
{
    over "Looks"
    {
        over "Paint"
        {
            over "Preview"
            {
                float inputs:roughness = 0.5
            }
        }
    }
}
"""


_WEAKER_DELETION_MODEL_USDA = """#usda 1.0
def Xform "World"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
                float inputs:roughness = 0.2
            }
        }
    }
}
"""


_SHARED_MODEL_USDA = """#usda 1.0
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint"
        {
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.2, 0.3, 0.4)
            }
        }
    }
}
"""


def test_layered_delta_rejects_source_spec_shared_with_dormant_variant(
    tmp_path: Path,
) -> None:
    """A dormant branch sharing a source layer must block the edit."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    # /World/A composes now; /World/B references the same model only under the
    # unselected "double" branch, so the active composition sees one user.
    root_text = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World" (
    variantSets = "layout"
    variants = {
        string layout = "single"
    }
)
{
    def Xform "A" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    variantSet "layout" = {
        "single" {
        }
        "double" {
            def Xform "B" (
                prepend references = @Model/model.usda@</Model>
            )
            {
            }
        }
    }
}
"""
    source_usdz = input_dir / "dormant-shared.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr("Scene/Model/model.usda", _SHARED_MODEL_USDA)

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/A/Looks/Paint/Preview"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.1, 0.1))
    edited_stage.GetRootLayer().Save()

    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _package_usdz(context, session_dir) is None
    assert "more than one composition arc" in context["usdz_packaging_error"]


def test_layered_delta_rejects_internal_dormant_alias_to_sublayer_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal arcs share the full root+sublayer stack, including variants."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    root_text = """#usda 1.0
(
    defaultPrim = "World"
    subLayers = [@Library/prototypes.usda@, @Library/branch.usda@]
)
def Xform "World" (
    variantSets = "layout"
    variants = {
        string layout = "single"
    }
)
{
    def Xform "A" (
        prepend references = </Library/Model>
    )
    {
    }
    def Xform "UsesDefault" (
        prepend references = @Deps/with_default.usda@
    )
    {
    }
    def Xform "MissingDefault" (
        prepend references = @Deps/without_default.usda@
    )
    {
    }
    variantSet "layout" = {
        "single" {
        }
        "double" {
            def Xform "B" (
                prepend references = </Library/Model>
            )
            {
            }
            def Scope "ArcAlias" (
                prepend inherits = </Library/Model>
                prepend specializes = </Library/Model>
            )
            {
            }
        }
    }
}
"""
    sublayer_text = (
        _SHARED_MODEL_USDA.replace(
            'def Xform "Model"',
            'class "Library"\n{\n    def Xform "Model"',
        ).rstrip()
        + "\n}\n"
    )
    source_usdz = input_dir / "dormant-internal-sublayer.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr("Scene/Library/prototypes.usda", sublayer_text)
        package.writestr(
            "Scene/Library/branch.usda",
            "#usda 1.0\n(\n    subLayers = [@prototypes.usda@]\n)\n",
        )
        package.writestr(
            "Scene/Deps/with_default.usda",
            '#usda 1.0\n(\n    defaultPrim = "Target"\n)\ndef Xform "Target" {}\n',
        )
        package.writestr(
            "Scene/Deps/without_default.usda",
            '#usda 1.0\ndef Xform "NoDefault" {}\n',
        )

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/A/Looks/Paint/Preview"))
    assert shader
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.1, 0.1))
    edited_stage.GetRootLayer().Save()

    list_info_keys = Sdf.PrimSpec.ListInfoKeys
    get_info = Sdf.PrimSpec.GetInfo

    def _list_info_keys_with_relative_path(spec: Any) -> list[str]:
        keys = list(list_info_keys(spec))
        if spec.path == Sdf.Path("/World"):
            keys.append("_coverageRelativePath")
        return keys

    def _get_info_with_relative_path(spec: Any, key: str) -> Any:
        if spec.path == Sdf.Path("/World") and key == "_coverageRelativePath":
            return Sdf.Path("Child")
        return get_info(spec, key)

    monkeypatch.setattr(
        Sdf.PrimSpec, "ListInfoKeys", _list_info_keys_with_relative_path
    )
    monkeypatch.setattr(Sdf.PrimSpec, "GetInfo", _get_info_with_relative_path)
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    assert _package_usdz(context, session_dir) is None
    assert "more than one composition arc" in context["usdz_packaging_error"]


def test_layered_delta_allows_single_referenced_source_spec(
    tmp_path: Path,
) -> None:
    """One reference to a model layer still packages the edit."""
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    root_text = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World"
{
    def Xform "A" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
"""
    source_usdz = input_dir / "single-reference.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr("Scene/Model/model.usda", _SHARED_MODEL_USDA)

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/A/Looks/Paint/Preview"))
    shader.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.1, 0.1))
    edited_stage.GetRootLayer().Save()

    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    packaged = _package_usdz(context, session_dir)
    assert packaged is not None
    assert context.get("usdz_packaging_error") is None

    packaged_stage = Usd.Stage.Open(packaged)
    composed = UsdShade.Shader(
        packaged_stage.GetPrimAtPath("/World/A/Looks/Paint/Preview")
    )
    assert composed.GetInput("diffuseColor").Get() == pytest.approx(
        Gf.Vec3f(0.9, 0.1, 0.1)
    )


def test_layered_package_allows_absolute_looking_string_value(
    tmp_path: Path,
) -> None:
    """An ordinary String value is not an asset dependency."""
    pytest.importorskip("pxr")
    from pxr import Sdf

    session_dir, input_dir, _prepared_dir, output_dir = _layered_paths(tmp_path)
    root_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Xform "Referenced" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
}
"""
    source_usdz = input_dir / "string-value.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr(
            "Scene/Model/model.usda",
            '#usda 1.0\ndef Xform "Model" {}\n',
        )
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_text, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }
    staged_root = _prepare_source_usdz_stage(context, session_dir)
    assert staged_root is not None
    extract_root = Path(context["source_usdz_extract_root"])

    # An ordinary String value whose text happens to look like an absolute
    # in-package asset dependency. Scanning serialized layer text misreads
    # this as a dependency and rejects an otherwise valid package.
    root_layer = Sdf.Layer.FindOrOpen(str(staged_root))
    assert root_layer is not None
    root_spec = root_layer.GetPrimAtPath("/Root")
    assert root_spec is not None
    decoy = extract_root / "cache" / "looks" / "team"
    note = Sdf.AttributeSpec(root_spec, "note", Sdf.ValueTypeNames.String)
    note.default = f"@{decoy}@"
    root_layer.Save()
    # The decoy really is present in the serialized text.
    assert f"@{decoy}@" in root_layer.ExportToString()

    packaged = _package_usdz(context, session_dir)

    assert "usdz_absolute_dependency_count" not in context
    assert "usdz_absolute_dependencies" not in context
    assert context.get("usdz_packaging_error") is None
    assert packaged is not None


def test_layered_reconstruction_ignores_non_archive_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anonymous/session and external specs cannot become editable targets."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    external_layer = tmp_path / "external.usda"
    external_layer.write_text(
        """#usda 1.0
def Xform "External"
{
    int externalValue = 7
}
""",
        encoding="utf-8",
    )
    source_usdz = input_dir / "external-layer.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            f"""#usda 1.0
(
    subLayers = [@{external_layer.as_posix()}@]
)
over "External"
{{
}}
""",
        )

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    original_open = Usd.Stage.Open
    added_session_layer = False

    def open_with_one_session_layer(
        path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal added_session_layer
        path_text = str(path)
        if (
            not added_session_layer
            and ".texture_agent_source_usdz" in path_text
            and path_text.endswith("root.usda")
        ):
            root_layer = Sdf.Layer.FindOrOpen(path_text)
            assert root_layer is not None
            session_layer = Sdf.Layer.CreateAnonymous("coverage-session.usda")
            session_spec = Sdf.CreatePrimInLayer(session_layer, "/External")
            session_spec.specifier = Sdf.SpecifierOver
            added_session_layer = True
            return original_open(root_layer, session_layer)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Usd.Stage, "Open", open_with_one_session_layer)
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    assert added_session_layer is True
    reconstructed = original_open(str(staged_root))
    assert reconstructed.GetPrimAtPath("/External").IsValid()
    assert reconstructed.GetAttributeAtPath("/External.externalValue").Get() == 7


def test_layered_reconstruction_remaps_all_path_container_shapes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copied payload specs rebase nested metadata and list-edited properties."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = input_dir / "payload.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr(
            "Scene/root.usda",
            """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend payload = @Payload/model.usda@</Model>
)
{
}
""",
        )
        package.writestr(
            "Scene/Payload/model.usda",
            """#usda 1.0
(
    defaultPrim = "Model"
)
def Xform "Model"
{
    def Scope "Looks"
    {
        def Material "Paint" (
            doc = "remove this prepared metadata"
        )
        {
        }
    }
}
""",
        )

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    assert edited_stage is not None
    paint = edited_stage.GetPrimAtPath("/Asset/Looks/Paint")
    paint.ClearMetadata("documentation")
    edited_stage.DefinePrim("/Novel", "Scope")
    edited_stage.DefinePrim("/Asset/Looks/Paint/NewGraph", "Scope")
    edited_stage.DefinePrim("/Asset/Looks/Paint/NewGraph/Child", "Shader")
    edited_layer = edited_stage.GetRootLayer()
    novel_spec = edited_layer.GetPrimAtPath("/Novel")
    assert novel_spec is not None
    novel_spec.SetInfo(
        "customData",
        {"rootMappedPath": Sdf.Path("/Novel")},
    )
    paint_spec = edited_layer.GetPrimAtPath("/Asset/Looks/Paint")
    assert paint_spec is not None
    paint_spec.SetInfo(
        "customData",
        {"rebasedPath": Sdf.Path("/Asset/Looks/Paint")},
    )
    graph_spec = edited_layer.GetPrimAtPath("/Asset/Looks/Paint/NewGraph")
    assert graph_spec is not None
    path_list = Sdf.PathListOp()
    path_list.prependedItems = [
        Sdf.Path("/Asset/Looks/Paint/NewGraph/Child.outputs:value")
    ]
    graph_spec.SetInfo(
        "customData",
        {
            "direct": Sdf.Path("/Asset/Looks/Paint/NewGraph/Child"),
            "relative": Sdf.Path("Child"),
            "nested": [
                {
                    "paths": path_list,
                }
            ],
        },
    )
    graph_spec.propertyOrder = ["inputs:link", "targets"]
    graph_spec.nameChildrenOrder = ["Child"]
    connection = Sdf.AttributeSpec(
        graph_spec,
        "inputs:link",
        Sdf.ValueTypeNames.Float,
        Sdf.VariabilityVarying,
    )
    connection.connectionPathList.prependedItems = [
        Sdf.Path("/Asset/Looks/Paint/NewGraph/Child.outputs:value")
    ]
    targets = Sdf.RelationshipSpec(graph_spec, "targets", False)
    targets.targetPathList.prependedItems = [
        Sdf.Path("/Asset/Looks/Paint/NewGraph/Child")
    ]
    edited_layer.Save()

    original_open = Usd.Stage.Open

    class _UnflattenedEditedStage:
        def __init__(self, stage: Any) -> None:
            self._stage = stage

        def Flatten(self) -> Any:  # noqa: N802 - mirrors the USD API
            return self._stage.GetRootLayer()

    def open_with_unflattened_output(
        path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        stage = original_open(path, *args, **kwargs)
        if str(path) == str(output_usd):
            return _UnflattenedEditedStage(stage)
        return stage

    tuple_remaps: list[tuple[Any, ...]] = []
    original_get_prim = Sdf.Layer.GetPrimAtPath

    class _TupleMetadataPrimSpec:
        def __init__(self, spec: Any) -> None:
            self._spec = spec

        def ListInfoKeys(self) -> list[str]:  # noqa: N802 - mirrors the Sdf API
            return [*self._spec.ListInfoKeys(), "_coverageTuple"]

        def GetInfo(self, key: str) -> Any:  # noqa: N802 - mirrors the Sdf API
            if key == "_coverageTuple":
                return (Sdf.Path("/Asset/Looks/Paint/NewGraph/Child"),)
            return self._spec.GetInfo(key)

        def SetInfo(  # noqa: N802 - mirrors the Sdf API
            self,
            key: str,
            value: Any,
        ) -> None:
            if key == "_coverageTuple":
                tuple_remaps.append(value)
                return
            self._spec.SetInfo(key, value)

        def __getattr__(self, name: str) -> Any:
            return getattr(self._spec, name)

    def get_prim_with_tuple_metadata(layer: Any, path: Any) -> Any:
        spec = original_get_prim(layer, path)
        if (
            spec is not None
            and str(path) == "/Model/Looks/Paint/NewGraph"
            and ".texture_agent_source_usdz" in str(layer.realPath)
        ):
            return _TupleMetadataPrimSpec(spec)
        return spec

    monkeypatch.setattr(Usd.Stage, "Open", open_with_unflattened_output)
    monkeypatch.setattr(Sdf.Layer, "GetPrimAtPath", get_prim_with_tuple_metadata)
    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None
    assert tuple_remaps == [
        (Sdf.Path("/Model/Looks/Paint/NewGraph/Child"),),
    ]
    reconstructed = original_open(str(staged_root))
    assert reconstructed.GetPrimAtPath("/Novel").IsValid()
    reconstructed_novel = reconstructed.GetRootLayer().GetPrimAtPath("/Novel")
    assert reconstructed_novel.GetInfo("customData")["rootMappedPath"] == Sdf.Path(
        "/Novel"
    )
    payload_layer = next(
        layer
        for layer in reconstructed.GetUsedLayers()
        if layer.realPath.endswith("Payload/model.usda")
    )
    payload_paint = original_get_prim(
        payload_layer,
        Sdf.Path("/Model/Looks/Paint"),
    )
    assert not payload_paint.HasInfo("documentation")
    payload_graph = original_get_prim(
        payload_layer,
        Sdf.Path("/Model/Looks/Paint/NewGraph"),
    )
    custom_data = payload_graph.GetInfo("customData")
    assert custom_data["direct"] == Sdf.Path("/Model/Looks/Paint/NewGraph/Child")
    assert custom_data["relative"] == Sdf.Path("Child")
    nested_path_list = custom_data["nested"][0]["paths"]
    assert nested_path_list.prependedItems == [
        Sdf.Path("/Model/Looks/Paint/NewGraph/Child.outputs:value")
    ]
    copied_connection = payload_layer.GetAttributeAtPath(
        "/Model/Looks/Paint/NewGraph.inputs:link"
    )
    assert copied_connection.connectionPathList.prependedItems == [
        Sdf.Path("/Model/Looks/Paint/NewGraph/Child.outputs:value")
    ]
    copied_targets = payload_layer.GetRelationshipAtPath(
        "/Model/Looks/Paint/NewGraph.targets"
    )
    assert copied_targets.targetPathList.prependedItems == [
        Sdf.Path("/Model/Looks/Paint/NewGraph/Child")
    ]


def test_layered_package_reuses_and_collision_suffixes_generated_member(
    tmp_path: Path,
) -> None:
    """Stored generated members are reused; occupied fallback names are skipped."""
    pytest.importorskip("pxr")

    session_dir, input_dir, _prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = input_dir / "simple.usdz"
    root_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
}
"""
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(root_text, encoding="utf-8")
    context = {
        "usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
        "usdz_generated_textures_member": (".texture_agent_generated_textures_saved"),
    }

    first_package = _package_usdz(context, session_dir)

    assert first_package is not None
    assert (
        context["usdz_generated_textures_member"]
        == ".texture_agent_generated_textures_saved"
    )
    extracted_root = Path(context["source_usdz_extract_root"])
    package_usd = Path(context["source_usdz_stage_path"])
    base_name = ".texture_agent_generated_textures"
    digest = hashlib.sha256(str(package_usd).encode("utf-8")).hexdigest()[:10]
    for name in (base_name, f"{base_name}_{digest}", f"{base_name}_{digest}_1"):
        (extracted_root / name).mkdir()
    context.pop("usdz_generated_textures_member")

    second_package = _package_usdz(context, session_dir)

    assert second_package is not None
    assert context["usdz_generated_textures_member"] == f"{base_name}_{digest}_2"


def test_layer_authored_asset_paths_covers_arrays_and_time_samples(
    tmp_path: Path,
) -> None:
    """Asset arrays, time samples, and non-asset specs are handled."""
    pytest.importorskip("pxr")
    from pxr import Sdf

    from ...service.workers.executor import _layer_authored_asset_paths

    layer = Sdf.Layer.CreateNew(str(tmp_path / "assets.usda"))
    prim = Sdf.PrimSpec(layer.pseudoRoot, "Root", Sdf.SpecifierDef)

    single = Sdf.AttributeSpec(prim, "single", Sdf.ValueTypeNames.Asset)
    single.default = Sdf.AssetPath("one.png")
    # An empty asset path contributes nothing.
    blank = Sdf.AttributeSpec(prim, "blank", Sdf.ValueTypeNames.Asset)
    blank.default = Sdf.AssetPath("")
    # Array-valued asset attributes are unpacked element by element.
    array = Sdf.AttributeSpec(prim, "many", Sdf.ValueTypeNames.AssetArray)
    array.default = Sdf.AssetPathArray(
        [Sdf.AssetPath("a.png"), Sdf.AssetPath(""), Sdf.AssetPath("b.png")]
    )
    # A non-asset spec is skipped entirely.
    note = Sdf.AttributeSpec(prim, "note", Sdf.ValueTypeNames.String)
    note.default = "@not/a/dependency@"
    # Time-sampled asset values are inspected too.
    sampled = Sdf.AttributeSpec(prim, "sampled", Sdf.ValueTypeNames.Asset)
    layer.SetTimeSample(sampled.path, 0.0, Sdf.AssetPath("frame0.png"))

    authored = _layer_authored_asset_paths(layer)

    assert "one.png" in authored
    assert "a.png" in authored and "b.png" in authored
    assert "frame0.png" in authored
    assert "" not in authored
    assert "not/a/dependency" not in authored


def test_layer_authored_asset_paths_tolerates_broken_layer() -> None:
    """A layer stub missing the dependency APIs yields no paths."""
    from types import SimpleNamespace

    from ...service.workers.executor import _layer_authored_asset_paths

    assert _layer_authored_asset_paths(SimpleNamespace()) == []


def test_layered_arc_index_handles_diamonds_and_external_targets(
    tmp_path: Path,
) -> None:
    """A layer reached twice is visited once; non-archive targets are skipped."""
    pytest.importorskip("pxr")
    from pxr import Usd

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    # root -> A and B, both of which reference the same C (diamond), plus a
    # dangling reference whose target never resolves to an extracted layer.
    root_text = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World"
{
    def Xform "A" (
        prepend references = @A/a.usda@</A>
    )
    {
    }
    def Xform "B" (
        prepend references = @B/b.usda@</B>
    )
    {
    }
    def Xform "Dangling" (
        prepend references = @Missing/none.usda@</Missing>
    )
    {
    }
}
"""
    source_usdz = input_dir / "diamond.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr(
            "Scene/A/a.usda",
            '#usda 1.0\ndef Xform "A" (\n    prepend references = '
            "@../C/c.usda@</C>\n)\n{\n}\n",
        )
        package.writestr(
            "Scene/B/b.usda",
            '#usda 1.0\ndef Xform "B" (\n    prepend references = '
            "@../C/c.usda@</C>\n)\n{\n}\n",
        )
        package.writestr("Scene/C/c.usda", '#usda 1.0\ndef Xform "C" {}\n')

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    context = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    # Building the sharing index must survive both shapes.
    staged_root = _prepare_source_usdz_stage(context, session_dir)
    assert staged_root is not None


def test_layered_delta_allows_untouched_prim_over_shared_source_spec(
    tmp_path: Path,
) -> None:
    """A shared source spec nobody edited must not refuse the package.

    Two instances compose one model, so every prim inside it reaches several
    composed paths over a single source spec. Resolving a destination is what
    enforces the shared-source guard, so a prim carrying no edit must never
    reach it, or an untouched material refuses the package on behalf of edits it
    never received. Observed on the M1126 Stryker acceptance asset once UV repair
    began writing a prepared stage: apply_textures succeeded, then packaging
    refused at ``/Body/Prototypes/Body/Looks/Diffuse`` -- a material no texture
    unit had touched.
    """
    pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    # Both branches compose now, so /Model/Looks/Paint is one source spec reached
    # from /World/A/Looks/Paint and /World/B/Looks/Paint.
    root_text = """#usda 1.0
(
    defaultPrim = "World"
)
def Xform "World"
{
    def Xform "A" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    def Xform "B" (
        prepend references = @Model/model.usda@</Model>
    )
    {
    }
    def Scope "Looks"
    {
        def Material "Trim"
        {
            def Shader "Preview"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.5, 0.5, 0.5)
            }
        }
    }
}
"""
    source_usdz = input_dir / "shared-untouched.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)
        package.writestr("Scene/Model/model.usda", _SHARED_MODEL_USDA)

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    # Edit only /World/Looks/Trim, which is authored in the root layer and shared
    # by nothing. The shared /Model/Looks/Paint is left exactly as composed.
    edited_stage = Usd.Stage.Open(str(output_usd))
    trim = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/Looks/Trim/Preview"))
    trim.GetInput("diffuseColor").Set(Gf.Vec3f(0.9, 0.1, 0.1))
    edited_stage.GetRootLayer().Save()

    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    assert "usdz_packaging_error" not in context

    # Success alone would also hold if staging silently dropped the edit.
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed is not None
    trim = UsdShade.Shader(reconstructed.GetPrimAtPath("/World/Looks/Trim/Preview"))
    assert trim.GetInput("diffuseColor").Get() == Gf.Vec3f(0.9, 0.1, 0.1)


def test_layered_delta_carries_a_property_deletion(tmp_path: Path) -> None:
    """A prim whose only edit removes a property must carry that removal.

    ``changed_properties`` iterates the edited spec, so a property present in the
    source and absent from the edit is invisible to it: the prim reads as
    untouched and the early-exit skips it. The copy loop is likewise additive and
    never removes a property, so the removal was dropped even before the
    early-exit existed.
    """
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = input_dir / "deletion.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", _DELETION_ROOT_USDA)

    source_stage = Usd.Stage.Open(str(source_usdz))
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    shader = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/Looks/Paint/Preview"))
    assert shader.GetInput("roughness"), "fixture must start with the property"
    shader.GetPrim().RemoveProperty("inputs:roughness")
    edited_stage.GetRootLayer().Save()

    # The edit must have persisted, or the rest of the test proves nothing.
    reopened = Usd.Stage.Open(str(output_usd))
    assert not UsdShade.Shader(
        reopened.GetPrimAtPath("/World/Looks/Paint/Preview")
    ).GetInput("roughness")

    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    assert "usdz_packaging_error" not in context
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed is not None
    rebuilt = UsdShade.Shader(reconstructed.GetPrimAtPath("/World/Looks/Paint/Preview"))
    assert rebuilt.GetPrim().IsValid()
    assert not rebuilt.GetInput("roughness"), "deletion was dropped by the writeback"
    # The rest of the prim must survive the removal.
    assert rebuilt.GetInput("diffuseColor").Get() is not None


def test_layered_delta_deletes_all_weaker_property_opinions(tmp_path: Path) -> None:
    """Removing a flattened property must not expose a weaker layer opinion."""
    pytest.importorskip("pxr")
    from pxr import Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = input_dir / "layered-deletion.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", _WEAKER_DELETION_ROOT_USDA)
        package.writestr("Scene/Model/model.usda", _WEAKER_DELETION_MODEL_USDA)

    source_stage = Usd.Stage.Open(str(source_usdz))
    source_shader = UsdShade.Shader(
        source_stage.GetPrimAtPath("/World/Looks/Paint/Preview")
    )
    source_roughness = source_shader.GetInput("roughness")
    assert source_roughness.Get() == 0.5
    source_stack = source_roughness.GetAttr().GetPropertyStack()
    assert len(source_stack) == 2
    assert sorted(float(spec.default) for spec in source_stack) == pytest.approx(
        [0.2, 0.5]
    )

    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    edited_shader = UsdShade.Shader(
        edited_stage.GetPrimAtPath("/World/Looks/Paint/Preview")
    )
    edited_shader.GetPrim().RemoveProperty("inputs:roughness")
    edited_stage.GetRootLayer().Save()

    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }
    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed is not None
    rebuilt = UsdShade.Shader(reconstructed.GetPrimAtPath("/World/Looks/Paint/Preview"))
    assert not rebuilt.GetInput("roughness"), (
        "deleting only the strongest spec exposed the referenced value"
    )
    assert rebuilt.GetInput("diffuseColor").Get() is not None


def test_layered_delta_deletion_ignores_non_archive_property_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deletion mutates archive specs, never session or external layers."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd, UsdShade

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    external_layer = tmp_path / "external.usda"
    external_layer.write_text(
        '#usda 1.0\ndef Xform "External" {\n    int externalValue = 7\n}\n',
        encoding="utf-8",
    )
    root_text = _DELETION_ROOT_USDA.replace(
        'defaultPrim = "World"',
        f'defaultPrim = "World"\n    subLayers = [@{external_layer.as_posix()}@]',
    )
    root_text += '\nover "External"\n{\n}\n'
    source_usdz = input_dir / "non-archive-property.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("Scene/root.usda", root_text)

    source_stage = Usd.Stage.Open(str(source_usdz))
    assert source_stage is not None
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    assert edited_stage is not None
    shader = UsdShade.Shader(edited_stage.GetPrimAtPath("/World/Looks/Paint/Preview"))
    shader.GetPrim().RemoveProperty("inputs:roughness")
    edited_stage.GetPrimAtPath("/External").RemoveProperty("externalValue")
    edited_stage.GetRootLayer().Save()

    original_open = Usd.Stage.Open
    added_session_layer = False

    def open_with_session_property(
        path: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        nonlocal added_session_layer
        path_text = str(path)
        if (
            not added_session_layer
            and ".texture_agent_source_usdz" in path_text
            and path_text.endswith("root.usda")
        ):
            root_layer = Sdf.Layer.FindOrOpen(path_text)
            assert root_layer is not None
            session_layer = Sdf.Layer.CreateAnonymous("coverage-session.usda")
            session_prim = Sdf.CreatePrimInLayer(
                session_layer,
                "/World/Looks/Paint/Preview",
            )
            session_prim.specifier = Sdf.SpecifierOver
            session_roughness = Sdf.AttributeSpec(
                session_prim,
                "inputs:roughness",
                Sdf.ValueTypeNames.Float,
            )
            session_roughness.default = 0.9
            added_session_layer = True
            return original_open(root_layer, session_layer)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Usd.Stage, "Open", open_with_session_property)
    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    assert added_session_layer is True
    reconstructed = original_open(str(staged_root))
    assert reconstructed is not None
    rebuilt = UsdShade.Shader(reconstructed.GetPrimAtPath("/World/Looks/Paint/Preview"))
    assert not rebuilt.GetInput("roughness")
    assert reconstructed.GetAttributeAtPath("/External.externalValue").Get() == 7
    external = original_open(str(external_layer))
    assert external.GetAttributeAtPath("/External.externalValue").Get() == 7


def test_asset_signature_falls_back_to_authored_path_when_resolver_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A resolver exception cannot turn an unchanged Apply asset into a delta."""
    pytest.importorskip("pxr")
    from pxr import Sdf, Usd

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_text = """#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{
    def Material "Paint"
    {
        asset inputs:file = @missing.png@
    }
}
"""
    source_usdz = input_dir / "resolver-fallback.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("scene.usda", source_text)
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    prepared_usd.write_text(source_text, encoding="utf-8")
    output_usd.write_text(source_text, encoding="utf-8")

    resolver_calls: list[str] = []

    def failing_resolver(_layer: Any, path: str) -> str:
        resolver_calls.append(path)
        raise ValueError("synthetic resolver failure")

    monkeypatch.setattr(Sdf, "ComputeAssetPathRelativeToLayer", failing_resolver)
    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    assert context["source_usdz_prepared_edit_layer_paths"] == []
    assert resolver_calls == ["missing.png", "missing.png"]
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert (
        reconstructed.GetAttributeAtPath("/Root/Paint.inputs:file").Get().path
        == "missing.png"
    )


def test_apply_uv_mutation_is_ignored_with_complete_source_uv_spec(
    tmp_path: Path,
) -> None:
    """Only preparation may publish UV specs; Apply cannot replace them."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_layer = tmp_path / "uv-source.usda"
    source_stage = Usd.Stage.CreateNew(str(source_layer))
    root = UsdGeom.Xform.Define(source_stage, "/Root").GetPrim()
    source_stage.SetDefaultPrim(root)
    mesh = UsdGeom.Mesh.Define(source_stage, "/Root/Body")
    source_values = Vt.Vec2fArray(
        [
            Gf.Vec2f(0.0, 0.0),
            Gf.Vec2f(1.0, 0.0),
            Gf.Vec2f(0.0, 1.0),
        ]
    )
    source_st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    assert source_st.Set(source_values)
    assert source_st.GetAttr().AddConnection(Sdf.Path("/Root/Body.points"))
    assert source_stage.GetRootLayer().Save()

    source_usdz = input_dir / "uv-source.usdz"
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.write(source_layer, "scene.usda")
    prepared_usd = prepared_dir / "prepared.usda"
    output_usd = output_dir / "textured_output.usda"
    assert source_stage.Flatten().Export(str(prepared_usd))
    assert source_stage.Flatten().Export(str(output_usd))

    edited_stage = Usd.Stage.Open(str(output_usd))
    edited_st = UsdGeom.PrimvarsAPI(
        edited_stage.GetPrimAtPath("/Root/Body")
    ).GetPrimvar("st")
    assert edited_st.Set(
        Vt.Vec2fArray(
            [
                Gf.Vec2f(0.25, 0.25),
                Gf.Vec2f(0.75, 0.25),
                Gf.Vec2f(0.25, 0.75),
            ]
        )
    )
    assert edited_st.GetAttr().SetConnections(
        [Sdf.Path("/Root/ApplyMustNotPublish.points")]
    )
    edited_prim = edited_stage.GetPrimAtPath("/Root/Body")
    edited_prim.SetCustomDataByKey("applyMarker", "keep")
    assert edited_stage.GetRootLayer().Save()
    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    reconstructed_st = UsdGeom.PrimvarsAPI(
        reconstructed.GetPrimAtPath("/Root/Body")
    ).GetPrimvar("st")
    assert reconstructed_st.Get() == source_values
    assert reconstructed_st.GetAttr().GetConnections() == [
        Sdf.Path("/Root/Body.points")
    ]
    assert (
        reconstructed.GetPrimAtPath("/Root/Body").GetCustomDataByKey("applyMarker")
        == "keep"
    )


def test_three_generation_instance_writeback_skips_inactive_baseline_backing(
    tmp_path: Path,
) -> None:
    """An inactive prototype retained in B is not a trusted UV destination."""
    pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt

    session_dir, input_dir, prepared_dir, output_dir = _layered_paths(tmp_path)
    source_usdz = _write_instanced_uv_usdz(
        tmp_path,
        input_dir / "instanced.usdz",
        instance_names=("Body",),
    )
    source_stage = Usd.Stage.Open(str(source_usdz))
    source_uvs = (
        UsdGeom.PrimvarsAPI(source_stage.GetPrimAtPath("/Body/Body/Part/Mesh"))
        .GetPrimvar("st")
        .Get()
    )
    first_flat_stage = Usd.Stage.Open(source_stage.Flatten())
    prepared_usd = prepared_dir / "prepared.usda"
    assert first_flat_stage.Flatten().Export(str(prepared_usd))
    prepared_stage = Usd.Stage.Open(str(prepared_usd))
    synthetic_backings = [
        spec
        for spec in prepared_stage.GetRootLayer().rootPrims
        if spec.specifier == Sdf.SpecifierOver
        and spec.name.startswith("Flattened_Prototype_")
        and not prepared_stage.GetPrimAtPath(spec.path).IsDefined()
    ]
    assert synthetic_backings
    prepared_uvs = Vt.Vec2fArray(
        [
            Gf.Vec2f(0.1, 0.1),
            Gf.Vec2f(0.9, 0.1),
            Gf.Vec2f(0.9, 0.9),
            Gf.Vec2f(0.1, 0.9),
        ]
    )
    prepared_st = UsdGeom.PrimvarsAPI(
        prepared_stage.GetPrimAtPath("/Body/Body/Part/Mesh")
    ).GetPrimvar("st")
    active_st_specs = [
        spec
        for spec in prepared_st.GetAttr().GetPropertyStack()
        if spec.layer == prepared_stage.GetRootLayer()
    ]
    assert len(active_st_specs) == 1
    superseded_backings = [
        backing
        for backing in synthetic_backings
        if not active_st_specs[0].path.HasPrefix(backing.path)
    ]
    assert superseded_backings
    active_st_specs[0].default = prepared_uvs
    assert prepared_stage.GetRootLayer().Save()
    assert prepared_st.Get() == prepared_uvs

    output_usd = output_dir / "textured_output.usda"
    assert prepared_stage.Flatten().Export(str(output_usd))
    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "usd_path": str(prepared_usd),
        "output_usd_paths": [str(output_usd)],
    }

    staged_root = _prepare_source_usdz_stage(context, session_dir)

    assert staged_root is not None, context.get("usdz_packaging_error")
    reconstructed = Usd.Stage.Open(str(staged_root))
    assert reconstructed.GetPrimAtPath("/Body/Body").IsInstance()
    assert reconstructed.GetPrimAtPath("/Body/Body/Part/Mesh").IsInstanceProxy()
    assert (
        UsdGeom.PrimvarsAPI(reconstructed.GetPrimAtPath("/Body/Body/Part/Mesh"))
        .GetPrimvar("st")
        .Get()
        == prepared_uvs
    )
    assert (
        UsdGeom.PrimvarsAPI(
            reconstructed.GetPrimAtPath("/Body/Prototypes/Body/Part/Mesh")
        )
        .GetPrimvar("st")
        .Get()
        == prepared_uvs
    )
    assert (
        UsdGeom.PrimvarsAPI(reconstructed.GetPrimAtPath("/Body/Unselected/Mesh"))
        .GetPrimvar("st")
        .Get()
        == source_uvs
    )
    assert not any(
        spec.name.startswith("Flattened_Prototype_")
        for spec in reconstructed.GetRootLayer().rootPrims
    )
    published_layer_paths = {
        Path(staged_root),
        *(Path(path) for path in context["source_usdz_prepared_edit_layer_paths"]),
    }
    assert context["source_usdz_prepared_edit_layer_paths"]
    for layer_path in published_layer_paths:
        published_layer = Sdf.Layer.FindOrOpen(str(layer_path))
        assert published_layer is not None
        assert "Flattened_Prototype_" not in published_layer.ExportToString()


def test_package_usdz_resolves_existing_source_package_member(
    tmp_path: Path,
) -> None:
    """An explicit source.usdz[member] texture resolves to the staged member."""
    pytest.importorskip("pxr")
    from PIL import Image
    from pxr import Usd

    session_dir, input_dir, _prepared_dir, output_dir = _layered_paths(tmp_path)
    source_texture = tmp_path / "source.png"
    Image.new("RGB", (2, 2), (12, 34, 56)).save(source_texture)
    source_bytes = source_texture.read_bytes()
    source_usdz = input_dir / "source-member.usdz"
    source_text = '#usda 1.0\n(\n    defaultPrim = "Root"\n)\ndef Xform "Root" {}\n'
    with zipfile.ZipFile(source_usdz, "w", zipfile.ZIP_STORED) as package:
        package.writestr("scene.usda", source_text)
        package.write(source_texture, "Textures/source.png")

    output_usd = output_dir / "textured_output.usda"
    output_usd.write_text(
        f"""#usda 1.0
(
    defaultPrim = "Root"
)
def Xform "Root"
{{
    def Shader "Shader"
    {{
        uniform token info:id = "UsdPreviewSurface"
        asset inputs:diffuse_texture = @{source_usdz}[Textures/source.png]@
    }}
}}
""",
        encoding="utf-8",
    )
    context: dict[str, Any] = {
        "source_usd_path": str(source_usdz),
        "output_usd_paths": [str(output_usd)],
    }

    packaged_path = _package_usdz(context, session_dir)

    assert packaged_path is not None, context.get("usdz_packaging_error")
    source_usdz.rename(tmp_path / "source-member-unavailable.usdz")
    packaged_stage = Usd.Stage.Open(packaged_path)
    packaged_value = packaged_stage.GetAttributeAtPath(
        "/Root/Shader.inputs:diffuse_texture"
    ).Get()
    assert packaged_value.path == "Textures/source.png"
    assert packaged_value.resolvedPath
    with zipfile.ZipFile(packaged_path) as package:
        assert package.read("Textures/source.png") == source_bytes
