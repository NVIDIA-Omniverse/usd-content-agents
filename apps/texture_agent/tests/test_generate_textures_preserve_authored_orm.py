# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Authored ORM preservation (issue #950).

Regression cover for the case where an asset ships a packed ORM and the pipeline
replaced it wholesale. Occlusion and metallic must come from the authored map;
roughness must remain whatever the backend generated, because roughness is the
channel a weathering edit legitimately moves.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from texture_agent.functions.material_discovery import MaterialInfo, PrimTextureUnit
from texture_agent.functions.texture_generation import GeneratedTextures
from texture_agent.tasks.generate_textures import (
    _local_artifact_sha256,
    _preserve_authored_orm,
)

AUTHORED = (230, 40, 250)  # occlusion, roughness, metallic -- a metal asset
GENERATED = (255, 118, 75)  # what the backends produce today


def _write_orm(path: Path, rgb: tuple[int, int, int], size: int = 16) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (size, size), rgb).save(path)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(
    key: str, *, orm_texture: str | None, bound: str = "/World/Mesh"
) -> PrimTextureUnit:
    return PrimTextureUnit(
        prim_path="",
        material_info=MaterialInfo(
            prim_path=f"/World/Looks/{key}",
            name=key,
            bound_prim_paths=[bound],
            orm_texture=orm_texture,
        ),
        key=key,
        prompt=f"weathered {key}",
        opacity=1.0,
    )


@pytest.fixture
def scene(tmp_path: Path) -> dict:
    """A stage-adjacent authored ORM plus a freshly generated one."""
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_orm(tmp_path / "Textures" / "authored_orm.png", AUTHORED)
    generated = _write_orm(tmp_path / "generated" / "Metal_orm.png", GENERATED)
    albedo = _write_orm(
        tmp_path / "generated" / "Metal_albedo.png",
        (80, 70, 60),
    )
    return {
        "usd_path": str(usd_path),
        "generated": generated,
        "albedo": albedo,
        "textures": GeneratedTextures(albedo=str(albedo), orm=str(generated)),
    }


def _passing_receipt_context(scene: dict) -> dict:
    mask = scene["generated"].with_name("Metal_weathering_mask.png")
    Image.new("L", (16, 16), 128).save(mask)
    mask_sha256 = _sha256(mask)
    artifacts = {
        "source_albedo_sha256": "1" * 64,
        "candidate_albedo_sha256": "2" * 64,
        "candidate_orm_sha256": "3" * 64,
        "final_albedo_sha256": _sha256(scene["albedo"]),
        "final_orm_sha256": _sha256(scene["generated"]),
        "weathering_mask_sha256": mask_sha256,
        "source_usd_sha256": _sha256(Path(scene["usd_path"])),
        "request_sha256": "5" * 64,
    }
    return {
        "usd_path": scene["usd_path"],
        "generated_textures": {"Metal": scene["textures"]},
        "projection_backend_results": {
            "Metal": {
                "source_asset_uri": Path(scene["usd_path"]).as_uri(),
                "metadata": {
                    "weathering": {
                        "schema_version": "texture-weathering-evidence.v1",
                        "effect": "rust",
                        "status": "pass",
                        "internal_plan": {"effect": "rust"},
                        "metrics": {"localization_ratio": 1.0},
                        "thresholds": {"localization_ratio_min": 0.95},
                        "failures": [],
                        "artifacts": artifacts,
                    }
                },
                "auxiliary_artifacts": {
                    "masks": {
                        "weathering": {
                            "uri": str(mask),
                            "sha256": mask_sha256,
                        }
                    }
                },
            }
        },
    }


def _channels(path: Path) -> tuple[int, int, int]:
    with Image.open(path) as img:
        return img.convert("RGB").getpixel((0, 0))


def test_authored_occlusion_and_metallic_replace_generated(scene: dict) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={"usd_path": scene["usd_path"]},
    )

    occlusion, roughness, metallic = _channels(scene["generated"])
    assert occlusion == AUTHORED[0], "authored occlusion must survive"
    assert metallic == AUTHORED[2], "authored metallic must survive"
    assert roughness == GENERATED[1], "generated roughness must be retained"
    assert len(diagnostics) == 1
    assert diagnostics[0]["code"] == "AUTHORED_ORM_PRESERVED"
    assert diagnostics[0]["details"]["source"] == "preserved_input"


def test_passing_rust_receipt_retains_localized_generated_metallic(scene: dict) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context=_passing_receipt_context(scene),
    )

    occlusion, roughness, metallic = _channels(scene["generated"])
    assert occlusion == AUTHORED[0]
    assert roughness == GENERATED[1]
    assert metallic == GENERATED[2]
    assert diagnostics[0]["details"]["preserved_components"] == ["occlusion"]
    assert diagnostics[0]["details"]["localized_components"] == ["metallic"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_digest",
        "stale_orm",
        "stale_mask",
        "missing_internal_plan",
        "invalid_artifacts",
        "missing_textures",
        "stale_albedo",
        "missing_mask",
        "stale_source",
    ],
)
def test_incomplete_or_stale_rust_receipt_preserves_authored_metallic(
    scene: dict,
    mutation: str,
) -> None:
    context = _passing_receipt_context(scene)
    record = context["projection_backend_results"]["Metal"]
    evidence = record["metadata"]["weathering"]
    if mutation == "missing_digest":
        evidence["artifacts"].pop("request_sha256")
    elif mutation == "stale_orm":
        evidence["artifacts"]["final_orm_sha256"] = "0" * 64
    elif mutation == "stale_mask":
        record["auxiliary_artifacts"]["masks"]["weathering"]["sha256"] = "0" * 64
    elif mutation == "missing_internal_plan":
        evidence.pop("internal_plan")
    elif mutation == "invalid_artifacts":
        evidence["artifacts"] = "not-artifacts"
    elif mutation == "missing_textures":
        context["generated_textures"] = {}
    elif mutation == "stale_albedo":
        evidence["artifacts"]["final_albedo_sha256"] = "0" * 64
    elif mutation == "missing_mask":
        record["auxiliary_artifacts"] = {}
    else:
        evidence["artifacts"]["source_usd_sha256"] = "0" * 64

    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[_unit("Metal", orm_texture="./Textures/authored_orm.png")],
        context=context,
    )

    assert _channels(scene["generated"])[2] == AUTHORED[2]


def test_local_artifact_digest_fails_closed_for_unreadable_inputs(
    tmp_path: Path,
) -> None:
    assert _local_artifact_sha256(None) is None
    assert _local_artifact_sha256(str(tmp_path / "missing.png")) is None
    assert _local_artifact_sha256("https://example.invalid/map.png") is None


@pytest.mark.parametrize("status", ["fail", None])
def test_unverified_rust_receipt_preserves_authored_metallic(
    scene: dict,
    status: str | None,
) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    evidence = {"effect": "rust"}
    if status is not None:
        evidence["status"] = status
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "projection_backend_results": {
                "Metal": {"metadata": {"weathering": evidence}}
            },
        },
    )

    assert _channels(scene["generated"])[2] == AUTHORED[2]


@pytest.mark.parametrize(
    "record",
    ["not-a-record", {"metadata": "not-metadata"}],
)
def test_malformed_weathering_receipt_preserves_authored_metallic(
    scene: dict,
    record: object,
) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "projection_backend_results": {"Metal": record},
        },
    )

    assert _channels(scene["generated"])[2] == AUTHORED[2]


def test_no_authored_orm_leaves_generated_untouched(scene: dict) -> None:
    unit = _unit("Metal", orm_texture=None)
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={"usd_path": scene["usd_path"]},
    )

    assert _channels(scene["generated"]) == GENERATED
    assert diagnostics == []


def test_regenerated_uvs_skip_preservation(scene: dict) -> None:
    """Authored texels are only valid in the atlas they were authored against."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "generated": 1,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/Mesh"],
            },
        },
    )

    assert _channels(scene["generated"]) == GENERATED
    # A skip must be observable, not silent: callers cannot otherwise tell
    # "preserved" from "quietly gave up".
    assert [d["code"] for d in diagnostics] == ["AUTHORED_ORM_NOT_PRESERVED"]
    assert diagnostics[0]["details"]["reason"] == "uv_layout_regenerated"
    assert diagnostics[0]["severity"] == "warning"


def test_untouched_uv_prims_still_preserve(scene: dict) -> None:
    """A UV mutation elsewhere in the stage must not disable preservation here."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "generated": 1,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/SomethingElse"],
            },
        },
    )

    assert _channels(scene["generated"])[0] == AUTHORED[0]


def test_stage_wide_uv_mutation_is_treated_as_unattributable(scene: dict) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {"generated": 1, "uv_scope": "stage"},
        },
    )

    assert _channels(scene["generated"]) == GENERATED


def test_preservation_is_idempotent(scene: dict) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    ctx = {"usd_path": scene["usd_path"]}
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]}, units=[unit], context=ctx
    )
    first = _channels(scene["generated"])
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]}, units=[unit], context=ctx
    )
    assert _channels(scene["generated"]) == first


def test_higher_res_authored_map_is_not_downsampled(tmp_path: Path) -> None:
    """Authored detail must survive: resolve upward, never shrink the ground truth.

    This is the SimReady shape -- a 4096 authored ORM against a 1024 generated
    one. Downsampling the authored map would smooth away the occlusion/metallic
    detail the preservation exists to protect.
    """
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_orm(tmp_path / "Textures" / "authored_orm.png", AUTHORED, size=64)
    generated = _write_orm(tmp_path / "generated" / "Metal_orm.png", GENERATED, size=16)
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")

    _preserve_authored_orm(
        new_generated={"Metal": GeneratedTextures(orm=str(generated))},
        units=[unit],
        context={"usd_path": str(usd_path)},
    )

    with Image.open(generated) as img:
        assert img.size == (64, 64), "output resolves up to the authored map"
    occlusion, roughness, metallic = _channels(generated)
    assert (occlusion, metallic) == (AUTHORED[0], AUTHORED[2])
    assert roughness == GENERATED[1]


def test_lower_res_authored_map_keeps_generated_resolution(tmp_path: Path) -> None:
    """The generated map is likewise never downsampled to meet a smaller authored one."""
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_orm(tmp_path / "Textures" / "authored_orm.png", AUTHORED, size=16)
    generated = _write_orm(tmp_path / "generated" / "Metal_orm.png", GENERATED, size=64)
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")

    _preserve_authored_orm(
        new_generated={"Metal": GeneratedTextures(orm=str(generated))},
        units=[unit],
        context={"usd_path": str(usd_path)},
    )

    with Image.open(generated) as img:
        assert img.size == (64, 64), "generated resolution retained"
    assert _channels(generated)[2] == AUTHORED[2]


def test_unresolvable_authored_orm_is_non_fatal(scene: dict) -> None:
    """Non-fatal, but reported: a silent skip is indistinguishable from success."""
    unit = _unit("Metal", orm_texture="./Textures/missing.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={"usd_path": scene["usd_path"]},
    )

    assert _channels(scene["generated"]) == GENERATED
    assert [d["code"] for d in diagnostics] == ["AUTHORED_ORM_NOT_PRESERVED"]
    assert diagnostics[0]["details"]["reason"] == "authored_orm_unresolvable"
    assert diagnostics[0]["severity"] == "warning"


def test_uv_report_without_mutations_does_not_block_preservation(scene: dict) -> None:
    """A uv_preparation record that changed nothing must not disable preservation."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "generated": 0,
                "normalized": 0,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/Mesh"],
            },
        },
    )

    assert _channels(scene["generated"])[2] == AUTHORED[2]


def test_target_scope_without_paths_is_unattributable(scene: dict) -> None:
    """target_prims scope with no listed prims cannot be attributed, so skip."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "normalized": 3,
                "uv_scope": "target_prims",
                "target_prim_paths": [],
            },
        },
    )

    assert _channels(scene["generated"]) == GENERATED


def test_missing_usd_path_is_a_no_op(scene: dict) -> None:
    """Without a stage path no unit is examined, so there is nothing to report."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={},
    )

    assert _channels(scene["generated"]) == GENERATED
    assert diagnostics == []


def test_orm_authored_directly_on_material_prim_is_discovered(tmp_path: Path) -> None:
    """MDL-style assets author ORM_texture on the material prim, not via a shader."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.functions.material_discovery import discover_materials

    stage = Usd.Stage.CreateNew(str(tmp_path / "asset.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Mesh")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Metal")
    prim = material.GetPrim()
    prim.CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath("./Textures/albedo.png"))
    prim.CreateAttribute("inputs:ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/authored_orm.png")
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Mesh")).Bind(material)
    stage.Save()

    (material_info,) = discover_materials(stage)
    assert material_info.orm_texture == "./Textures/authored_orm.png"


def test_guard_covers_every_prepare_uvs_mutation_field() -> None:
    """The UV-mutation guard must know every counter prepare_uvs can report.

    A field missing here fails open: the guard never fires, and authored texels
    are composited onto an atlas that moved. Pinned against the literals in
    prepare_uvs rather than duplicated by hand.
    """
    import re
    from pathlib import Path as _Path

    from texture_agent.tasks.generate_textures import _UV_MUTATION_COUNT_FIELDS

    src = _Path("apps/texture_agent/texture_agent/tasks/prepare_uvs.py").read_text(
        encoding="utf-8"
    )
    # A mutation counter is either initialised to integer 0 in the actions dict
    # literal, or assigned from one of the UV-mutating helpers. Deliberately NOT
    # filtered against the names the guard already knows: a counter added to
    # prepare_uvs must fail this test until the guard handles it. Path/scope keys
    # are excluded structurally rather than by name, since they are assigned from
    # list()/str() rather than an int-returning call.
    counters = {m.group(1) for m in re.finditer(r'"(\w+)":\s*0,', src)}
    counters |= {
        m.group(1)
        for m in re.finditer(
            r'actions\["(\w+)"\]\s*=\s*(?:int\(|\w*(?:uvs|interpolation)\w*\()', src
        )
    }

    assert counters, "could not locate prepare_uvs counter fields — update this test"
    missing = counters - set(_UV_MUTATION_COUNT_FIELDS)
    assert not missing, f"guard misses prepare_uvs mutation counters: {sorted(missing)}"


@pytest.mark.parametrize(
    "field", ["generated", "fixed_interpolation", "degenerate_repaired", "normalized"]
)
def test_each_mutation_field_blocks_preservation(scene: dict, field: str) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                field: 2,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/Mesh"],
            },
        },
    )

    assert _channels(scene["generated"]) == GENERATED, (
        f"{field} moves UVs but preservation still fired"
    )


def test_differing_aspect_ratios_do_not_stretch_either_map(tmp_path: Path) -> None:
    """Per-axis max would give 64x64 here and stretch both maps out of correspondence."""
    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    authored = tmp_path / "Textures" / "authored_orm.png"
    authored.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 16), AUTHORED).save(authored)  # 1024 px
    generated = tmp_path / "generated" / "Metal_orm.png"
    generated.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 64), GENERATED).save(generated)  # 1024 px, other aspect

    _preserve_authored_orm(
        new_generated={"Metal": GeneratedTextures(orm=str(generated))},
        units=[_unit("Metal", orm_texture="./Textures/authored_orm.png")],
        context={"usd_path": str(usd_path)},
    )

    with Image.open(generated) as img:
        assert img.size in {(64, 16), (16, 64)}, (
            "must adopt one map's shape, not a hybrid"
        )
        assert img.size != (64, 64), "per-axis max would stretch both maps"


def test_authored_channels_survive_blending_at_default_opacity(tmp_path: Path) -> None:
    """The channels must be exact in what apply consumes, not just in generate output.

    apply_textures reads ``blended_textures``, and ``_blend_orm`` lerps every
    channel toward material constants at ``opacity`` (0.85 by default). Preserving
    before that step left authored metallic 249 arriving as ~212 and occlusion
    pulled back toward 255, while the diagnostic still claimed it was restored.
    Uses the real default opacity, not 1.0.
    """
    from texture_agent.tasks.blend_textures import BlendTexturesTask

    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_orm(tmp_path / "Textures" / "authored_orm.png", AUTHORED, size=8)
    generated = _write_orm(tmp_path / "generated" / "Metal_orm.png", GENERATED, size=8)
    albedo = tmp_path / "generated" / "Metal_albedo.png"
    Image.new("RGB", (8, 8), (60, 60, 60)).save(albedo)

    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    unit.opacity = 0.85  # the shipped default; 1.0 would hide the defect
    # base_metalness unset is the norm for MDL/OmniPBR assets, so the blend base
    # for metallic is 0.0 -- the worst case for a metal.
    assert unit.material_info.base_metalness is None

    result = BlendTexturesTask().run(
        {
            "prim_texture_units": [unit],
            "generated_textures": {
                "Metal": GeneratedTextures(albedo=str(albedo), orm=str(generated))
            },
            "working_dir": str(tmp_path),
            "usd_path": str(usd_path),
            "blend_config": {"output_size": 8},
        }
    )

    blended_orm = Path(result["blended_textures"]["Metal"].orm)
    occlusion, _roughness, metallic = _channels(blended_orm)
    assert occlusion == AUTHORED[0], f"authored occlusion lost in blend: {occlusion}"
    assert metallic == AUTHORED[2], f"authored metallic lost in blend: {metallic}"

    diagnostics = result.get("blend_textures_diagnostics", [])
    codes = [d["code"] for d in diagnostics]
    assert "AUTHORED_ORM_PRESERVED" in codes
    assert "AUTHORED_ORM_NOT_PRESERVED" not in codes
    assert all(diagnostic["stage"] == "blend_textures" for diagnostic in diagnostics)


def test_blend_preserves_native_generated_resolution_and_authored_orm(
    tmp_path: Path,
) -> None:
    """The full profile must not reduce Step1X's published 4096 maps to 1024."""
    from texture_agent.tasks.blend_textures import BlendTexturesTask

    usd_path = tmp_path / "asset.usda"
    usd_path.write_text("#usda 1.0\n", encoding="utf-8")
    _write_orm(tmp_path / "Textures" / "authored_orm.png", AUTHORED, size=32)
    generated_orm = _write_orm(
        tmp_path / "generated" / "Metal_orm.png",
        GENERATED,
        size=32,
    )
    generated_albedo = tmp_path / "generated" / "Metal_albedo.png"
    generated_normal = tmp_path / "generated" / "Metal_normal.png"
    Image.new("RGB", (32, 32), (60, 60, 60)).save(generated_albedo)
    Image.new("RGB", (32, 32), (128, 128, 255)).save(generated_normal)

    result = BlendTexturesTask().run(
        {
            "prim_texture_units": [
                _unit("Metal", orm_texture="./Textures/authored_orm.png")
            ],
            "generated_textures": {
                "Metal": GeneratedTextures(
                    albedo=str(generated_albedo),
                    normal=str(generated_normal),
                    orm=str(generated_orm),
                )
            },
            "working_dir": str(tmp_path),
            "usd_path": str(usd_path),
            "blend_config": {"output_size": 8},
            "texture_config": {
                "custom_parameters": {"preserve_generated_resolution": True}
            },
        }
    )

    blended = result["blended_textures"]["Metal"]
    for texture_path in (blended.albedo, blended.normal, blended.orm):
        with Image.open(texture_path) as image:
            assert image.size == (32, 32)
    assert _channels(Path(blended.orm)) == (
        AUTHORED[0],
        GENERATED[1],
        AUTHORED[2],
    )


@pytest.mark.parametrize("include_false_flag", [False, True])
def test_blend_uses_configured_size_without_resolution_opt_in(
    tmp_path: Path,
    include_false_flag: bool,
) -> None:
    from texture_agent.tasks.blend_textures import BlendTexturesTask

    generated_dir = tmp_path / "generated"
    generated_dir.mkdir()
    generated_albedo = generated_dir / "Metal_albedo.png"
    generated_normal = generated_dir / "Metal_normal.png"
    generated_orm = generated_dir / "Metal_orm.png"
    Image.new("RGB", (32, 32), (60, 60, 60)).save(generated_albedo)
    Image.new("RGB", (32, 32), (128, 128, 255)).save(generated_normal)
    Image.new("RGB", (32, 32), GENERATED).save(generated_orm)
    blend_config: dict[str, object] = {"output_size": 8}
    if include_false_flag:
        blend_config["preserve_generated_resolution"] = False

    result = BlendTexturesTask().run(
        {
            "prim_texture_units": [_unit("Metal", orm_texture=None)],
            "generated_textures": {
                "Metal": GeneratedTextures(
                    albedo=str(generated_albedo),
                    normal=str(generated_normal),
                    orm=str(generated_orm),
                )
            },
            "working_dir": str(tmp_path),
            "blend_config": blend_config,
        }
    )

    blended = result["blended_textures"]["Metal"]
    for texture_path in (blended.albedo, blended.normal, blended.orm):
        with Image.open(texture_path) as image:
            assert image.size == (8, 8)


@pytest.mark.parametrize("invalid_value", [None, 1, "true", {}])
def test_blend_rejects_invalid_resolution_flag_before_outputs(
    tmp_path: Path,
    invalid_value: object,
) -> None:
    from texture_agent.tasks.blend_textures import BlendTexturesTask

    with pytest.raises(ValueError, match="must be a boolean"):
        BlendTexturesTask().run(
            {
                "prim_texture_units": [],
                "generated_textures": {},
                "working_dir": str(tmp_path),
                "blend_config": {
                    "output_size": 8,
                    "preserve_generated_resolution": invalid_value,
                },
            }
        )

    assert not (tmp_path / "textures").exists()


def test_subset_bound_material_is_treated_as_unattributable(scene: dict) -> None:
    """A materialBind GeomSubset leaves bound_prim_paths empty in per_material mode.

    Without the subset paths there is nothing to attribute a UV mutation against,
    and an `any()` over an empty set would report "UVs did not move" — failing
    open on exactly the case the guard exists to catch.
    """
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    unit.material_info.bound_prim_paths = []
    unit.material_info.bound_subset_paths = []

    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "generated": 2,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/Mesh"],
            },
        },
    )

    assert _channels(scene["generated"]) == GENERATED, "guard failed open"


def test_subset_paths_attribute_the_mutation(scene: dict) -> None:
    """A subset-bound unit is matched through bound_subset_paths."""
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    unit.material_info.bound_prim_paths = []
    unit.material_info.bound_subset_paths = ["/World/Mesh/sub0"]

    _preserve_authored_orm(
        new_generated={"Metal": scene["textures"]},
        units=[unit],
        context={
            "usd_path": scene["usd_path"],
            "uv_preparation": {
                "generated": 2,
                "uv_scope": "target_prims",
                "target_prim_paths": ["/World/Mesh"],
            },
        },
    )

    assert _channels(scene["generated"]) == GENERATED, "subset overlap not detected"


def test_entry_without_an_orm_is_skipped(scene: dict) -> None:
    unit = _unit("Metal", orm_texture="./Textures/authored_orm.png")
    diagnostics = _preserve_authored_orm(
        new_generated={"Metal": GeneratedTextures(albedo="x.png", orm=None)},
        units=[unit],
        context={"usd_path": scene["usd_path"]},
    )
    assert diagnostics == []


def test_unknown_key_is_skipped(scene: dict) -> None:
    diagnostics = _preserve_authored_orm(
        new_generated={"NotAUnit": scene["textures"]},
        units=[_unit("Metal", orm_texture="./Textures/authored_orm.png")],
        context={"usd_path": scene["usd_path"]},
    )
    assert diagnostics == []
    assert _channels(scene["generated"]) == GENERATED


def test_coerce_prefers_resolved_path_but_falls_back_to_authored() -> None:
    """ORM paths must be layer-correct.

    ``Sdf.AssetPath.path`` is relative to the layer that authored it, not the
    stage root, so a material coming from a reference or sublayer would resolve
    against the wrong directory. The resolver's absolute path is layer-correct by
    construction; when it is empty (unresolvable asset) the authored spelling is
    still returned so the caller can report it.
    """
    from pxr import Sdf

    from texture_agent.functions.material_discovery import _coerce_texture_path

    unresolved = Sdf.AssetPath("./Textures/orm.png")
    assert (
        _coerce_texture_path(unresolved, prefer_resolved=True) == "./Textures/orm.png"
    )
    assert _coerce_texture_path(unresolved) == "./Textures/orm.png"

    resolved = Sdf.AssetPath("./Textures/orm.png", "/abs/Textures/orm.png")
    assert (
        _coerce_texture_path(resolved, prefer_resolved=True) == "/abs/Textures/orm.png"
    )
    # Without the flag the authored spelling is preserved for round-tripping.
    assert _coerce_texture_path(resolved) == "./Textures/orm.png"


def test_glt_f_metallic_roughness_is_not_treated_as_packed_orm() -> None:
    """glTF metallicRoughness has no occlusion in R; preserving it would write garbage.

    Pins the deliberate exclusion so a later contributor widening the set has to
    make that decision explicitly.
    """
    from texture_agent.functions.material_discovery import _ORM_TEXTURE_INPUTS

    for token in (
        "metallic_roughness",
        "metallicroughness",
        "metallic_roughness_texture",
        "metallicroughnesstexture",
    ):
        assert token not in _ORM_TEXTURE_INPUTS


@pytest.mark.parametrize("file_input", ["file", "filename"])
def test_orm_from_a_texture_reader_shader_is_discovered(
    tmp_path: Path, file_input: str
) -> None:
    """MaterialX / UsdPreviewSurface networks carry the path on a reader shader.

    The path arrives on a ``file``/``filename`` input of a texture-reader shader
    rather than on an input literally named ``orm_texture``, so the reader's own
    name is the only signal — the same asymmetry the albedo branch already
    handles via ``_is_albedo_texture_name``.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.functions.material_discovery import discover_materials

    stage = Usd.Stage.CreateNew(str(tmp_path / f"asset_{file_input}.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Mesh")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Metal")

    reader = UsdShade.Shader.Define(stage, "/World/Looks/Metal/orm_texture_reader")
    reader.CreateIdAttr("UsdUVTexture")
    reader.CreateInput(file_input, Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/authored_orm.png")
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Mesh")).Bind(material)
    stage.Save()

    (info,) = discover_materials(stage)
    assert info.orm_texture == "./Textures/authored_orm.png", (
        f"ORM on a reader '{file_input}' input was not classified"
    )


def test_albedo_reader_is_not_mistaken_for_orm(tmp_path: Path) -> None:
    """The reader-name heuristic must not swallow albedo readers."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.functions.material_discovery import discover_materials

    stage = Usd.Stage.CreateNew(str(tmp_path / "albedo_only.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Mesh")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Painted")
    reader = UsdShade.Shader.Define(stage, "/World/Looks/Painted/albedo_texture_reader")
    reader.CreateIdAttr("UsdUVTexture")
    reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/albedo.png")
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Mesh")).Bind(material)
    stage.Save()

    (info,) = discover_materials(stage)
    assert info.base_color_texture == "./Textures/albedo.png"
    assert info.orm_texture is None, "albedo reader must not be classified as ORM"


def test_later_resolved_alias_upgrades_an_authored_fallback(tmp_path: Path) -> None:
    """A resolver-backed candidate must win over an earlier authored-only one.

    _ORM_TEXTURE_INPUTS holds several aliases. If the first one encountered does
    not resolve (missing file), storing it and stopping would keep a
    layer-relative path that mis-resolves for referenced materials, even though a
    later alias resolves cleanly.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.functions.material_discovery import discover_materials

    (tmp_path / "Textures").mkdir(parents=True, exist_ok=True)
    _write_orm(tmp_path / "Textures" / "real_orm.png", AUTHORED, size=4)

    stage = Usd.Stage.CreateNew(str(tmp_path / "aliases.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Mesh")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Metal")
    prim = material.GetPrim()
    # USD visits "inputs:ORM_texture" before "inputs:ORMTexture" (verified
    # empirically, not assumed), so the unresolvable alias must be the former
    # for the upgrade path to be exercised.
    prim.CreateAttribute("inputs:ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/does_not_exist.png")
    )
    prim.CreateAttribute("inputs:ORMTexture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/real_orm.png")
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Mesh")).Bind(material)
    stage.Save()

    (info,) = discover_materials(stage)
    assert info.orm_texture is not None
    assert info.orm_texture.endswith("real_orm.png"), (
        f"resolved alias did not win: {info.orm_texture}"
    )
    assert Path(info.orm_texture).is_absolute(), "expected the resolver-backed path"


def test_shader_reader_resolved_alias_upgrades_fallback(tmp_path: Path) -> None:
    """Same upgrade rule on the shader-network path."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade

    from texture_agent.functions.material_discovery import discover_materials

    (tmp_path / "Textures").mkdir(parents=True, exist_ok=True)
    _write_orm(tmp_path / "Textures" / "real_orm.png", AUTHORED, size=4)

    stage = Usd.Stage.CreateNew(str(tmp_path / "reader_aliases.usda"))
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Sphere.Define(stage, "/World/Mesh")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = UsdShade.Material.Define(stage, "/World/Looks/Metal")
    shader = UsdShade.Shader.Define(stage, "/World/Looks/Metal/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("ORM_texture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/does_not_exist.png")
    )
    shader.CreateInput("ORMTexture", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("./Textures/real_orm.png")
    )
    UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Mesh")).Bind(material)
    stage.Save()

    (info,) = discover_materials(stage)
    assert info.orm_texture is not None
    assert info.orm_texture.endswith("real_orm.png")
    # endswith alone would also pass on the unresolved relative spelling,
    # which would not prove the upgrade happened at all.
    assert Path(info.orm_texture).is_absolute(), "expected the resolver-backed path"
