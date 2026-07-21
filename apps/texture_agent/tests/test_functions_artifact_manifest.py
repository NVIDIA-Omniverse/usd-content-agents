# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

pytest.importorskip("pxr")


def _write_textured_stage(output_usd: Path, texture_ref: str) -> None:
    from pxr import Sdf, Usd, UsdShade

    stage = Usd.Stage.CreateNew(str(output_usd))
    mat = UsdShade.Material.Define(stage, "/Root/Looks/Steel")
    mat.GetPrim().CreateAttribute(
        "inputs:base_color_texture_file", Sdf.ValueTypeNames.Asset
    ).Set(Sdf.AssetPath(texture_ref))
    stage.GetRootLayer().Save()


def _write_png(path: Path, color: tuple[int, int, int] = (10, 20, 30)) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)
    return str(path)


def _backend_diagnostic(
    code: str,
    *,
    severity: str = "warning",
    message: str = "diagnostic",
    details: dict | None = None,
) -> dict:
    return {
        "schema_version": "texture-agent-diagnostic.v1",
        "code": code,
        "severity": severity,
        "stage": "generate_textures",
        "prim_path": "/Root/Mesh",
        "material_name": "Aluminum_Matte",
        "message": message,
        "recommended_action": "Inspect backend artifacts.",
        "details": details or {},
    }


def test_selected_materials_coalesces_none_detail_policy_to_default() -> None:
    from texture_agent.functions.artifact_manifest import _selected_materials

    selected = _selected_materials(
        {
            "prim_texture_units": [
                SimpleNamespace(
                    key="Plastic_Green",
                    material_info=SimpleNamespace(
                        name="Plastic_Green",
                        prim_path="/Root/Looks/Plastic_Green",
                    ),
                    prim_path="/Root/PCB",
                    prompt="green material",
                    opacity=1.0,
                    detail_policy=None,
                    seed=None,
                )
            ]
        }
    )

    assert selected[0]["detail_policy"] == "default"


def test_planning_section_accepts_dict_payload_and_schema_errors(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        _planning_section,
        validate_artifacts_manifest_schema,
    )

    section = _planning_section(
        {
            "texture_plan": {
                "decision": {"state": "ready", "execution_allowed": True},
                "counts": {"selected_unit_count": 1},
                "limits": {"hard_cap": 64},
            }
        },
        tmp_path,
    )

    assert section["plan_available"] is True
    assert section["decision_state"] == "ready"
    assert section["execution_allowed"] is True
    plan_path = tmp_path / "texture_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "decision": {"state": "ready", "execution_allowed": True},
                "counts": {"selected_unit_count": 2},
                "limits": {"hard_cap": 64},
            }
        ),
        encoding="utf-8",
    )
    path_section = _planning_section({"texture_plan_path": str(plan_path)}, tmp_path)
    assert path_section["plan_available"] is True
    assert path_section["counts"] == {"selected_unit_count": 2}
    missing_section = _planning_section(
        {"texture_plan_path": str(tmp_path / "missing_plan.json")},
        tmp_path,
    )
    assert missing_section["plan_available"] is False
    assert "planning section must be an object" in validate_artifacts_manifest_schema(
        {"planning": "bad"}
    )
    assert "planning.texture_plan is required" in validate_artifacts_manifest_schema(
        {"planning": {"plan_available": True}}
    )


def test_artifacts_manifest_schema_contract(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
        write_artifacts_manifest,
    )

    cache = tmp_path / "session" / "cache"
    prepared_dir = cache / "prepared"
    output_dir = cache / "output"
    textures_dir = cache / "textures"
    prepared_dir.mkdir(parents=True)
    output_dir.mkdir()
    textures_dir.mkdir()

    input_usd = tmp_path / "session" / "input" / "scene.usda"
    input_usd.parent.mkdir()
    _write_textured_stage(input_usd, "")

    texture = textures_dir / "steel_albedo.png"
    Image.new("RGB", (8, 8), (10, 20, 30)).save(texture)
    output_usd = output_dir / "textured_output.usda"
    _write_textured_stage(output_usd, "../textures/steel_albedo.png")

    uv_report = prepared_dir / "uv_report.json"
    uv_report.write_text(
        json.dumps(
            {
                "schema_version": "texture-agent-uv-report.v1",
                "prepared_usd": str(input_usd),
                "summary": {"mesh_count": 1, "valid_count": 1},
            }
        ),
        encoding="utf-8",
    )

    context = {
        "working_dir": str(cache),
        "usd_path": str(input_usd),
        "config": {
            "input": {"usd_path": str(input_usd)},
            "project": {"name": "demo", "session_id": "sid"},
            "steps": {"render": {"enabled": False}},
        },
        "uv_preparation": {"uv_report_path": str(uv_report), "generated": 0},
        "material_textures": {"Steel": "brushed steel"},
        "prim_texture_units": [
            SimpleNamespace(
                key="Steel",
                material_info=SimpleNamespace(
                    name="Steel",
                    prim_path="/Root/Looks/Steel",
                ),
                prim_path="/Root/Mesh",
                prompt="brushed steel",
                opacity=0.7,
                detail_policy="surface_only",
                seed=None,
            )
        ],
        "prim_paths": ["/Root/Mesh"],
        "output_usd_paths": [str(output_usd)],
        "texture_config": {
            "backend": "service",
            "endpoint": "https://example.invalid",
            "size": 8,
            "custom_parameters": {"api_key": "SHOULD_NOT_APPEAR"},
        },
        "warnings": [],
        "timings": {"total": 1.2},
    }

    manifest = build_artifacts_manifest(
        context,
        status="completed",
        service_urls={"manifest": "/artifacts/sid/manifest"},
        duration_seconds=3,
    )

    assert validate_artifacts_manifest_schema(manifest) == []
    assert manifest["outputs"]["portability"]["portable"] is True
    assert manifest["prepared"]["uv_summary"] == {"mesh_count": 1, "valid_count": 1}
    assert manifest["backend"]["endpoint"] == "<configured>"
    assert manifest["backend"]["custom_parameters"]["api_key"] == "<redacted>"
    selected = manifest["materials"]["selected"][0]
    assert selected["detail_policy"] == "surface_only"
    assert manifest["prompts"]["units"][0]["detail_policy"] == "surface_only"

    manifest_path = write_artifacts_manifest(context, payload=manifest)
    reloaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert validate_artifacts_manifest_schema(reloaded) == []


def test_artifacts_manifest_projection_full_success_schema(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )
    from texture_agent.functions.texture_generation import (
        GeneratedTextures,
        MapArtifact,
    )

    cache = tmp_path / "session" / "cache"
    albedo = _write_png(cache / "generated" / "Aluminum_Matte_albedo.png")
    normal = _write_png(
        cache / "generated" / "Aluminum_Matte_normal.png", (128, 128, 255)
    )
    orm = _write_png(cache / "generated" / "Aluminum_Matte_orm.png", (255, 200, 0))
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "https://backend.invalid/v1",
            "size": 8,
        },
        "generated_textures": {
            "Aluminum_Matte": GeneratedTextures(
                albedo=albedo,
                normal=normal,
                orm=orm,
            )
        },
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {
                    "albedo": MapArtifact(uri=albedo, width=8, height=8),
                    "normal": MapArtifact(uri=normal, width=8, height=8),
                    "orm": MapArtifact(uri=orm, width=8, height=8, packing="ORM"),
                },
                "auxiliary_artifacts": {
                    "masks": {"uv_islands": "file:///tmp/uv_mask.png"},
                    "coverage": {"heatmap": "file:///tmp/coverage.png"},
                    "debug": {"preview": "file:///tmp/debug.png"},
                },
                "metadata": {
                    "backend_name": "fake_projection_backend",
                    "model": "fake-projection-v1",
                    "capabilities": {
                        "image_conditioning": True,
                        "normal_map": True,
                        "orm": True,
                        "masks": True,
                        "coverage": True,
                    },
                    "projection": {"method": "uv_projection"},
                    "editing": {"strength": 0.7, "seed": 116},
                    "coverage": {"target_coverage": 0.98},
                },
                "diagnostics": [],
                "variant_asset_uri": str(tmp_path / "variant.usd"),
                "variant_name": "Aluminum_Matte",
                "endpoint": "https://backend.invalid/v1",
            }
        },
    }

    manifest = build_artifacts_manifest(context, status="completed")

    assert validate_artifacts_manifest_schema(manifest) == []
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["channel_state"]["albedo"] == "present"
    assert projection["channel_state"]["normal"] == "present"
    assert projection["channel_state"]["orm"] == "present"
    assert projection["channel_state"]["roughness"] == "absent"
    assert projection["artifacts"]["masks"]["uv_islands"] == "file:///tmp/uv_mask.png"
    assert projection["artifacts"]["coverage"]["heatmap"] == "file:///tmp/coverage.png"
    assert projection["artifacts"]["debug"]["preview"] == "file:///tmp/debug.png"
    assert projection["capabilities"]["coverage"] is True
    assert projection["projection"] == {"method": "uv_projection"}
    assert projection["editing"] == {"strength": 0.7, "seed": 116}
    assert manifest["backend"]["projection"]["unit_count"] == 1
    assert (
        manifest["backend"]["projection"]["capabilities"]["Aluminum_Matte"][
            "normal_map"
        ]
        is True
    )


def test_artifacts_manifest_projection_degraded_success_schema(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )
    from texture_agent.functions.texture_generation import (
        GeneratedTextures,
        MapArtifact,
    )

    cache = tmp_path / "session" / "cache"
    albedo = _write_png(cache / "generated" / "Aluminum_Matte_albedo.png")
    normal = _write_png(
        cache / "generated" / "Aluminum_Matte_normal.png", (128, 128, 255)
    )
    orm = _write_png(cache / "generated" / "Aluminum_Matte_orm.png", (255, 200, 0))
    diagnostic = _backend_diagnostic(
        "BACKEND_MAP_MISSING",
        details={"missing_maps": ["normal", "orm", "roughness", "metalness"]},
    )
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {"backend": "service", "endpoint": "https://backend"},
        "generated_textures": {
            "Aluminum_Matte": GeneratedTextures(
                albedo=albedo,
                normal=normal,
                orm=orm,
            )
        },
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {"albedo": MapArtifact(uri=albedo, width=8, height=8)},
                "metadata": {
                    "capabilities": {"normal_map": False, "orm": False},
                    "degraded_channels": ["normal", "orm"],
                },
                "degraded_channels": ["normal", "orm"],
                "diagnostics": [diagnostic],
                "endpoint": "https://backend",
            }
        },
        "generate_textures_diagnostics": [diagnostic],
    }

    manifest = build_artifacts_manifest(context, status="completed")

    assert validate_artifacts_manifest_schema(manifest) == []
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["channel_state"]["normal"] == "synthesized_neutral"
    assert projection["channel_state"]["orm"] == "packed_from_channels_or_constants"
    assert projection["channel_state"]["roughness"] == "absent"
    assert projection["channel_state"]["metalness"] == "absent"
    assert projection["degraded_channels"] == ["normal", "orm"]
    assert projection["warnings"][0]["code"] == "BACKEND_MAP_MISSING"
    assert manifest["status"]["diagnostics"][0]["code"] == "BACKEND_MAP_MISSING"


def test_artifacts_manifest_projection_low_coverage_warning_schema(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )
    from texture_agent.functions.texture_generation import (
        GeneratedTextures,
        MapArtifact,
    )

    cache = tmp_path / "session" / "cache"
    albedo = _write_png(cache / "generated" / "Aluminum_Matte_albedo.png")
    diagnostic = _backend_diagnostic(
        "BACKEND_LOW_COVERAGE",
        message="Backend reported low target coverage.",
        details={"target_coverage": 0.42, "threshold": 0.75},
    )
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {"backend": "service", "endpoint": "https://backend"},
        "generated_textures": {
            "Aluminum_Matte": GeneratedTextures(
                albedo=albedo,
                normal="",
                orm="",
            )
        },
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {"albedo": MapArtifact(uri=albedo, width=8, height=8)},
                "auxiliary_artifacts": {
                    "coverage_mask": {
                        "uri": "file:///tmp/coverage_mask.png",
                        "coverage": 0.42,
                    }
                },
                "metadata": {
                    "capabilities": {"coverage": True},
                    "coverage": {"target_coverage": 0.42, "threshold": 0.75},
                },
                "diagnostics": [diagnostic],
                "endpoint": "https://backend",
            }
        },
        "generate_textures_diagnostics": [diagnostic],
    }

    manifest = build_artifacts_manifest(context, status="completed")

    assert validate_artifacts_manifest_schema(manifest) == []
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["coverage"] == {"target_coverage": 0.42, "threshold": 0.75}
    assert projection["artifacts"]["coverage"]["uri"] == "file:///tmp/coverage_mask.png"
    assert projection["warnings"][0]["code"] == "BACKEND_LOW_COVERAGE"
    assert (
        manifest["backend"]["projection"]["coverage"]["Aluminum_Matte"][
            "target_coverage"
        ]
        == 0.42
    )


def test_artifacts_manifest_projection_failed_partial_response_schema(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )

    cache = tmp_path / "session" / "cache"
    cache.mkdir(parents=True)
    diagnostic = _backend_diagnostic(
        "BACKEND_MAP_MISSING",
        severity="error",
        message="Backend did not return required albedo map.",
        details={"missing_maps": ["albedo"]},
    )
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {"backend": "service", "endpoint": "https://backend"},
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {},
                "metadata": {
                    "capabilities": {"normal_map": False, "orm": False},
                    "degraded_channels": ["albedo"],
                },
                "diagnostics": [diagnostic],
                "endpoint": "https://backend",
            }
        },
        "generate_textures_diagnostics": [diagnostic],
        "generate_textures_errors": [
            {
                "material": "Aluminum_Matte",
                "type": "RuntimeError",
                "message": "Backend did not return required albedo map.",
            }
        ],
    }

    manifest = build_artifacts_manifest(context, status="failed")

    assert validate_artifacts_manifest_schema(manifest) == []
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["map_count"] == 0
    assert projection["channel_state"]["albedo"] == "missing"
    assert projection["channel_state"]["normal"] == "absent"
    assert manifest["status"]["errors"]["generate_textures"][0]["material"] == (
        "Aluminum_Matte"
    )
    assert manifest["status"]["diagnostics"][0]["severity"] == "error"


def test_artifacts_manifest_redacts_fake_secret_everywhere(tmp_path: Path) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )
    from texture_agent.functions.texture_generation import MapArtifact

    cache = tmp_path / "session" / "cache"
    albedo = _write_png(cache / "generated" / "Aluminum_Matte_albedo.png")
    diagnostic = _backend_diagnostic(
        "BACKEND_MAP_MISSING",
        message=(
            "Backend rejected Bearer sk-FAKESECRET12345678 at "
            "https://backend.invalid?api_key=FAKESECRET"
        ),
        details={
            "endpoint": "https://backend.invalid?token=FAKESECRET",
            "debug_url": "https://debug.invalid?secret=FAKESECRET",
        },
    )
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {
            "backend": "service",
            "endpoint": "https://backend.invalid?api_key=FAKESECRET",
            "custom_parameters": {
                "api_key": "FAKESECRET",
                "nested": {"access_token": "FAKESECRET"},
            },
        },
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": {
                    "albedo": MapArtifact(
                        uri=f"file://{albedo}?token=FAKESECRET",
                        width=8,
                        height=8,
                    )
                },
                "auxiliary_artifacts": {
                    "debug": {"request": "https://debug.invalid?secret=FAKESECRET"}
                },
                "metadata": {
                    "endpoint": "https://backend.invalid?token=FAKESECRET",
                    "api_key": "FAKESECRET",
                    "notes": "Bearer sk-FAKESECRET12345678",
                },
                "diagnostics": [diagnostic],
                "endpoint": "https://backend.invalid?api_key=FAKESECRET",
            }
        },
        "generate_textures_diagnostics": [diagnostic],
        "generate_textures_errors": [
            {
                "material": "Aluminum_Matte",
                "message": "failed with token=FAKESECRET",
            }
        ],
    }

    manifest = build_artifacts_manifest(context, status="failed")
    serialized = json.dumps(manifest, sort_keys=True)

    assert validate_artifacts_manifest_schema(manifest) == []
    assert "FAKESECRET" not in serialized
    assert "sk-FAKESECRET" not in serialized
    assert manifest["backend"]["endpoint"] == "<configured>"
    assert manifest["backend"]["custom_parameters"]["api_key"] == "<redacted>"
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["endpoint"] == "<configured>"
    assert projection["metadata"]["endpoint"] == "<configured>"
    assert projection["metadata"]["api_key"] == "<redacted>"
    assert projection["diagnostics"][0]["details"]["endpoint"] == "<configured>"


def test_artifacts_manifest_projection_tolerates_malformed_backend_fields(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        build_artifacts_manifest,
        validate_artifacts_manifest_schema,
    )
    from texture_agent.functions.texture_generation import GeneratedTextures

    cache = tmp_path / "session" / "cache"
    albedo = _write_png(cache / "generated" / "Aluminum_Matte_albedo.png")
    context = {
        "working_dir": str(cache),
        "usd_path": str(tmp_path / "ladder.usd"),
        "texture_config": {"backend": "service"},
        "generated_textures": {
            "Aluminum_Matte": GeneratedTextures(albedo=albedo, normal="", orm="")
        },
        "projection_backend_results": {
            "Aluminum_Matte": {
                "maps": "malformed",
                "metadata": "malformed",
                "diagnostics": "malformed",
                "auxiliary_artifacts": "malformed",
                "degraded_channels": "malformed",
            }
        },
    }

    manifest = build_artifacts_manifest(context, status="completed")

    assert validate_artifacts_manifest_schema(manifest) == []
    projection = manifest["textures"]["projection_backend"]["Aluminum_Matte"]
    assert projection["maps"] == {}
    assert projection["diagnostics"] == []
    assert projection["channel_state"]["albedo"] == "present"
    assert manifest["backend"]["projection"]["unit_count"] == 1


def test_portability_validation_rejects_relative_paths_outside_bundle(
    tmp_path: Path,
) -> None:
    from texture_agent.functions.artifact_manifest import (
        validate_output_texture_portability,
    )

    cache = tmp_path / "session" / "cache"
    output_dir = cache / "output"
    output_dir.mkdir(parents=True)
    outside = tmp_path / "session" / "outside"
    outside.mkdir()
    Image.new("RGB", (4, 4), (1, 2, 3)).save(outside / "escape.png")

    output_usd = output_dir / "textured_output.usda"
    _write_textured_stage(output_usd, "../../outside/escape.png")

    portability = validate_output_texture_portability(output_usd)

    assert portability["portable"] is False
    assert portability["non_relative_texture_paths"] == ["../../outside/escape.png"]
    assert portability["diagnostics"][0]["code"] == "PACKAGE_ABSOLUTE_TEXTURE_PATH"
