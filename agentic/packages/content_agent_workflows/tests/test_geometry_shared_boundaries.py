# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import shutil
import types
from pathlib import Path
from typing import Any

import pytest
from cad_verifier.sim_ready import (
    INSERTION_FIXTURE_PROFILE_ID,
    STATIC_VISUAL_PROFILE_ID,
    resolve_official_simready_profile,
)
from PIL import Image, ImageDraw
from pydantic import SecretStr, ValidationError

from content_agent_workflows.geometry import audit as geometry_audit
from content_agent_workflows.geometry import rendering as rendering_module
from content_agent_workflows.geometry import scene_ops
from content_agent_workflows.geometry import workflow as geometry_workflow
from content_agent_workflows.geometry.rendering import (
    GeometryRenderEvidence,
    render_geometry_evidence,
)
from content_agent_workflows.geometry.validation import run_geometry_usd_validation
from content_agent_workflows.geometry.workflow import GeometryWorkflowInput
from content_agent_workflows.runtime_validation import workflow as runtime_workflow


def _tiny_usda(path: Path) -> Path:
    path.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset" {}
""",
        encoding="utf-8",
    )
    return path


def _executed_ovrtx_fields(
    mode: str = "rt2",
    sensor_updates: int = 32,
) -> dict[str, object]:
    return {
        "ovrtx_render_mode": mode,
        "ovrtx_num_sensor_updates": sensor_updates,
        "active_aov": "LdrColor",
    }


def _local_ovrtx_render_response(
    labels: list[str],
    image_paths: list[str],
    *,
    mode: str = "pt",
    sensor_updates: int = 64,
) -> dict[str, object]:
    return {
        "backend": "ovrtx",
        "cameras": labels,
        "results": [
            {
                "camera": label,
                "images": [path],
                **_executed_ovrtx_fields(mode, sensor_updates),
            }
            for label, path in zip(labels, image_paths, strict=True)
        ],
    }


def _retained_candidate_render_evidence(
    *,
    source: Path,
    image_path: Path,
    report_path: Path,
) -> rendering_module.GeometryRenderEvidence:
    source = source.resolve()
    image_path = image_path.resolve()
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    result = rendering_module.GeometryRenderEvidence(
        status="pass",
        backend="ovrtx",
        preset="hero",
        source_usd_path=str(source),
        source_usd_sha256=source_sha256,
        source_usd_sha256_after_render=source_sha256,
        ovrtx_render_mode="rt2",
        ovrtx_num_sensor_updates=32,
        active_aov="LdrColor",
        image_paths=[str(image_path)],
        image_bindings=[
            rendering_module.GeometryRenderImageBinding(
                path=str(image_path),
                sha256=image_sha256,
                role="view",
                view="hero",
                source_usd_sha256=source_sha256,
                ovrtx_render_mode="rt2",
                ovrtx_num_sensor_updates=32,
                active_aov="LdrColor",
            )
        ],
        shared_render_status="completed",
        metadata={
            "backend": "ovrtx",
            "image_count": 1,
            "response_cameras": ["hero"],
            "stage_preparation": [
                {
                    "usd_path": str(source),
                    "usd_sha256": source_sha256,
                }
            ],
            "usd_path_count": 1,
            "requested_ovrtx_render_mode": "rt2",
            "requested_ovrtx_num_sensor_updates": 32,
            "executed_ovrtx_settings": _executed_ovrtx_fields(),
            "executed_ovrtx_settings_verified": True,
            "active_aov": "LdrColor",
        },
        report_path=str(report_path.resolve()),
    )
    report_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return result


def test_geometry_defaults_do_not_claim_runtime_or_simready(tmp_path: Path) -> None:
    params = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "out",
    )

    assert params.target_profile == STATIC_VISUAL_PROFILE_ID
    assert params.runtime_validation_mode == "skip"
    assert params.simready_mode == "skip"
    assert params.run_runtime_validation is None


def test_workflow_forwards_explicit_ovrtx_quality(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "asset.usda")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    image_path = tmp_path / "right.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    image_sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()
    report = tmp_path / "render" / "geometry_render_evidence_six_view.json"
    captured: dict[str, object] = {}

    def fake_render_geometry_evidence(
        **kwargs: object,
    ) -> rendering_module.GeometryRenderEvidence:
        captured.update(kwargs)
        return rendering_module.GeometryRenderEvidence(
            status="pass",
            backend="ovrtx",
            preset="six_view",
            source_usd_path=str(source),
            source_usd_sha256=source_sha256,
            source_usd_sha256_after_render=source_sha256,
            ovrtx_render_mode="pt",
            ovrtx_num_sensor_updates=64,
            active_aov="LdrColor",
            image_paths=[str(image_path)],
            image_bindings=[
                rendering_module.GeometryRenderImageBinding(
                    path=str(image_path),
                    sha256=image_sha256,
                    role="view",
                    view="right",
                    source_usd_sha256=source_sha256,
                    ovrtx_render_mode="pt",
                    ovrtx_num_sensor_updates=64,
                    active_aov="LdrColor",
                )
            ],
            shared_render_status="completed",
            metadata={
                "requested_ovrtx_render_mode": "pt",
                "requested_ovrtx_num_sensor_updates": 64,
                "executed_ovrtx_settings": _executed_ovrtx_fields("pt", 64),
                "executed_ovrtx_settings_verified": True,
                "active_aov": "LdrColor",
            },
            report_path=str(report),
        )

    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        fake_render_geometry_evidence,
    )
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "out",
        render_evidence=True,
        render_ovrtx_mode="pt",
        render_ovrtx_num_sensor_updates=64,
    )

    check, failures, warnings, artifacts, preview_renders, _report_path = (
        geometry_workflow._render_evidence(
            usd_path=source,
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert captured["ovrtx_mode"] == "pt"
    assert captured["ovrtx_num_sensor_updates"] == 64
    assert check is not None and check.status == "pass"
    assert check.metadata["ovrtx_render_mode"] == "pt"
    assert check.metadata["ovrtx_num_sensor_updates"] == 64
    assert check.metadata["active_aov"] == "LdrColor"
    assert all(
        artifact.metadata["ovrtx_render_mode"] == "pt"
        and artifact.metadata["ovrtx_num_sensor_updates"] == 64
        and artifact.metadata["active_aov"] == "LdrColor"
        for artifact in artifacts
    )
    assert all(
        preview["ovrtx_render_mode"] == "pt"
        and preview["ovrtx_num_sensor_updates"] == 64
        and preview["active_aov"] == "LdrColor"
        for preview in preview_renders
    )
    assert failures == []
    assert warnings == []


def test_legacy_runtime_boolean_maps_to_explicit_proxy_mode(tmp_path: Path) -> None:
    enabled = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "enabled",
        run_runtime_validation=True,
    )
    disabled = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "disabled",
        run_runtime_validation=False,
    )

    assert enabled.runtime_validation_mode == "temporary_loadability_proxy"
    assert disabled.runtime_validation_mode == "skip"


def test_explicit_runtime_mode_takes_precedence_over_legacy_boolean(
    tmp_path: Path,
) -> None:
    explicit_skip = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "skip",
        runtime_validation_mode="skip",
        run_runtime_validation=True,
    )
    explicit_authored = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "authored",
        runtime_validation_mode="authored_physics",
        run_runtime_validation=False,
    )

    assert explicit_skip.runtime_validation_mode == "skip"
    assert explicit_authored.runtime_validation_mode == "authored_physics"


@pytest.mark.parametrize(
    ("policy", "flatten", "deinstance", "deduplicate"),
    [
        ("preserve_correspondence", False, False, False),
        ("runtime_efficiency", True, True, True),
    ],
)
def test_optimizer_policy_uses_shared_scene_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    policy: str,
    flatten: bool,
    deinstance: bool,
    deduplicate: bool,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    output = tmp_path / f"{policy}.usda"
    seen: dict[str, Any] = {}

    def fake_run(_self: object, context: dict[str, Any]) -> dict[str, Any]:
        seen.update(context["optimization_config"])
        shutil.copy2(context["input_usd_path"], context["output_usd_path"])
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {"correspondence_map": {}},
        }

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", fake_run)
    monkeypatch.setattr(
        scene_ops,
        "_geometry_fidelity_check",
        lambda _source, _output: {"status": "pass", "passed": True},
    )

    result = scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy=policy,  # type: ignore[arg-type]
    )

    settings = seen["scene_optimizer_settings"]
    assert seen["flatten_prototypes"] is flatten
    assert settings["enable_deinstance"] is deinstance
    assert settings["enable_split_meshes"] is True
    assert settings["enable_deduplicate"] is deduplicate
    assert result["status"] == "completed"


def test_optimizer_rejects_output_with_renderable_bound_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
)
def Mesh "Asset"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "optimized.usda"

    def fake_run(_self: object, context: dict[str, Any]) -> dict[str, Any]:
        output_path = Path(context["output_usd_path"])
        output_path.write_text(
            source.read_text(encoding="utf-8").replace("(1, 0, 0)", "(3, 0, 0)"),
            encoding="utf-8",
        )
        return {
            **context,
            "optimization_success": True,
            "optimization_metadata": {},
        }

    monkeypatch.setattr(scene_ops.OptimizeUSDTask, "run", fake_run)

    result = scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="preserve_correspondence",
    )

    assert result["status"] == "fidelity_fallback"
    assert result["artifact_role"] == "normalized_copy"
    assert result["geometry_fidelity"]["status"] == "fail"
    assert Path(result["rejected_output_usd"]).is_file()
    assert output.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")


def test_exact_optimizer_fidelity_skips_redundant_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pxr import Usd

    source = _tiny_usda(tmp_path / "source.usda")
    output = tmp_path / "output.usda"
    shutil.copy2(source, output)
    stage = Usd.Stage.Open(str(output))
    stage.GetRootLayer().customLayerData = {"optimizer": "exact-packaging-only"}
    stage.GetRootLayer().Save()
    monkeypatch.setattr(
        scene_ops,
        "_renderable_bounds_m",
        lambda _path: pytest.fail(
            "exact geometry must not recompute renderable bounds"
        ),
    )

    result = scene_ops._geometry_fidelity_check(source, output)

    assert result["passed"] is True
    assert result["exact_world_geometry_match"] is True
    assert result["source"]["status"] == "proven_equal"
    assert result["output"]["status"] == "proven_equal"


def test_optimizer_rejects_unsupported_legacy_operation_names(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    with pytest.raises(ValueError, match="Unsupported Scene Optimizer"):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=tmp_path / "output.usda",
            optimization_config={"scene_optimizer_settings": {"meshCleanup": True}},
        )

    with pytest.raises(ValueError, match="Unsupported Scene Optimizer"):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=tmp_path / "top-level-output.usda",
            optimization_config={"decimateMeshes": True},
        )


def test_optimizer_rejects_in_place_shared_optimization(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    original = source.read_bytes()

    with pytest.raises(ValueError, match="distinct output path"):
        scene_ops.optimize_geometry(source_usd=source, output_usd=source)

    assert source.read_bytes() == original


def test_skip_normalization_flattens_relocated_composed_usd(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    child = tmp_path / "child.usda"
    child.write_text(
        """#usda 1.0
def Xform "Referenced"
{
    def Cube "Body"
    {
    }
}
""",
        encoding="utf-8",
    )
    source = tmp_path / "source.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset" (
    prepend references = @child.usda@</Referenced>
)
{
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "elsewhere" / "normalized.usdc"

    result = scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )
    child.unlink()

    stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert stage is not None
    assert stage.GetPrimAtPath("/Asset/Body").IsValid()
    assert (
        sum(
            not layer.anonymous for layer in stage.GetUsedLayers(includeClipLayers=True)
        )
        == 1
    )
    assert result["status"] == "skipped"


@pytest.mark.parametrize("source_format", ["usda", "usdc"])
def test_skip_flattens_relocated_generic_usd_without_changing_encoding(
    tmp_path: Path,
    source_format: str,
) -> None:
    from pxr import Sdf, Usd

    child = tmp_path / "child.usda"
    child.write_text(
        """#usda 1.0
def Xform "Referenced"
{
    custom string marker = "child"
}
""",
        encoding="utf-8",
    )
    source = tmp_path / f"source.{source_format}"
    if source_format == "usda":
        source.write_text(
            """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @child.usda@</Referenced>
)
{
}
""",
            encoding="utf-8",
        )
    else:
        stage = Usd.Stage.CreateNew(str(source))
        root = stage.DefinePrim("/Asset", "Xform")
        stage.SetDefaultPrim(root)
        root.GetReferences().AddReference("child.usda", "/Referenced")
        assert stage.GetRootLayer().Save()

    output = tmp_path / "elsewhere" / "normalized.usd"
    scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )

    child.unlink()
    layer = Sdf.Layer.OpenAsAnonymous(str(output))
    assert layer is not None
    assert (output.read_bytes()[:8] == b"PXR-USDC") is (source_format == "usdc")
    output_stage = Usd.Stage.Open(str(output), load=Usd.Stage.LoadAll)
    assert output_stage is not None
    marker = output_stage.GetPrimAtPath("/Asset").GetAttribute("marker")
    assert marker.Get() == "child"


def test_skip_normalization_rejects_unresolved_composition_dependencies(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
)
def Xform "Asset" (
    prepend references = @missing.usda@</Missing>
)
{
}
""",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unresolved dependencies"):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=tmp_path / "normalized.usdc",
            policy="skip",
        )


def test_in_place_digest_mismatch_never_deletes_source(tmp_path: Path) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    original = source.read_bytes()

    with pytest.raises(ValueError, match="changed before optimization"):
        scene_ops.optimize_geometry(
            source_usd=source,
            output_usd=source,
            policy="skip",
            expected_source_sha256="0" * 64,
        )

    assert source.read_bytes() == original


def test_unknown_optimizer_status_cannot_be_claimed_as_pass() -> None:
    check, warnings, _artifacts = geometry_workflow._optimization_check(
        {"status": "mystery", "artifact_role": "unknown"}
    )

    assert check.status == "warning"
    assert warnings
    assert "not claimed as optimized" in warnings[0]


def test_completed_optimizer_degradation_is_not_labeled_unavailable() -> None:
    check, warnings, _artifacts = geometry_workflow._optimization_check(
        {
            "status": "completed",
            "artifact_role": "optimized_geometry",
            "degraded_reason": "deduplication skipped",
        }
    )

    assert check.status == "warning"
    assert check.summary == "Shared OptimizeUSDTask completed with degradation."
    assert warnings == [
        "Shared Scene Optimizer completed with a reported degradation: "
        "deduplication skipped"
    ]


def test_unavailable_usd_cli_inspection_is_a_visible_warning(tmp_path: Path) -> None:
    report_path = tmp_path / "geometry_inspection.json"
    report_path.write_text("{}\n", encoding="utf-8")

    check, warnings, artifacts = geometry_workflow._inspection_check(
        {
            "status": "inspection_unavailable",
            "candidate_count": 0,
            "error": "pxr unavailable",
        },
        report_path,
    )

    assert check.status == "warning"
    assert warnings == [
        "Shared usd-cli mesh inspection was unavailable: pxr unavailable"
    ]
    assert artifacts[0].metadata["status"] == "unavailable"


def test_geometry_workflow_never_overwrites_existing_usd_source(
    tmp_path: Path,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    original = source.read_bytes()

    result = geometry_workflow.run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out",
            output_usd_path=source,
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )

    assert source.read_bytes() == original
    assert result.source_usd_path == str(source)
    assert result.geometry_usd_path != str(source)
    assert Path(result.geometry_usd_path or "").is_file()


def test_handoff_manifest_reports_actual_preserved_usd_units(tmp_path: Path) -> None:
    source = tmp_path / "centimeter-y-up.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 0.01
    upAxis = "Y"
)
def Xform "Asset" {}
""",
        encoding="utf-8",
    )

    result = geometry_workflow.run_geometry_workflow(
        GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out",
            optimization_policy="skip",
            run_shared_usd_validation=False,
            run_asset_audit=False,
        )
    )
    manifest = json.loads(
        Path(result.handoff_manifest_path or "").read_text(encoding="utf-8")
    )

    assert manifest["units"] == {"up_axis": "Y", "meters_per_unit": 0.01}


def test_composed_metric_normalization_authors_z_up_meters_and_default_prim(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "centimeter-y-up-no-default.usda"
    source.write_text(
        """#usda 1.0
(
    metersPerUnit = 0.01
    upAxis = "Y"
)
def Xform "Asset"
{
    def Cube "Body"
    {
        double size = 100
    }
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "geometry.usdc"
    scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )

    result = scene_ops.canonicalize_usd_stage_metrics(
        output,
        tmp_path / "geometry_stage_metrics.json",
    )

    stage = Usd.Stage.Open(str(output))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/Asset"
    assert str(UsdGeom.GetStageUpAxis(stage)).upper() == "Z"
    assert float(UsdGeom.GetStageMetersPerUnit(stage)) == pytest.approx(1.0)
    assert result["changes"] == [
        "set_default_prim:/Asset",
        "normalize_up_axis:Y->Z",
        "normalize_meters_per_unit:0.01->1",
    ]
    assert output.read_bytes()[:8] == b"PXR-USDC"


def test_metric_normalization_compensates_default_and_sibling_roots(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "multi-root-with-default.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Primary"
    metersPerUnit = 0.01
    upAxis = "Y"
)
def Xform "Primary" {}
def Xform "Sibling" {}
""",
        encoding="utf-8",
    )

    result = scene_ops.canonicalize_usd_stage_metrics(
        source,
        tmp_path / "geometry_stage_metrics.json",
    )

    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    assert stage.GetDefaultPrim().GetPath() == "/Primary"
    assert result["normalized_root_branches"] == ["/Primary", "/Sibling"]
    for prim_path in result["normalized_root_branches"]:
        op_names = [
            str(op.GetOpName())
            for op in UsdGeom.Xformable(
                stage.GetPrimAtPath(prim_path)
            ).GetOrderedXformOps()
        ]
        assert op_names[0] == "xformOp:transform:geometryStageMetrics"


def test_metric_normalization_preserves_unauthored_default_for_multi_root_stage(
    tmp_path: Path,
) -> None:
    from pxr import Usd

    source = tmp_path / "multi-root-no-default.usda"
    source.write_text(
        """#usda 1.0
(
    metersPerUnit = 0.001
    upAxis = "Z"
)
def Xform "Left" {}
def Xform "Right" {}
""",
        encoding="utf-8",
    )

    result = scene_ops.canonicalize_usd_stage_metrics(
        source,
        tmp_path / "geometry_stage_metrics.json",
    )

    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    assert not stage.GetDefaultPrim().IsValid()
    assert result["default_prim"] is None
    assert result["normalized_root_branches"] == ["/Left", "/Right"]
    for prim_path in result["normalized_root_branches"]:
        assert stage.GetPrimAtPath(prim_path).HasAttribute(
            "xformOp:transform:geometryStageMetrics"
        )


def test_skip_converts_binary_crate_to_real_usda(tmp_path: Path) -> None:
    from pxr import Usd

    source = tmp_path / "source.usdc"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
    stage.GetRootLayer().Save()
    output = tmp_path / "output.usda"

    scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )

    assert output.read_bytes()[:8] != b"PXR-USDC"
    assert output.read_text(encoding="utf-8").startswith("#usda")
    assert Usd.Stage.Open(str(output)) is not None


def test_composed_export_refreshes_cached_output_layer_after_replace(
    tmp_path: Path,
) -> None:
    from pxr import Sdf, Usd

    source = tmp_path / "source.usda"
    source_stage = Usd.Stage.CreateNew(str(source))
    source_root = source_stage.DefinePrim("/Fresh", "Xform")
    source_stage.SetDefaultPrim(source_root)
    assert source_stage.GetRootLayer().Save()

    output = tmp_path / "output.usda"
    stale_stage = Usd.Stage.CreateNew(str(output))
    stale_root = stale_stage.DefinePrim("/Stale", "Xform")
    stale_stage.SetDefaultPrim(stale_root)
    assert stale_stage.GetRootLayer().Save()
    cached_output_layer = Sdf.Layer.FindOrOpen(str(output))
    assert cached_output_layer is not None
    assert cached_output_layer.defaultPrim == "Stale"

    scene_ops._export_composed_usd(source, output, file_format="usda")

    assert cached_output_layer.defaultPrim == "Fresh"
    published_layer = Sdf.Layer.OpenAsAnonymous(str(output))
    assert published_layer is not None
    assert published_layer.defaultPrim == "Fresh"


def test_skip_preserves_dependency_free_binary_usdc_bytes(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "source.usdc"
    stage = Usd.Stage.CreateNew(str(source))
    stage.DefinePrim("/Asset", "Xform")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/Asset"))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.GetRootLayer().Save()
    source_bytes = source.read_bytes()
    output = tmp_path / "elsewhere" / "output.usdc"

    scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )

    assert output.read_bytes() == source_bytes


def test_missing_texture_does_not_block_composed_geometry_copy(tmp_path: Path) -> None:
    from pxr import Usd

    source = tmp_path / "missing-texture.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Asset"
{
    def Shader "Surface"
    {
        asset inputs:file = @missing-albedo.png@
    }
}
""",
        encoding="utf-8",
    )
    output = tmp_path / "elsewhere" / "geometry.usdc"

    result = scene_ops.optimize_geometry(
        source_usd=source,
        output_usd=output,
        policy="skip",
    )

    assert result["status"] == "skipped"
    assert result["missing_asset_dependencies"] == ["missing-albedo.png"]
    check, warnings, _artifacts = geometry_workflow._optimization_check(result)
    assert check.status == "warning"
    assert warnings == [
        "Geometry preserved unresolved appearance dependencies for downstream "
        "material/render repair: missing-albedo.png"
    ]
    assert Usd.Stage.Open(str(output)) is not None


def test_fail_on_validation_error_raises_after_writing_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    def failed_preflight(
        usd_path: Path,
        output_dir: Path,
        profile_id: str,
        **_kwargs: object,
    ) -> tuple[
        list[geometry_workflow.ValidationCheck],
        list[str],
        list[str],
        list[geometry_workflow.EvidenceArtifact],
        Path,
    ]:
        del usd_path, profile_id
        check = geometry_workflow.ValidationCheck(
            name="cad_preflight.mesh_topology",
            status="fail",
            summary="forced failure",
            failures=["forced topology failure"],
        )
        return (
            [check],
            ["forced topology failure"],
            [],
            [],
            output_dir / "cad_preflight_report.json",
        )

    monkeypatch.setattr(geometry_workflow, "_cad_preflight_checks", failed_preflight)
    output_dir = tmp_path / "out"

    with pytest.raises(RuntimeError, match="Geometry validation failed"):
        geometry_workflow.run_geometry_workflow(
            GeometryWorkflowInput(
                source_path=source,
                output_dir=output_dir,
                optimization_policy="skip",
                run_shared_usd_validation=False,
                run_asset_audit=False,
                fail_on_validation_error=True,
            )
        )

    assert (output_dir / "geometry_validation_evidence.json").is_file()
    assert (output_dir / "content_agents_manifest.json").is_file()


def test_authored_runtime_mode_never_creates_proxy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    monkeypatch.setattr(
        runtime_workflow, "_has_authored_rigid_body", lambda _path: False
    )
    monkeypatch.setattr(
        runtime_workflow,
        "_prepare_temporary_proxy",
        lambda *_args, **_kwargs: pytest.fail("proxy path must not run"),
    )

    result = runtime_workflow.run_runtime_validation(
        runtime_workflow.RuntimeValidationRequest(
            asset_path=source,
            output_dir=tmp_path / "runtime",
            mode="authored_physics",
            engine="fake",
        )
    )

    assert result.status == "fail"
    assert result.temporary_proxy_used is False
    assert "requires an authored rigid body" in result.failures[0]


def test_shared_render_adapter_validates_individual_views_and_builds_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_paths: list[str] = []
    for index in range(6):
        path = tmp_path / f"view-{index}.png"
        image = Image.new("RGB", (320, 320), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((60 + index, 55, 260, 270), fill="white")
        image.save(path)
        image_paths.append(str(path))

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        assert _kwargs["policy"]["runtime_render_views"] == ["review_6"]
        assert _kwargs["policy"]["render_studio_lighting"] is True
        assert _kwargs["policy"]["render_studio_dome"] is True
        assert _kwargs["policy"]["render_studio_dome_intensity"] == 20.0
        assert _kwargs["policy"]["render_studio_reflection_safe"] is True
        assert _kwargs["policy"]["render_studio_hdri_intensity"] == 600.0
        assert _kwargs["policy"]["render_studio_reflection_intensity"] == 100.0
        assert _kwargs["policy"]["render_ovrtx_mode"] == "pt"
        assert _kwargs["policy"]["render_ovrtx_num_sensor_updates"] == 64
        labels = ["right", "left", "front", "back", "top", "iso"]
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                labels,
                image_paths,
                mode="pt",
                sensor_updates=64,
            ),
            "metadata": {"response_cameras": labels},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
        studio_lighting=True,
        studio_dome=True,
        studio_dome_intensity=20.0,
        studio_reflection_safe=True,
        studio_hdri_intensity=600.0,
        studio_reflection_intensity=100.0,
        ovrtx_mode="pt",
        ovrtx_num_sensor_updates=64,
    )

    assert result.status == "pass"
    assert (
        result.schema_version
        == rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
    )
    assert result.requested_backend == "ovrtx"
    assert result.backend == "ovrtx"
    assert result.renderer == "ovrtx"
    assert result.renderer_identity_verified is True
    assert result.renderer_identity_evidence["source"] == "explicit_local_backend"
    assert len(result.image_validation) == 6
    assert all(item["passed"] for item in result.image_validation)
    assert result.response_validation["passed"] is True
    assert Path(result.presentation_image_path or "").is_file()
    assert Path(result.report_path or "").is_file()
    assert Path(result.report_path or "").name == (
        "geometry_render_evidence_six_view.json"
    )
    assert (
        geometry_audit._audit_ovrtx_metadata(
            Path(result.report_path or ""),
            True,
            render_image_path=Path(result.image_paths[0]),
        )
        == []
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    assert result.source_usd_path == str(source.resolve())
    assert result.source_usd_sha256 == source_sha256
    assert result.source_usd_sha256_after_render == source_sha256
    assert len(result.image_bindings) == 7
    assert {binding.role for binding in result.image_bindings} == {
        "view",
        "presentation",
    }
    assert all(
        binding.source_usd_sha256 == source_sha256 for binding in result.image_bindings
    )
    assert all(
        binding.sha256 == hashlib.sha256(Path(binding.path).read_bytes()).hexdigest()
        for binding in result.image_bindings
    )
    assert result.ovrtx_render_mode == "pt"
    assert result.ovrtx_num_sensor_updates == 64
    assert result.active_aov == "LdrColor"
    assert all(
        binding.ovrtx_render_mode == "pt"
        and binding.ovrtx_num_sensor_updates == 64
        and binding.active_aov == "LdrColor"
        for binding in result.image_bindings
    )
    report = json.loads(Path(result.report_path or "").read_text(encoding="utf-8"))
    assert report["source_usd_sha256"] == source_sha256
    assert len(report["image_bindings"]) == 7
    assert report["ovrtx_render_mode"] == "pt"
    assert report["ovrtx_num_sensor_updates"] == 64
    assert report["active_aov"] == "LdrColor"
    assert report["metadata"]["ovrtx_render_mode"] == "pt"
    assert report["metadata"]["ovrtx_num_sensor_updates"] == 64
    assert report["metadata"]["active_aov"] == "LdrColor"
    assert report["metadata"]["executed_ovrtx_settings_verified"] is True
    assert report["metadata"]["executed_ovrtx_settings"] == {
        "ovrtx_render_mode": "pt",
        "ovrtx_num_sensor_updates": 64,
        "active_aov": "LdrColor",
    }

    with pytest.raises(ValueError, match="must be positive"):
        render_geometry_evidence(
            usd_path=source,
            output_dir=tmp_path / "invalid-quality",
            preset="hero",
            ovrtx_num_sensor_updates=0,
        )

    unavailable = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="turntable",
    )
    assert Path(unavailable.report_path or "").name == (
        "geometry_render_evidence_turntable.json"
    )
    assert unavailable.report_path != result.report_path
    assert Path(result.report_path or "").is_file()
    unavailable_signals = geometry_audit._audit_ovrtx_metadata(
        Path(unavailable.report_path or ""),
        True,
    )
    assert [signal.code for signal in unavailable_signals] == [
        "render.required_ovrtx_unsuccessful"
    ]


def test_six_view_framing_audit_does_not_split_an_individual_render(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "individual-view.png"
    image = Image.new("RGB", (1024, 1024), "black")
    ImageDraw.Draw(image).rectangle((85, 383, 812, 640), fill="white")
    image.save(image_path)

    assert (
        geometry_audit._audit_image_framing(
            image_path,
            width=1024,
            height=1024,
        )
        == []
    )


def test_shared_render_uses_declared_forward_axis_for_semantic_views(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pxr import Usd

    source = _tiny_usda(tmp_path / "minus-y-forward.usda")
    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    custom_data = dict(stage.GetRootLayer().customLayerData)
    custom_data["geometrySourceCoordinateSystem"] = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "forward_axis": "-Y",
        "handedness": "right",
    }
    stage.GetRootLayer().customLayerData = custom_data
    assert stage.GetRootLayer().Save()

    image_paths: list[str] = []
    for index in range(6):
        path = tmp_path / f"semantic-view-{index}.png"
        image = Image.new("RGB", (320, 320), "black")
        ImageDraw.Draw(image).rectangle((55, 50, 265, 275), fill="white")
        image.save(path)
        image_paths.append(str(path))

    expected_directions = ["-x", "+x", "-y", "+y", "+z", "-x-y+z"]

    def fake_render(**kwargs: object) -> dict[str, Any]:
        policy = kwargs["policy"]
        assert isinstance(policy, dict)
        assert policy["runtime_render_views"] == expected_directions
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                expected_directions,
                image_paths,
                mode="pt",
                sensor_updates=64,
            ),
            "metadata": {"response_cameras": expected_directions},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
        ovrtx_mode="pt",
        ovrtx_num_sensor_updates=64,
    )

    assert result.status == "pass"
    assert result.metadata["semantic_view_plan"] == {
        "status": "resolved",
        "coordinate_source": "geometrySourceCoordinateSystem",
        "coordinate_system": {
            "up_axis": "Z",
            "forward_axis": "-Y",
            "handedness": "right",
        },
        "labels": ["right", "left", "front", "back", "top", "iso"],
        "directions": expected_directions,
    }


def test_metric_normalization_retains_canonical_forward_axis(tmp_path: Path) -> None:
    from pxr import Usd

    source = tmp_path / "centimeter-y-up.usda"
    source.write_text(
        """#usda 1.0
(
    defaultPrim = "Asset"
    metersPerUnit = 0.01
    upAxis = "Y"
    customLayerData = {
        dictionary geometrySourceCoordinateSystem = {
            string forward_axis = "-Z"
            string handedness = "right"
            double meters_per_unit = 0.01
            string up_axis = "Y"
        }
    }
)
def Xform "Asset" {}
""",
        encoding="utf-8",
    )

    result = scene_ops.canonicalize_usd_stage_metrics(
        source,
        tmp_path / "geometry_stage_metrics.json",
    )

    stage = Usd.Stage.Open(str(source))
    assert stage is not None
    expected = {
        "meters_per_unit": 1.0,
        "up_axis": "Z",
        "forward_axis": "+Y",
        "handedness": "right",
    }
    assert (
        stage.GetRootLayer().customLayerData["geometryCanonicalCoordinateSystem"]
        == expected
    )
    assert result["canonical_coordinate_system"] == expected


def test_geometry_render_evidence_rejects_unverified_backend_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                ["hero"], [str(image_path)]
            ),
            "metadata": {"response_cameras": ["hero"]},
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        ovrtx_mode="pt",
        ovrtx_num_sensor_updates=64,
    )

    assert result.status == "fail"
    assert result.backend == "remote"
    assert any(
        issue["code"] == "render.ovrtx_provenance_unverified"
        for issue in result.shared_render_issues
    )


def test_shared_render_fails_when_source_usd_changes_during_render(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        source.write_text('#usda 1.0\ndef Xform "Changed" {}\n', encoding="utf-8")
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                ["hero"], [str(image_path)]
            ),
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
    )

    assert result.status == "fail"
    assert result.source_usd_sha256 == source_sha256
    assert result.source_usd_sha256_after_render != source_sha256
    assert result.image_bindings[0].source_usd_sha256 == source_sha256
    assert any(
        issue["code"] == "render.source_usd_changed"
        for issue in result.shared_render_issues
    )


def test_shared_render_writes_fail_closed_evidence_for_missing_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_render(**_kwargs: object) -> dict[str, Any]:
        pytest.fail("a missing source must not reach the renderer")

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fail_render,
    )

    result = render_geometry_evidence(
        usd_path=tmp_path / "missing.usda",
        output_dir=tmp_path / "render",
        preset="turntable",
    )

    assert result.status == "fail"
    assert result.source_usd_sha256 is None
    assert Path(result.report_path or "").is_file()
    assert result.shared_render_issues[0]["code"] == (
        "render.source_usd_unavailable_before_render"
    )


def test_render_evidence_rejects_null_source_digest_for_passing_report() -> None:
    with pytest.raises(ValueError, match="may be null only"):
        rendering_module.GeometryRenderEvidence(
            status="pass",
            backend="ovrtx",
            preset="hero",
            source_usd_path="/tmp/source.usda",
            source_usd_sha256=None,
            ovrtx_render_mode="rt2",
            ovrtx_num_sensor_updates=32,
        )


def test_geometry_render_evidence_v3_requires_recorded_ovrtx_settings() -> None:
    payload = {
        "schema_version": rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION,
        "status": "unavailable",
        "backend": "ovrtx",
        "preset": "turntable",
        "source_usd_path": "/tmp/source.usda",
        "source_usd_sha256": "0" * 64,
        "source_usd_sha256_after_render": "0" * 64,
    }

    with pytest.raises(ValidationError) as missing_settings:
        rendering_module.GeometryRenderEvidence.model_validate(payload)

    missing_locations = {error["loc"] for error in missing_settings.value.errors()}
    assert ("ovrtx_render_mode",) in missing_locations
    assert ("ovrtx_num_sensor_updates",) in missing_locations

    legacy_payload = {
        **payload,
        "schema_version": "content-agent-workflows.geometry-render-evidence.v2",
        "ovrtx_render_mode": "pt",
        "ovrtx_num_sensor_updates": 64,
    }
    with pytest.raises(ValidationError) as legacy_schema:
        rendering_module.GeometryRenderEvidence.model_validate(legacy_payload)
    assert ("schema_version",) in {
        error["loc"] for error in legacy_schema.value.errors()
    }


def test_shared_render_source_mutation_overrides_renderer_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    def unavailable_render(**_kwargs: object) -> dict[str, Any]:
        source.unlink()
        return {
            "status": "unavailable",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                ["hero"], [str(image_path)]
            ),
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        unavailable_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
    )

    assert result.shared_render_status == "unavailable"
    assert result.status == "fail"
    assert any(
        issue["code"] == "render.source_usd_unavailable_after_render"
        for issue in result.shared_render_issues
    )


def test_shared_render_unavailable_is_not_a_derived_view_count_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "unavailable",
            "backend": "ovrtx",
            "image_paths": [],
            "issues": [
                {
                    "code": "render.renderer_unavailable",
                    "severity": "warn",
                    "message": "No shared renderer is configured.",
                    "subject": str(source),
                    "details": {},
                }
            ],
            "render_response": None,
            "metadata": {},
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == "unavailable"
    assert result.shared_render_status == "unavailable"
    assert not any(
        issue["code"] == "render.view_count_mismatch"
        for issue in result.shared_render_issues
    )


def test_shared_render_fails_when_source_usd_changes_during_grid_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    labels = ["right", "left", "front", "back", "top", "iso"]
    image_paths: list[str] = []
    for label in labels:
        image_path = tmp_path / f"{label}.png"
        image = Image.new("RGB", (320, 320), "black")
        ImageDraw.Draw(image).rectangle((55, 50, 265, 275), fill="white")
        image.save(image_path)
        image_paths.append(str(image_path))

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(labels, image_paths),
            "metadata": {"response_cameras": labels},
        }

    original_compose_grid = rendering_module._compose_grid

    def mutate_source_then_compose(*args: object, **kwargs: object) -> Path:
        source.write_text('#usda 1.0\ndef Xform "Changed" {}\n', encoding="utf-8")
        return original_compose_grid(*args, **kwargs)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering._compose_grid",
        mutate_source_then_compose,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == "fail"
    assert result.source_usd_sha256 == source_sha256
    assert result.source_usd_sha256_after_render != source_sha256
    assert any(
        issue["code"] == "render.source_usd_changed"
        for issue in result.shared_render_issues
    )


def test_shared_render_fails_when_view_changes_during_grid_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    labels = ["right", "left", "front", "back", "top", "iso"]
    image_paths: list[str] = []
    for label in labels:
        image_path = tmp_path / f"{label}.png"
        Image.new("RGB", (320, 320), "white").save(image_path)
        image_paths.append(str(image_path))

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(labels, image_paths),
            "metadata": {"response_cameras": labels},
        }

    original_compose_grid = rendering_module._compose_grid

    def mutate_view_then_compose(*args: object, **kwargs: object) -> Path:
        Image.new("RGB", (320, 320), "black").save(image_paths[0])
        return original_compose_grid(*args, **kwargs)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering._compose_grid",
        mutate_view_then_compose,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == "fail"
    assert any(
        issue["code"] == "render.image_changed"
        and issue["details"]["phase"] == "presentation composition"
        for issue in result.shared_render_issues
    )


def test_shared_render_binds_the_same_view_bytes_that_were_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                ["hero"], [str(image_path)]
            ),
            "metadata": {"response_cameras": ["hero"]},
        }

    original_validate = rendering_module._validate_image_paths

    def mutate_after_validation(
        *args: object, **kwargs: object
    ) -> list[dict[str, Any]]:
        results = original_validate(*args, **kwargs)
        Image.new("RGB", (320, 320), "black").save(image_path)
        return results

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering._validate_image_paths",
        mutate_after_validation,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
    )

    assert result.status == "fail"
    assert result.image_bindings == []
    assert any(
        issue["code"] == "render.image_changed"
        and issue["details"]["phase"] == "image validation"
        for issue in result.shared_render_issues
    )


def test_shared_render_forwards_prompt_grounded_focus_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "focus.png"
    image = Image.new("RGB", (320, 320), "black")
    ImageDraw.Draw(image).rectangle((50, 45, 270, 275), fill="white")
    image.save(image_path)
    captured: dict[str, Any] = {}

    def fake_render(**kwargs: object) -> dict[str, Any]:
        captured.update(kwargs)
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": _local_ovrtx_render_response(["+x"], [str(image_path)]),
            "metadata": {"response_cameras": ["+x"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        focus_prim_path="/World/Target",
        isolate_prim_paths=("/World/Target", "/World/Tool"),
        camera_margin=3.0,
        view_directions=("+x",),
    )

    assert result.status == "pass"
    assert captured["policy"] == {
        "render_backend": "ovrtx",
        "runtime_render_views": ["+x"],
        "render_image_width": 1024,
        "render_image_height": 1024,
        "render_studio_lighting": False,
        "render_focus_prim_path": "/World/Target",
        "render_isolate_prim_paths": ["/World/Target", "/World/Tool"],
        "render_frame_isolated_context": True,
        "render_camera_margin": 3.0,
        "render_camera_projection": "perspective",
        "render_ovrtx_mode": "pt",
        "render_ovrtx_num_sensor_updates": 64,
    }


def test_shared_render_reports_view_count_mismatch_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_paths: list[str] = []
    for index in range(5):
        path = tmp_path / f"view-{index}.png"
        Image.new("RGB", (320, 320), "white").save(path)
        image_paths.append(str(path))

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(
                ["right", "left", "front", "back", "top"], image_paths
            ),
            "metadata": {"response_cameras": ["right", "left", "front", "back", "top"]},
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == "fail"
    assert result.presentation_image_path is None
    assert len(result.image_paths) == 5
    assert any(
        issue["code"] == "render.view_count_mismatch"
        for issue in result.shared_render_issues
    )


def test_required_ovrtx_evidence_advances_only_its_exact_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _tiny_usda(tmp_path / "sdf_candidate.usda")
    image_path = tmp_path / "candidate_hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    evidence = _retained_candidate_render_evidence(
        source=candidate,
        image_path=image_path,
        report_path=tmp_path / "candidate_render_evidence.json",
    )
    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        lambda **_kwargs: evidence,
    )
    params = GeometryWorkflowInput(
        source_path=candidate,
        output_dir=tmp_path / "out",
        render_preset="hero",
    )

    check, failures, warnings, artifacts, previews, _report_path = (
        geometry_workflow._render_evidence(
            usd_path=candidate,
            output_dir=tmp_path / "out",
            params=params,
            required=True,
        )
    )

    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert check is not None
    assert check.status == "pass"
    assert failures == []
    assert warnings == []
    assert check.metadata["accepted_image_count"] == 1
    assert check.metadata["candidate_binding"]["status"] == "pass"
    assert check.metadata["candidate_binding"]["candidate_sha256_before_render"] == (
        candidate_sha256
    )
    assert any(artifact.kind == "ovrtx_candidate_binding" for artifact in artifacts)
    assert any(
        preview["claim_scope"] == "accepted_render_evidence"
        and preview["candidate_usd_sha256"] == candidate_sha256
        for preview in previews
    )


def test_sdf_candidate_rejects_images_rendered_from_different_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _tiny_usda(tmp_path / "sdf_candidate.usda")
    different_geometry = tmp_path / "different.usda"
    different_geometry.write_text(
        '#usda 1.0\ndef Xform "DifferentGeometry" {}\n',
        encoding="utf-8",
    )
    image_path = tmp_path / "different_geometry_hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    stale_evidence = _retained_candidate_render_evidence(
        source=different_geometry,
        image_path=image_path,
        report_path=tmp_path / "stale_render_evidence.json",
    )
    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        lambda **_kwargs: stale_evidence,
    )
    params = GeometryWorkflowInput(
        source_path=candidate,
        output_dir=tmp_path / "out",
        render_preset="hero",
    )

    check, failures, _warnings, artifacts, previews, _report_path = (
        geometry_workflow._render_evidence(
            usd_path=candidate,
            output_dir=tmp_path / "out",
            params=params,
            required=True,
        )
    )

    assert check is not None
    assert check.status == "fail"
    assert check.metadata["accepted_image_count"] == 0
    assert check.metadata["candidate_binding"]["status"] == "fail"
    assert any("source digest does not match" in failure for failure in failures)
    assert any("bound to different geometry" in failure for failure in failures)
    assert all(
        preview["claim_scope"] != "accepted_render_evidence" for preview in previews
    )
    view_artifact = next(
        artifact for artifact in artifacts if artifact.kind == "ovrtx_render_view"
    )
    assert view_artifact.metadata["claim_scope"] == "diagnostic_only"
    binding_artifact = next(
        artifact for artifact in artifacts if artifact.kind == "ovrtx_candidate_binding"
    )
    retained_binding = json.loads(
        Path(binding_artifact.path).read_text(encoding="utf-8")
    )
    assert retained_binding["status"] == "fail"


def test_required_ovrtx_evidence_rejects_changed_image_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _tiny_usda(tmp_path / "sdf_candidate.usda")
    image_path = tmp_path / "candidate_hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    evidence = _retained_candidate_render_evidence(
        source=candidate,
        image_path=image_path,
        report_path=tmp_path / "candidate_render_evidence.json",
    )
    image_path.write_bytes(b"stale image bytes")
    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        lambda **_kwargs: evidence,
    )
    params = GeometryWorkflowInput(
        source_path=candidate,
        output_dir=tmp_path / "out",
        render_preset="hero",
    )

    check, failures, _warnings, _artifacts, previews, _report_path = (
        geometry_workflow._render_evidence(
            usd_path=candidate,
            output_dir=tmp_path / "out",
            params=params,
            required=True,
        )
    )

    assert check is not None
    assert check.status == "fail"
    assert check.metadata["accepted_image_count"] == 0
    assert any("changed after evidence collection" in failure for failure in failures)
    assert all(preview["claim_scope"] == "diagnostic_only" for preview in previews)


def test_required_ovrtx_evidence_rechecks_candidate_after_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _tiny_usda(tmp_path / "sdf_candidate.usda")
    image_path = tmp_path / "candidate_hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    evidence = _retained_candidate_render_evidence(
        source=candidate,
        image_path=image_path,
        report_path=tmp_path / "candidate_render_evidence.json",
    )

    def mutate_candidate_after_render(**_kwargs: object) -> object:
        candidate.write_text(
            '#usda 1.0\ndef Xform "MutatedAfterRender" {}\n',
            encoding="utf-8",
        )
        return evidence

    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        mutate_candidate_after_render,
    )
    params = GeometryWorkflowInput(
        source_path=candidate,
        output_dir=tmp_path / "out",
        render_preset="hero",
    )

    check, failures, _warnings, _artifacts, _previews, _report_path = (
        geometry_workflow._render_evidence(
            usd_path=candidate,
            output_dir=tmp_path / "out",
            params=params,
            required=True,
        )
    )

    assert check is not None
    assert check.status == "fail"
    assert any("changed or disappeared during rendering" in item for item in failures)


def test_shared_render_warning_is_nonfatal_and_surfaces_in_workflow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [
                {
                    "code": "render.camera_advisory",
                    "severity": "warn",
                    "message": "Camera framing is usable but not preferred.",
                }
            ],
            "render_response": _local_ovrtx_render_response(
                ["hero"], [str(image_path)]
            ),
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "out",
        render_evidence=True,
        render_preset="hero",
    )

    check, failures, warnings, _artifacts, _previews, report_path = (
        geometry_workflow._render_evidence(
            usd_path=source,
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert check is not None
    assert check.status == "warning"
    assert failures == []
    assert warnings == ["Camera framing is usable but not preferred."]
    report = json.loads(Path(report_path or "").read_text(encoding="utf-8"))
    assert report["status"] == "pass"


@pytest.mark.parametrize(
    ("corrupt_after_validation", "expected_status"),
    [(False, "pass"), (True, "fail")],
)
def test_presentation_grid_failure_revalidates_individual_views(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corrupt_after_validation: bool,
    expected_status: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    labels = ["right", "left", "front", "back", "top", "iso"]
    image_paths: list[str] = []
    for index, label in enumerate(labels):
        image_path = tmp_path / f"{label}.png"
        image = Image.new("RGB", (320, 320), "black")
        draw = ImageDraw.Draw(image)
        draw.rectangle((55 + index, 50, 265, 275), fill="white")
        image.save(image_path)
        image_paths.append(str(image_path))

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(labels, image_paths),
            "metadata": {"response_cameras": labels},
        }

    def fail_compose(*_args: object, **_kwargs: object) -> Path:
        if corrupt_after_validation:
            Path(image_paths[0]).write_bytes(b"corrupted after validation")
        raise OSError("decode race")

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering._compose_grid",
        fail_compose,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == expected_status
    assert result.presentation_image_path is None
    assert len(result.image_paths) == 6
    assert all(item["passed"] for item in result.image_validation) is (
        not corrupt_after_validation
    )
    presentation_issue = next(
        issue
        for issue in result.shared_render_issues
        if issue["code"] == "render.presentation_grid_failed"
    )
    assert presentation_issue["severity"] == "warn"


@pytest.mark.parametrize(
    ("severity", "expected_status", "expected_failures", "expected_warnings"),
    [
        ("warning", "warning", [], ["Image contrast is too low."]),
        ("error", "fail", ["Image contrast is too low."], []),
    ],
)
def test_geometry_render_composition_surfaces_issues_even_on_inconsistent_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    severity: str,
    expected_status: str,
    expected_failures: list[str],
    expected_warnings: list[str],
) -> None:
    report_path = tmp_path / "render-report.json"
    report_path.write_text("{}\n", encoding="utf-8")
    render_call: dict[str, Any] = {}
    fake_result = types.SimpleNamespace(
        status="pass",
        backend="ovrtx",
        renderer="ovrtx",
        renderer_identity_verified=True,
        preset="hero",
        image_paths=[],
        presentation_image_path=None,
        shared_render_status="completed",
        shared_render_issues=[],
        image_validation=[
            {
                "passed": False,
                "issues": [
                    {
                        "code": "render.low_contrast",
                        "message": "Image contrast is too low.",
                        "severity": severity,
                    }
                ],
            }
        ],
        report_path=str(report_path),
    )

    def fake_render_geometry_evidence(**kwargs: Any) -> types.SimpleNamespace:
        render_call.update(kwargs)
        return fake_result

    monkeypatch.setattr(
        geometry_workflow,
        "render_geometry_evidence",
        fake_render_geometry_evidence,
    )
    params = GeometryWorkflowInput(
        source_path=tmp_path / "source.usda",
        output_dir=tmp_path / "out",
        render_evidence=True,
        render_preset="hero",
        render_remote_api_key=SecretStr("endpoint-test-token"),
        render_remote_allow_unauthenticated_identity=True,
    )

    check, failures, warnings, artifacts, _previews, _report = (
        geometry_workflow._render_evidence(
            usd_path=tmp_path / "source.usda",
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert check is not None
    assert check.status == expected_status
    assert failures == expected_failures
    assert warnings == expected_warnings
    assert render_call["remote_api_key"] == "endpoint-test-token"
    assert render_call["remote_allow_unauthenticated_identity"] is True
    report_artifact = next(
        artifact
        for artifact in artifacts
        if artifact.kind == "shared_render_evidence_report"
    )
    assert report_artifact.metadata["claim_scope"] == "render_diagnostics"


def test_usd_validation_preserves_upstream_severity_when_policy_downgrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    upstream = {
        "status": "success",
        "issues": [
            {
                "rule": "UsdAsciiPerformanceChecker",
                "severity": "failure",
                "message": "Use crate storage.",
            }
        ],
    }
    monkeypatch.setattr(
        "content_agent_workflows.geometry.validation.validate_usd",
        lambda *_args, **_kwargs: upstream,
    )

    result = run_geometry_usd_validation(source, tmp_path / "validation")

    assert result.status == "pass"
    assert result.failures == []
    assert result.warnings == ["Use crate storage."]
    assert result.upstream_result["issues"][0]["severity"] == "failure"
    assert result.policy_downgrades[0]["geometry_severity"] == "warning"


def test_geometry_validation_delegates_only_missing_texture_dependencies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    upstream = {
        "status": "success",
        "issues": [
            {
                "rule": "MissingReferenceChecker",
                "severity": "failure",
                "message": "Found unresolvable external dependency '/textures/albedo.png'.",
            },
            {
                "rule": "MissingReferenceChecker",
                "severity": "failure",
                "message": "Found unresolvable external dependency '/payload/body.usd'.",
            },
        ],
    }
    monkeypatch.setattr(
        "content_agent_workflows.geometry.validation.validate_usd",
        lambda *_args, **_kwargs: upstream,
    )

    result = run_geometry_usd_validation(source, tmp_path / "validation")

    assert result.status == "fail"
    assert result.failures == [
        "Found unresolvable external dependency '/payload/body.usd'."
    ]
    assert result.warnings == [
        "Found unresolvable external dependency '/textures/albedo.png'."
    ]


def test_mesh_normalization_authors_normals_without_changing_topology(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "triangle.usda"
    source.write_text(
        """#usda 1.0
def Mesh "Triangle"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    uniform token subdivisionScheme = "none"
}
""",
        encoding="utf-8",
    )

    report = scene_ops.author_missing_mesh_normals(
        source, tmp_path / "normalization.json"
    )

    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Triangle"))
    assert report["status"] == "pass"
    assert report["modified_mesh_count"] == 1
    assert report["points_or_topology_changed"] is False
    assert len(mesh.GetNormalsAttr().Get()) == 3
    assert mesh.GetNormalsInterpolation() == UsdGeom.Tokens.faceVarying
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2]


def test_mesh_normalization_preserves_existing_primvar_normals(tmp_path: Path) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "primvar_normals.usda"
    source.write_text(
        """#usda 1.0
def Mesh "Triangle"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    normal3f[] primvars:normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1)] (
        interpolation = "faceVarying"
    )
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    uniform token subdivisionScheme = "none"
}
""",
        encoding="utf-8",
    )

    report = scene_ops.author_missing_mesh_normals(
        source, tmp_path / "normalization.json"
    )

    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Triangle"))
    assert report["modified_mesh_count"] == 0
    assert report["unchanged_mesh_count"] == 1
    assert not (mesh.GetNormalsAttr().Get() or [])


def test_mesh_normalization_authors_complete_normals_with_degenerate_faces(
    tmp_path: Path,
) -> None:
    from pxr import Usd, UsdGeom

    source = tmp_path / "degenerate_face.usda"
    source.write_text(
        """#usda 1.0
def Mesh "Mixed"
{
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 0, 1]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    uniform token subdivisionScheme = "none"
}
""",
        encoding="utf-8",
    )

    report = scene_ops.author_missing_mesh_normals(
        source, tmp_path / "normalization.json"
    )

    stage = Usd.Stage.Open(str(source))
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/Mixed"))
    assert report["status"] == "pass"
    assert report["modified_mesh_count"] == 1
    assert len(mesh.GetNormalsAttr().Get()) == 6
    assert list(mesh.GetFaceVertexIndicesAttr().Get()) == [0, 1, 2, 0, 0, 1]


@pytest.mark.parametrize(
    ("foundation_status", "passed", "expected_check", "expected_simready"),
    [
        ("BLOCKED", False, "warning", "not_evaluated"),
        ("PASS", True, "pass", "pass"),
        ("FAIL", False, "fail", "fail"),
    ],
)
def test_only_formal_foundation_report_sets_simready_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    foundation_status: str,
    passed: bool,
    expected_check: str,
    expected_simready: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    report_path = tmp_path / "simready-profile.json"
    report_path.write_text(
        json.dumps({"schema_version": "test.simready.v1"}), encoding="utf-8"
    )
    fake_report = types.SimpleNamespace(
        status=foundation_status,
        passed=passed,
        errors=[] if passed else ["profile finding"],
        warnings=[],
        issues=[],
        profile_target="Prop-Robotics-Neutral@1.0.0",
        report_path=str(report_path),
    )
    monkeypatch.setattr(
        geometry_workflow,
        "run_simready_profile_validation",
        lambda _params: fake_report,
    )
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "out",
        simready_mode="validate",
    )

    check, _failures, _warnings, _artifacts, simready, *_paths = (
        geometry_workflow._simready_check(
            usd_path=source,
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert check.status == expected_check
    assert simready == expected_simready


def test_blocked_foundation_validation_does_not_route_asset_conformance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    report_path = tmp_path / "simready-profile.json"
    report_path.write_text("{}\n", encoding="utf-8")
    blocked_report = types.SimpleNamespace(
        status="BLOCKED",
        passed=False,
        errors=["Foundation runtime unavailable"],
        warnings=[],
        issues=[],
        profile_target="Prop-Robotics-Neutral@1.0.0",
        report_path=str(report_path),
    )
    monkeypatch.setattr(
        geometry_workflow,
        "run_simready_profile_validation",
        lambda _params: blocked_report,
    )
    monkeypatch.setattr(
        geometry_workflow,
        "run_simready_profile_conformance",
        lambda _params: pytest.fail("BLOCKED validation must not route conformance"),
    )

    result = geometry_workflow._simready_check(
        usd_path=source,
        output_dir=tmp_path / "out",
        params=GeometryWorkflowInput(
            source_path=source,
            output_dir=tmp_path / "out",
            simready_mode="validate_and_route_conformance",
        ),
    )

    check, _failures, _warnings, _artifacts, simready, _report, conformance = result
    assert check.status == "warning"
    assert simready == "not_evaluated"
    assert conformance is None


def test_cad_profile_aliases_are_shared_with_geometry() -> None:
    assert (
        resolve_official_simready_profile(local_profile_id=STATIC_VISUAL_PROFILE_ID)
        == "Prop-Robotics-Neutral"
    )
    assert (
        resolve_official_simready_profile(local_profile_id=INSERTION_FIXTURE_PROFILE_ID)
        == "Prop-Robotics-Physx"
    )


class _FakeHealthResponse:
    def __init__(self, payload: dict[str, Any], *, status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.closed = False

    def iter_content(self, *, chunk_size: int) -> list[bytes]:
        del chunk_size
        return [json.dumps(self.payload).encode("utf-8")]

    def close(self) -> None:
        self.closed = True


def _install_health_response(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, Any],
    *,
    status_code: int = 200,
    seen: dict[str, Any] | None = None,
    api_key: str = "test-token",
) -> _FakeHealthResponse:
    response = _FakeHealthResponse(payload, status_code=status_code)

    def fake_get(url: str, **kwargs: Any) -> _FakeHealthResponse:
        if seen is not None:
            seen.update({"url": url, **kwargs})
        return response

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.requests.get",
        fake_get,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.get_nvcf_api_key",
        lambda: api_key,
    )
    return response


def test_geometry_remote_api_key_is_excluded_from_serialization_and_repr(
    tmp_path: Path,
) -> None:
    params = GeometryWorkflowInput(
        source_path=tmp_path / "asset.usda",
        output_dir=tmp_path / "out",
        render_remote_api_key=SecretStr("endpoint-secret"),
    )

    assert "render_remote_api_key" not in params.model_dump()
    assert "endpoint-secret" not in repr(params)


def test_render_evidence_backfills_requested_backend_for_legacy_consumers() -> None:
    source_sha256 = "0" * 64
    local = GeometryRenderEvidence(
        status="unavailable",
        backend="ovrtx",
        preset="hero",
        source_usd_path="asset.usda",
        source_usd_sha256=source_sha256,
        ovrtx_render_mode="rt2",
        ovrtx_num_sensor_updates=32,
    )
    remote = GeometryRenderEvidence(
        status="unavailable",
        backend="remote",
        preset="hero",
        source_usd_path="asset.usda",
        source_usd_sha256=source_sha256,
        ovrtx_render_mode="rt2",
        ovrtx_num_sensor_updates=32,
    )

    assert local.requested_backend == "ovrtx"
    assert remote.requested_backend == "remote"
    assert "requested_backend" not in GeometryRenderEvidence.model_json_schema().get(
        "required", []
    )


@pytest.mark.parametrize(
    ("response_update", "entry_update", "metadata_update", "expected_field"),
    [
        ({"status": "failed"}, {}, {}, "render_response.status"),
        ({}, {"status": "failed"}, {}, "render_response.results[0].status"),
        ({}, {"error": "renderer failed"}, {}, "render_response.results[0].error"),
        (
            {"failures": ["renderer failed"]},
            {},
            {},
            "render_response.failures[0]",
        ),
        ({}, {}, {"renderer_status": "failed"}, "metadata.renderer_status"),
    ],
)
def test_shared_render_rejects_renderer_failure_metadata_with_healthy_image(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response_update: dict[str, Any],
    entry_update: dict[str, Any],
    metadata_update: dict[str, Any],
    expected_field: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    ImageDraw.Draw(image).rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        entry = {
            "camera": "hero",
            "images": [str(image_path)],
            **entry_update,
        }
        response = {
            "backend": "ovrtx",
            "cameras": ["hero"],
            "results": [entry],
            **response_update,
        }
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": response,
            "metadata": {"response_cameras": ["hero"], **metadata_update},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
    )

    assert result.status == "fail"
    assert result.image_validation[0]["passed"] is True
    issue = next(
        item
        for item in result.shared_render_issues
        if item["code"] == "render.renderer_reported_failure"
    )
    assert expected_field in {
        finding["field"] for finding in issue["details"]["findings"]
    }


def test_generic_remote_render_cannot_be_accepted_as_ovrtx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [{"camera": "hero", "images": [str(image_path)]}],
            },
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    _install_health_response(
        monkeypatch,
        {
            "service": "generic-rendering-api",
            "renderer": "generic",
            "status": "healthy",
            "gpu_initialized": True,
        },
    )
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "out",
        render_evidence=True,
        render_preset="hero",
        render_backend="remote",
        render_remote_base_url="https://generic-renderer.example.test",
        render_remote_api_key=SecretStr("test-token"),
    )

    check, failures, _warnings, artifacts, previews, report_path = (
        geometry_workflow._render_evidence(
            usd_path=source,
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert check is not None
    assert check.status == "fail"
    assert any("valid OVRTX health payload" in item for item in failures)
    assert "ovrtx_render_view" not in {artifact.kind for artifact in artifacts}
    assert "unverified_remote_render_view" in {artifact.kind for artifact in artifacts}
    assert all(
        preview["claim_scope"] != "accepted_render_evidence" for preview in previews
    )
    report = json.loads(Path(report_path or "").read_text(encoding="utf-8"))
    assert report["status"] == "fail"
    assert report["backend"] == "remote"
    assert report["renderer"] is None
    assert report["renderer_identity_verified"] is False
    assert any(
        issue["code"] == "render.ovrtx_health_payload_invalid"
        for issue in report["shared_render_issues"]
    )
    audit_signals = geometry_audit._audit_ovrtx_metadata(
        Path(report_path or ""),
        True,
    )
    assert [signal.code for signal in audit_signals] == [
        "render.required_ovrtx_unsuccessful"
    ]


def test_remote_ovrtx_health_observation_is_bound_and_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)
    endpoint = "https://ovrtx-renderer.example.test"
    render_seen: dict[str, Any] = {}
    health_seen: dict[str, Any] = {}

    def fake_render(**kwargs: object) -> dict[str, Any]:
        render_seen.update(kwargs)
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [
                    {
                        "camera": "hero",
                        "images": [str(image_path)],
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 64,
                        "active_aov": "LdrColor",
                    }
                ],
            },
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    health_response = _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
            "renderer_initialized": True,
            "daemon_running": True,
        },
        seen=health_seen,
        api_key="hosted-key-must-not-leak",
    )
    params = GeometryWorkflowInput(
        source_path=source,
        output_dir=tmp_path / "out",
        render_evidence=True,
        render_preset="hero",
        render_backend="remote",
        render_remote_base_url=endpoint,
        render_remote_api_key=SecretStr("test-token"),
    )

    check, failures, warnings, artifacts, previews, report_path = (
        geometry_workflow._render_evidence(
            usd_path=source,
            output_dir=tmp_path / "out",
            params=params,
        )
    )

    assert check is not None
    assert check.status == "pass"
    assert failures == []
    assert warnings == []
    assert "ovrtx_render_view" in {artifact.kind for artifact in artifacts}
    assert any(
        preview["claim_scope"] == "accepted_render_evidence" for preview in previews
    )
    report = json.loads(Path(report_path or "").read_text(encoding="utf-8"))
    assert report["backend"] == "remote"
    assert report["requested_backend"] == "remote"
    assert report["renderer"] == "ovrtx"
    assert report["renderer_identity_verified"] is True
    assert report["render_redirects_allowed"] is False
    assert report["ovrtx_render_mode"] == "pt"
    assert report["ovrtx_num_sensor_updates"] == 64
    assert report["metadata"]["executed_ovrtx_settings_verified"] is True
    assert report["metadata"]["executed_ovrtx_settings"] == {
        "ovrtx_render_mode": "pt",
        "ovrtx_num_sensor_updates": 64,
        "active_aov": "LdrColor",
    }
    assert report["renderer_identity_evidence"]["endpoint"] == endpoint
    assert report["renderer_identity_evidence"]["health_url"] == (f"{endpoint}/health")
    assert report["renderer_identity_evidence"]["http_status"] == 200
    assert report["renderer_identity_evidence"]["observed_at"].endswith("Z")
    assert report["renderer_identity_evidence"]["trust_mode"] == "endpoint_bearer_token"
    retained_health = report["renderer_identity_evidence"]["health"]
    assert retained_health["gpu_initialized"] is True
    assert set(retained_health) == {
        "service",
        "renderer",
        "status",
        "gpu_initialized",
    }
    assert render_seen["policy"]["render_base_url"] == endpoint
    assert render_seen["policy"]["render_api_key"] == "test-token"
    assert render_seen["policy"]["render_allow_redirects"] is False
    assert health_seen["url"] == f"{endpoint}/health"
    assert health_seen["allow_redirects"] is False
    assert health_seen["stream"] is True
    assert health_seen["headers"]["Authorization"] == "Bearer test-token"
    assert health_seen["timeout"] == (10.0, 5.0)
    assert health_response.closed is True
    assert (
        geometry_audit._audit_ovrtx_metadata(
            Path(report_path or ""),
            True,
        )
        == []
    )
    report["render_redirects_allowed"] = True
    Path(report_path or "").write_text(json.dumps(report), encoding="utf-8")
    redirecting_signals = geometry_audit._audit_ovrtx_metadata(
        Path(report_path or ""),
        True,
    )
    assert [signal.code for signal in redirecting_signals] == ["render.required_ovrtx"]


@pytest.mark.parametrize(
    ("entry_settings", "expected_issue"),
    [
        ({}, "render.remote_ovrtx_settings_unverified"),
        (
            {
                "ovrtx_render_mode": "rt2",
                "ovrtx_num_sensor_updates": 32,
                "active_aov": "LdrColor",
            },
            "render.remote_ovrtx_settings_mismatch",
        ),
    ],
)
def test_remote_ovrtx_requires_executed_settings_matching_the_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_settings: dict[str, object],
    expected_issue: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [
                    {
                        "camera": "hero",
                        "images": [str(image_path)],
                        **entry_settings,
                    }
                ],
            },
            "metadata": {"response_cameras": ["hero"]},
        },
    )
    _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / expected_issue,
        preset="hero",
        backend="remote",
        remote_base_url="https://ovrtx-renderer.example.test",
        remote_api_key="test-token",
        ovrtx_mode="pt",
        ovrtx_num_sensor_updates=64,
    )

    assert result.status == "fail"
    assert expected_issue in {issue["code"] for issue in result.shared_render_issues}


@pytest.mark.parametrize(
    ("entry_settings", "expected_issue"),
    [
        ({}, "render.local_ovrtx_settings_unverified"),
        (
            {
                "ovrtx_render_mode": "rt2",
                "ovrtx_num_sensor_updates": 32,
                "active_aov": "LdrColor",
            },
            "render.local_ovrtx_settings_mismatch",
        ),
    ],
)
def test_local_ovrtx_requires_executed_settings_matching_the_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_settings: dict[str, object],
    expected_issue: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "ovrtx",
                "cameras": ["hero"],
                "results": [
                    {
                        "camera": "hero",
                        "images": [str(image_path)],
                        **entry_settings,
                    }
                ],
            },
            "metadata": {"response_cameras": ["hero"]},
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / expected_issue,
        preset="hero",
        ovrtx_mode="pt",
        ovrtx_num_sensor_updates=64,
    )

    assert result.status == "fail"
    assert expected_issue in {issue["code"] for issue in result.shared_render_issues}


def test_canonical_nvcf_endpoint_uses_hosted_key_for_probe_and_render(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    ImageDraw.Draw(image).rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)
    endpoint = "https://test-function.invocation.api.nvcf.nvidia.com"
    render_seen: dict[str, Any] = {}
    health_seen: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> dict[str, Any]:
        render_seen.update(kwargs)
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [
                    {
                        "camera": "hero",
                        "images": [str(image_path)],
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 64,
                        "active_aov": "LdrColor",
                    }
                ],
            },
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
        },
        seen=health_seen,
        api_key="hosted-test-token",
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="remote",
        remote_base_url=endpoint,
    )

    assert result.status == "pass"
    assert result.renderer_identity_evidence["trust_mode"] == "nvcf_bearer_token"
    assert health_seen["headers"]["Authorization"] == "Bearer hosted-test-token"
    assert render_seen["policy"]["render_api_key"] == "hosted-test-token"


def test_custom_http_endpoint_rejects_bearer_without_leaking_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    def unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("a bearer credential must not be sent over plaintext HTTP")

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.requests.get",
        unexpected_call,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        unexpected_call,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.get_nvcf_api_key",
        unexpected_call,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="remote",
        remote_base_url="http://ovrtx-renderer.example.test",
        remote_api_key="endpoint-secret",
    )

    assert result.status == "fail"
    assert any(
        issue["code"] == "render.ovrtx_bearer_requires_https"
        for issue in result.shared_render_issues
    )
    report_text = Path(result.report_path or "").read_text(encoding="utf-8")
    assert "endpoint-secret" not in report_text


def test_nonlocal_remote_ovrtx_identity_requires_auth_or_operator_trust(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    render_seen: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> dict[str, Any]:
        render_seen.update(kwargs)
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [{"camera": "hero", "images": [str(image_path)]}],
            },
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.get_nvcf_api_key",
        lambda: "hosted-key-must-not-leak",
    )

    def unexpected_health_request(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an untrusted non-loopback endpoint must not be probed")

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.requests.get",
        unexpected_health_request,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="remote",
        remote_base_url="https://ovrtx-renderer.example.test",
    )

    assert result.status == "fail"
    assert result.renderer_identity_verified is False
    assert render_seen["policy"]["render_api_key"] == ""
    assert any(
        issue["code"] == "render.ovrtx_health_auth_required"
        for issue in result.shared_render_issues
    )


@pytest.mark.parametrize(
    ("endpoint", "rejection_reason"),
    [
        (
            "https://user:super-secret@ovrtx-renderer.example.test",
            "userinfo",
        ),
        (
            "https://ovrtx-renderer.example.test?token=super-secret",
            "query",
        ),
        (
            "https://ovrtx-renderer.example.test#super-secret",
            "fragment",
        ),
        ("https://ovrtx-renderer.example.test?", "query"),
        ("https://ovrtx-renderer.example.test#", "fragment"),
        (
            "ftp://user:super-secret@ovrtx-renderer.example.test",
            "unsupported_scheme",
        ),
    ],
)
def test_unsafe_remote_endpoint_is_not_probed_rendered_or_persisted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    rejection_reason: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")

    def unexpected_call(*_args: object, **_kwargs: object) -> None:
        pytest.fail("an unsafe renderer endpoint must not be probed or rendered")

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.requests.get",
        unexpected_call,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        unexpected_call,
    )
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.get_nvcf_api_key",
        unexpected_call,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / rejection_reason,
        preset="hero",
        backend="remote",
        remote_base_url=endpoint,
    )

    assert result.status == "fail"
    assert result.renderer_endpoint is None
    assert result.renderer_identity_verified is False
    assert result.renderer_identity_evidence == {}
    endpoint_issue = next(
        issue
        for issue in result.shared_render_issues
        if issue["code"] == "render.ovrtx_endpoint_rejected"
    )
    assert rejection_reason in endpoint_issue["details"]["rejection_reasons"]
    report_text = Path(result.report_path or "").read_text(encoding="utf-8")
    if "super-secret" in endpoint:
        assert "super-secret" not in report_text
    assert endpoint not in report_text


@pytest.mark.parametrize(
    ("endpoint", "operator_trust", "expected_trust_mode"),
    [
        ("http://127.0.0.1:8000", False, "loopback"),
        (
            "http://ovrtx-rendering-api:8000",
            True,
            "operator_trusted_unauthenticated",
        ),
    ],
)
def test_explicitly_trusted_unauthenticated_ovrtx_identity_modes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    endpoint: str,
    operator_trust: bool,
    expected_trust_mode: str,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    ImageDraw.Draw(image).rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)
    render_seen: dict[str, Any] = {}

    def fake_render(**kwargs: Any) -> dict[str, Any]:
        render_seen.update(kwargs)
        return {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [
                    {
                        "camera": "hero",
                        "images": [str(image_path)],
                        "ovrtx_render_mode": "pt",
                        "ovrtx_num_sensor_updates": 64,
                        "active_aov": "LdrColor",
                    }
                ],
            },
            "metadata": {"response_cameras": ["hero"]},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )
    health_seen: dict[str, Any] = {}
    _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
        },
        seen=health_seen,
        api_key="hosted-key-must-not-leak",
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / expected_trust_mode,
        preset="hero",
        backend="remote",
        remote_base_url=endpoint,
        remote_allow_unauthenticated_identity=operator_trust,
    )

    assert result.status == "pass"
    assert result.renderer_identity_verified is True
    assert result.renderer_identity_evidence["trust_mode"] == expected_trust_mode
    assert "Authorization" not in health_seen["headers"]
    assert render_seen["policy"]["render_api_key"] == ""


def test_remote_ovrtx_health_observation_is_size_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [{"camera": "hero", "images": [str(image_path)]}],
            },
            "metadata": {"response_cameras": ["hero"]},
        },
    )
    health_response = _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
            "padding": "x" * (64 * 1024),
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="remote",
        remote_base_url="https://ovrtx-renderer.example.test",
        remote_api_key="test-token",
    )

    assert result.status == "fail"
    assert result.renderer_identity_verified is False
    assert any(
        issue["code"] == "render.ovrtx_health_payload_invalid"
        for issue in result.shared_render_issues
    )
    assert "health" not in result.renderer_identity_evidence
    assert health_response.closed is True


def test_remote_ovrtx_health_observation_rejects_after_elapsed_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    Image.new("RGB", (320, 320), "white").save(image_path)
    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [{"camera": "hero", "images": [str(image_path)]}],
            },
            "metadata": {"response_cameras": ["hero"]},
        },
    )
    health_response = _install_health_response(
        monkeypatch,
        {
            "service": "ovrtx-rendering-api",
            "renderer": "ovrtx",
            "status": "healthy",
            "gpu_initialized": True,
        },
    )
    clock_calls = 0

    def fake_monotonic() -> float:
        nonlocal clock_calls
        clock_calls += 1
        return 0.0 if clock_calls == 1 else 31.0

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.time.monotonic",
        fake_monotonic,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="remote",
        remote_base_url="https://ovrtx-renderer.example.test",
        remote_api_key="test-token",
    )

    assert result.status == "fail"
    assert any(
        issue["code"] == "render.ovrtx_health_probe_timed_out"
        for issue in result.shared_render_issues
    )
    assert health_response.closed is True


def test_requested_backend_cannot_overwrite_actual_backend_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    image_path = tmp_path / "hero.png"
    image = Image.new("RGB", (320, 320), "black")
    draw = ImageDraw.Draw(image)
    draw.rectangle((60, 55, 260, 270), fill="white")
    image.save(image_path)

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        lambda **_kwargs: {
            "status": "completed",
            "backend": "remote",
            "image_paths": [str(image_path)],
            "issues": [],
            "render_response": {
                "backend": "remote",
                "cameras": ["hero"],
                "results": [{"camera": "hero", "images": [str(image_path)]}],
            },
            "metadata": {"response_cameras": ["hero"]},
        },
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="hero",
        backend="ovrtx",
    )

    assert result.status == "fail"
    assert result.requested_backend == "ovrtx"
    assert result.backend == "remote"
    assert result.renderer is None
    assert result.renderer_identity_verified is False
    mismatch = next(
        issue
        for issue in result.shared_render_issues
        if issue["code"] == "render.backend_provenance_mismatch"
    )
    assert mismatch["details"] == {
        "requested_backend": "ovrtx",
        "actual_backend": "remote",
    }


def test_ovrtx_audit_rejects_spoofed_local_backend_name(tmp_path: Path) -> None:
    report_path = tmp_path / "spoofed-render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
                ),
                "status": "pass",
                "shared_render_status": "completed",
                "image_paths": [str(tmp_path / "spoofed.png")],
                "backend": "fake_ovrtx",
                "requested_backend": "ovrtx",
                "renderer": "ovrtx",
                "renderer_identity_verified": True,
                "renderer_identity_evidence": {
                    "source": "explicit_local_backend",
                    "renderer": "ovrtx",
                    "ready": True,
                },
            }
        ),
        encoding="utf-8",
    )

    signals = geometry_audit._audit_ovrtx_metadata(report_path, True)

    assert [signal.code for signal in signals] == ["render.required_ovrtx"]


@pytest.mark.parametrize(
    ("renderer_metadata", "response_validation"),
    [
        ({"renderer_status": "failed"}, {}),
        ({"error": "renderer failed"}, {}),
        ({}, {"metadata": {"failures": ["renderer failed"]}}),
    ],
)
def test_ovrtx_audit_blocks_retained_renderer_failure_metadata(
    tmp_path: Path,
    renderer_metadata: dict[str, Any],
    response_validation: dict[str, Any],
) -> None:
    image_path = tmp_path / "hero.png"
    report_path = tmp_path / "render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
                ),
                "status": "pass",
                "shared_render_status": "completed",
                "shared_render_issues": [],
                "image_paths": [str(image_path)],
                "backend": "ovrtx",
                "requested_backend": "ovrtx",
                "renderer": "ovrtx",
                "renderer_identity_verified": True,
                "renderer_identity_evidence": {
                    "source": "explicit_local_backend",
                    "renderer": "ovrtx",
                    "ready": True,
                },
                "metadata": renderer_metadata,
                "response_validation": response_validation,
            }
        ),
        encoding="utf-8",
    )

    signals = geometry_audit._audit_ovrtx_metadata(report_path, True)

    assert [signal.code for signal in signals] == ["render.renderer_reported_failure"]
    assert signals[0].blocking is True


def test_ovrtx_audit_requires_supplied_render_image(tmp_path: Path) -> None:
    report_path = tmp_path / "render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
                ),
                "status": "pass",
                "shared_render_status": "completed",
                "image_paths": [str(tmp_path / "recorded-view.png")],
                "backend": "ovrtx",
                "requested_backend": "ovrtx",
                "renderer": "ovrtx",
                "renderer_identity_verified": True,
                "renderer_identity_evidence": {
                    "source": "explicit_local_backend",
                    "renderer": "ovrtx",
                    "ready": True,
                },
            }
        ),
        encoding="utf-8",
    )

    signals = geometry_audit._audit_render_artifacts(
        render_image_path=None,
        render_metadata_path=report_path,
        render_preset="hero",
        require_ovrtx_render=True,
    )

    assert "render.required_ovrtx_image_missing" in {signal.code for signal in signals}


def test_ovrtx_audit_binds_supplied_image_to_render_report(tmp_path: Path) -> None:
    recorded_path = tmp_path / "recorded-view.png"
    unrelated_path = tmp_path / "unrelated-view.png"
    for path in (recorded_path, unrelated_path):
        image = Image.new("RGB", (320, 320), "black")
        ImageDraw.Draw(image).rectangle((60, 55, 260, 270), fill="white")
        image.save(path)
    report_path = tmp_path / "render-report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": (
                    rendering_module.GEOMETRY_RENDER_EVIDENCE_SCHEMA_VERSION
                ),
                "status": "pass",
                "shared_render_status": "completed",
                "image_paths": [str(recorded_path)],
                "backend": "ovrtx",
                "requested_backend": "ovrtx",
                "renderer": "ovrtx",
                "renderer_identity_verified": True,
                "renderer_identity_evidence": {
                    "source": "explicit_local_backend",
                    "renderer": "ovrtx",
                    "ready": True,
                },
                "ovrtx_render_mode": "pt",
                "ovrtx_num_sensor_updates": 64,
                "active_aov": "LdrColor",
                "image_bindings": [
                    {
                        "path": str(recorded_path),
                        "active_aov": "LdrColor",
                    }
                ],
                "metadata": {
                    "requested_ovrtx_render_mode": "pt",
                    "requested_ovrtx_num_sensor_updates": 64,
                    "executed_ovrtx_settings": _executed_ovrtx_fields("pt", 64),
                    "executed_ovrtx_settings_verified": True,
                    "active_aov": "LdrColor",
                },
            }
        ),
        encoding="utf-8",
    )

    signals = geometry_audit._audit_render_artifacts(
        render_image_path=unrelated_path,
        render_metadata_path=report_path,
        render_preset="hero",
        require_ovrtx_render=True,
    )

    assert "render.required_ovrtx_image_unbound" in {signal.code for signal in signals}
    bound_signals = geometry_audit._audit_render_artifacts(
        render_image_path=recorded_path,
        render_metadata_path=report_path,
        render_preset="hero",
        require_ovrtx_render=True,
    )
    assert not [signal for signal in bound_signals if signal.blocking]


def test_asset_audit_selects_accepted_individual_view_not_grid(
    tmp_path: Path,
) -> None:
    presentation_path = tmp_path / "grid.png"
    accepted_path = tmp_path / "front.png"
    diagnostic_path = tmp_path / "diagnostic.png"
    previews = [
        {
            "kind": "render_presentation_grid",
            "path": str(presentation_path),
            "claim_scope": "presentation_only",
        },
        {
            "kind": "unverified_remote_render_view",
            "path": str(diagnostic_path),
            "claim_scope": "diagnostic_only",
        },
        {
            "kind": "ovrtx_render_view",
            "path": str(accepted_path),
            "claim_scope": "accepted_render_evidence",
        },
    ]

    assert geometry_workflow._accepted_render_image_path(previews) == (
        accepted_path.resolve()
    )
    assert geometry_workflow._accepted_render_image_path(previews[:2]) is None
    assert geometry_workflow._presentation_render_image_path(previews) == (
        presentation_path.resolve()
    )


def test_six_view_framing_audits_grid_not_individual_view(tmp_path: Path) -> None:
    individual_path = tmp_path / "front.png"
    individual = Image.new("RGB", (1024, 1024), "black")
    ImageDraw.Draw(individual).rectangle((90, 420, 934, 610), fill="white")
    individual.save(individual_path)

    presentation_path = tmp_path / "six-view.png"
    presentation = Image.new("RGB", (3072, 2048), "black")
    draw = ImageDraw.Draw(presentation)
    for row in range(2):
        for column in range(3):
            left = column * 1024
            top = row * 1024
            draw.rectangle(
                (left + 96, top + 300, left + 928, top + 720),
                fill="white",
            )
    presentation.save(presentation_path)

    signals = geometry_audit._audit_render_artifacts(
        render_image_path=individual_path,
        render_presentation_path=presentation_path,
        render_metadata_path=None,
        render_preset="six_view",
        require_ovrtx_render=False,
    )

    assert "render.image_content_clipped" not in {signal.code for signal in signals}

    draw.rectangle((0, 300, 928, 720), fill="white")
    presentation.save(presentation_path)
    clipped_signals = geometry_audit._audit_render_artifacts(
        render_image_path=individual_path,
        render_presentation_path=presentation_path,
        render_metadata_path=None,
        render_preset="six_view",
        require_ovrtx_render=False,
    )
    assert "render.image_content_clipped" in {signal.code for signal in clipped_signals}


def test_shared_render_does_not_exempt_blank_or_artifacted_edge_on_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _tiny_usda(tmp_path / "source.usda")
    labels = ["right", "left", "front", "back", "top", "bottom"]
    image_paths: list[str] = []
    for index, _label in enumerate(labels):
        path = tmp_path / f"view-{index}.png"
        image = Image.new("RGB", (320, 320), "black")
        draw = ImageDraw.Draw(image)
        if index == 2:
            draw.rectangle((24, 159, 296, 159), fill="white")
        else:
            draw.rectangle((60, 55, 260, 270), fill="white")
        image.save(path)
        image_paths.append(str(path))

    def fake_render(**_kwargs: object) -> dict[str, Any]:
        return {
            "status": "completed",
            "backend": "ovrtx",
            "image_paths": image_paths,
            "issues": [],
            "render_response": _local_ovrtx_render_response(labels, image_paths),
            "metadata": {"response_cameras": labels},
        }

    monkeypatch.setattr(
        "content_agent_workflows.geometry.rendering.render_usd_visual_evidence",
        fake_render,
    )

    result = render_geometry_evidence(
        usd_path=source,
        output_dir=tmp_path / "render",
        preset="six_view",
    )

    assert result.status == "fail"
    assert result.image_validation[2]["passed"] is False
    assert {issue["code"] for issue in result.image_validation[2]["issues"]} >= {
        "render.blank_image",
        "ovrtx.render_artifact_detected",
    }
    assert not any(
        issue["code"] == "render.edge_on_thin_geometry"
        for issue in result.shared_render_issues
    )
    assert result.metadata["edge_on_thin_views"] == []
