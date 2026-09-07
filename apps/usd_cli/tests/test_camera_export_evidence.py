# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused calibration, rig-schema, and renderer-evidence regressions."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from pxr import Gf, Usd, UsdGeom
from usd_core.camera import author_camera
from usd_core.camera_analysis.authoring import RIG_SCHEMA, read_rig_metadata
from usd_core.camera_analysis.calibration import camera_calibration
from usd_core.camera_analysis.cancellation import (
    CameraAnalysisCancelled,
    cancellation_scope,
)
from usd_core.camera_analysis.contracts import SceneAnalysisPolicy
from usd_core.camera_analysis.evidence import (
    CAMERA_OBSERVATION_CAPABILITY,
    OBSERVATION_AOVS,
    OBSERVATION_IMAGE_CHANNELS,
    OVSTAGE_ATTACHED_CAPABILITY,
    OVSTAGE_TRANSPORT,
    PARITY_V1_MINIMUM_MASK_IOU,
    QUALIFIED_RUNTIME_VERSIONS,
    RGB_RENDER_CAPABILITY,
    SEMANTIC_OVERLAY_CAPABILITY,
    USD_DEFAULT_ANALYSIS_TIME,
    VERIFICATION_EVIDENCE_SCHEMA,
    WARP_DLPACK_REDUCTION_CAPABILITY,
    WORKER_PROTOCOL_VERSION,
    _newton_ovrtx_parity,
    canonical_json_digest,
    parity_v1_depth_tolerances,
    semantic_label,
    verification_semantic_roles,
    verify_rig_with_ovrtx,
)
from usd_core.camera_analysis.identifiers import camera_stable_identity
from usd_core.camera_analysis.rig_export import (
    build_rig_document,
    publish_json_document,
    publish_rig_document,
    validate_verification_generation,
)
from usd_core.camera_analysis.scene import build_scene_analysis_ir


def _rig_stage(*, cameras: int = 2):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    rig = UsdGeom.Xform.Define(stage, "/World/Rig")
    rig.GetPrim().SetCustomDataByKey("usdCameraRigSchema", RIG_SCHEMA)
    rig.GetPrim().SetCustomDataByKey("usdCameraRigMethod", "max_coverage")
    rig.GetPrim().SetCustomDataByKey("usdCameraRigConfigJson", '{"seed":17}')
    paths = []
    for index in range(1, cameras + 1):
        path = f"/World/Rig/Camera_{index:03d}"
        camera = author_camera(
            stage,
            path,
            (float(index - 1), 0.0, 10.0),
            (0.0, 0.0, 0.0),
            focal=50.0,
            h_ap=40.0,
            v_ap=20.0,
            near=0.1,
            far=100.0,
        )
        camera.GetPrim().SetCustomDataByKey("usdCameraStableId", f"camera-{index:03d}")
        paths.append(path)
    return stage, paths


@pytest.mark.parametrize("malformed", ["", "{", "[]", 7])
def test_present_malformed_rig_config_metadata_fails_closed(malformed: object) -> None:
    stage, _paths = _rig_stage(cameras=1)
    prim = stage.GetPrimAtPath("/World/Rig")
    prim.SetCustomDataByKey("usdCameraRigConfigJson", malformed)

    with pytest.raises(ValueError, match="camera rig config metadata is malformed"):
        read_rig_metadata(prim)


def test_absent_rig_config_metadata_remains_supported() -> None:
    stage = Usd.Stage.CreateInMemory()
    prim = UsdGeom.Xform.Define(stage, "/Rig").GetPrim()

    assert read_rig_metadata(prim) == {}


class _QualifiedBackend:
    name = "ovrtx"

    def __init__(self) -> None:
        self.render_called = False

    def runtime_identity(self) -> dict:
        return {
            "worker_protocol_version": WORKER_PROTOCOL_VERSION,
            "capabilities": [RGB_RENDER_CAPABILITY, OVSTAGE_ATTACHED_CAPABILITY],
            "stage_transport": OVSTAGE_TRANSPORT,
            "runtime_versions": dict(QUALIFIED_RUNTIME_VERSIONS),
        }

    def render(self, stage, cameras, width, height, out_dir, mode, names):
        self.render_called = True
        results = []
        for camera, name in zip(cameras, names, strict=True):
            path = Path(out_dir) / f"{name}.png"
            path.write_bytes(b"deterministic-rgb-evidence\0" + camera.encode("utf-8"))
            results.append(
                SimpleNamespace(
                    camera=camera,
                    path=str(path),
                    blank_suspect=False,
                    render_time=123.456,  # deliberately excluded from canonical evidence
                    ovrtx_render_mode="rt2",
                    ovrtx_num_sensor_updates=64,
                    active_aov="LdrColor",
                    renderer_identity={"engine": "ovrtx", "protocol_version": 4},
                )
            )
        return list(reversed(results))  # backend ordering is not evidence ordering


class _ObservationBackend(_QualifiedBackend):
    def runtime_identity(self) -> dict:
        identity = super().runtime_identity()
        identity["capabilities"].extend(
            [
                CAMERA_OBSERVATION_CAPABILITY,
                WARP_DLPACK_REDUCTION_CAPABILITY,
                SEMANTIC_OVERLAY_CAPABILITY,
            ]
        )
        return identity

    def observe(
        self,
        stage,
        cameras,
        width,
        height,
        out_dir,
        *,
        aovs,
        artifact_aovs,
        semantic_labels,
        names,
        **_kwargs,
    ):
        from PIL import Image

        assert list(aovs) == OBSERVATION_AOVS
        assignments = tuple(
            {"path": path, "label": semantic_labels[path]}
            for path in sorted(semantic_labels)
        )
        rendered_labels = [
            {"id": index, "label": f"usd_cli: {label};", "pixels": 1}
            for index, label in enumerate(semantic_labels.values(), start=1)
        ]
        results = []
        for camera, name in zip(cameras, names, strict=True):
            artifact_paths = {}
            for artifact_aov in artifact_aovs:
                if artifact_aov == "LdrColor":
                    path = Path(out_dir) / f"{name}__LdrColor.png"
                    image = Image.new("RGB", (width, height), (12, 24, 48))
                    for x in range(width):
                        value = int(255 * x / max(1, width - 1))
                        for y in range(height):
                            image.putpixel((x, y), (value, 255 - value, (x * 7) % 256))
                    image.save(path)
                else:
                    path = Path(out_dir) / f"{name}__{artifact_aov}.npy"
                    values = (
                        np.ones((height, width, 1), dtype=np.float32)
                        if artifact_aov == "DistanceToCameraSD"
                        else np.ones((height, width, 1), dtype=np.uint32)
                    )
                    np.save(path, values, allow_pickle=False)
                artifact_paths[artifact_aov] = str(path)
            summaries = {}
            for aov in OBSERVATION_AOVS:
                reduction = (
                    "cpu_semantic_metadata_decode_v1"
                    if aov == "SemanticIdMap"
                    else "warp_cuda_dlpack_v1"
                )
                channels = OBSERVATION_IMAGE_CHANNELS.get(aov, 1)
                statistics = {
                    "element_count": (
                        1 if aov == "SemanticIdMap" else width * height * channels
                    ),
                    "reduction": reduction,
                }
                if aov == "DistanceToCameraSD":
                    statistics["valid_count"] = width * height
                    statistics["nonzero_count"] = width * height
                    statistics["minimum"] = 1.0
                    statistics["maximum"] = 1.0
                if aov == "SemanticSegmentation":
                    statistics["semantic_labels"] = rendered_labels
                summaries[aov] = {
                    "shape": (
                        [1] if aov == "SemanticIdMap" else [height, width, channels]
                    ),
                    "dtype": "float32",
                    "statistics": statistics,
                }
            results.append(
                SimpleNamespace(
                    camera=camera,
                    render_product=f"/Render/{name}",
                    aovs=summaries,
                    artifacts=artifact_paths,
                    ovrtx_render_mode="rt2",
                    ovrtx_num_sensor_updates=64,
                    semantic_assignments=assignments,
                )
            )
        return results


def test_calibration_matches_openusd_filmback_offsets_and_floor_top_plane():
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(paths[0]))
    camera.CreateHorizontalApertureOffsetAttr(2.0)
    camera.CreateVerticalApertureOffsetAttr(-1.0)

    calibration = camera_calibration(
        stage,
        scene,
        paths[0],
        resolution=(1000, 500),
        floor_bounds_m=(np.asarray([-1.0, -1.0, -0.1]), np.asarray([1.0, 1.0, 0.1])),
    )

    assert np.allclose(
        np.asarray(calibration["intrinsics"]["K"]),
        np.asarray([[1250.0, 0.0, 450.0], [0.0, 1250.0, 225.0], [0.0, 0.0, 1.0]]),
    )
    projection = calibration["calibration"]["projection_verification"]
    assert projection["reference"].startswith("OpenUSD GfCamera")
    assert projection["passed"] is True
    assert projection["max_error_px"] < 1.0e-9
    assert calibration["homography"]["floor_z_m"] == pytest.approx(0.1)
    assert calibration["homography"]["verification"]["passed"] is True
    assert calibration["intrinsics"]["raw_unit"] == "tenths_of_scene_unit"


def test_rig_document_declares_edge_origin_half_integer_pixel_centers():
    stage, _paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)

    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(5, 3),
    )

    intrinsics = np.asarray(document["cameras"][0]["intrinsics"]["K"])
    assert intrinsics[0, 2] == pytest.approx(2.5)
    assert intrinsics[1, 2] == pytest.approx(1.5)
    assert document["conventions"]["pixel_coordinates"] == (
        "continuous edge-origin coordinates; integer values denote pixel edges "
        "and pixel (column, row) has center (column + 0.5, row + 0.5)"
    )


def test_homography_uses_measured_accessible_floor_and_rejects_nonplanarity():
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)
    bounds = (np.asarray([-5.0, -5.0, -0.1]), np.asarray([5.0, 5.0, 5.0]))
    planar = {
        "surface_z_m": 0.0,
        "surface_max_deviation_m": 2.0e-6,
        "surface_planarity_tolerance_m": 1.0e-4,
    }

    calibration = camera_calibration(
        stage,
        scene,
        paths[0],
        resolution=(640, 480),
        floor_bounds_m=bounds,
        floor_surface=planar,
    )
    assert calibration["homography"]["eligible"] is True
    assert calibration["homography"]["floor_z_m"] == pytest.approx(0.0)
    assert calibration["homography"]["source"] == ("accessible_coverage_samples_median")

    nonplanar = dict(planar, surface_max_deviation_m=0.25)
    calibration = camera_calibration(
        stage,
        scene,
        paths[0],
        resolution=(640, 480),
        floor_bounds_m=bounds,
        floor_surface=nonplanar,
    )
    assert calibration["homography"] == {
        "eligible": False,
        "reason": "nonplanar_accessible_surface",
        "floor_z_m": 0.0,
        "source": "accessible_coverage_samples_median",
        "max_deviation_m": 0.25,
        "planarity_tolerance_m": 1.0e-4,
    }


def test_ineligible_calibration_retains_lossless_visibility():
    stage, paths = _rig_stage(cameras=1)
    camera_prim = stage.GetPrimAtPath(paths[0])
    UsdGeom.Xformable(camera_prim).AddScaleOp().Set(Gf.Vec3f(2.0, 1.0, 1.0))
    scene = build_scene_analysis_ir(stage)
    visibility = {
        "schema": "usd-cli.visibility-grid.v1",
        "regions": [
            {
                "outline": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "holes": [],
                "area_m2": 0.5,
            }
        ],
    }

    calibration = camera_calibration(
        stage,
        scene,
        paths[0],
        resolution=(640, 480),
        visibility=visibility,
    )

    assert calibration["calibration"]["eligible"] is False
    assert "non_rigid_camera_transform" in calibration["calibration"]["reasons"]
    assert calibration["visibility"] == visibility


def test_calibration_rejects_authored_zero_lens_values_instead_of_using_fallbacks():
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(paths[0]))
    camera.GetFocalLengthAttr().Set(0.0)

    with pytest.raises(ValueError, match="nonpositive focal length"):
        camera_calibration(stage, scene, paths[0], resolution=(640, 480))


def test_qualified_evidence_is_versioned_relocatable_and_deterministic(tmp_path):
    stage, paths = _rig_stage()
    scene = build_scene_analysis_ir(stage)

    first, first_paths = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_QualifiedBackend(),
        resolution=(640, 480),
        output_dir=tmp_path / "bundle_a" / "rig_evidence",
        artifact_base_dir=tmp_path / "bundle_a",
        source_digest=scene.source_digest,
    )
    second, second_paths = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_QualifiedBackend(),
        resolution=(640, 480),
        output_dir=tmp_path / "bundle_b" / "rig_evidence",
        artifact_base_dir=tmp_path / "bundle_b",
        source_digest=scene.source_digest,
    )

    assert first == second
    assert first_paths != second_paths
    assert first["schema"] == VERIFICATION_EVIDENCE_SCHEMA
    assert first["artifact_path_base"] == "rig_document_directory"
    assert first["scope"] == "ovrtx_rgb_artifact_grounding"
    assert [item["camera"] for item in first["artifacts"]] == paths
    assert all(
        not Path(item["relative_path"]).is_absolute() for item in first["artifacts"]
    )
    assert all(
        item["relative_path"].startswith("rig_evidence/") for item in first["artifacts"]
    )
    assert all("render_time_s" not in item for item in first["artifacts"])
    assert first["assertions"]["metric_distance_verified"] is False
    unsigned = copy.deepcopy(first)
    evidence_digest = unsigned.pop("evidence_digest")
    assert evidence_digest == canonical_json_digest(unsigned)


def test_verification_fails_before_render_for_unqualified_runtime(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)

    class OldBackend(_QualifiedBackend):
        def runtime_identity(self):
            identity = super().runtime_identity()
            identity["runtime_versions"]["ovrtx"] = "0.3.0.312915"
            return identity

    backend = OldBackend()
    with pytest.raises(RuntimeError, match="failed qualification: ovrtx"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=backend,
            resolution=(640, 480),
            output_dir=tmp_path / "evidence",
            source_digest=scene.source_digest,
        )
    assert backend.render_called is False
    assert not (tmp_path / "evidence").exists()


def test_verification_rejects_evidence_directory_outside_rig_base(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)
    backend = _QualifiedBackend()

    with pytest.raises(ValueError, match="within the rig document directory"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=backend,
            resolution=(64, 48),
            output_dir=tmp_path / "outside" / "evidence",
            artifact_base_dir=tmp_path / "bundle",
            source_digest=scene.source_digest,
        )
    assert backend.render_called is False
    assert not (tmp_path / "outside").exists()


def test_bundled_remote_verification_fails_closed_without_worker_identity(tmp_path):
    from usd_core.render.remote import RemoteRenderBackend

    assert not callable(getattr(RemoteRenderBackend, "runtime_identity", None))
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)

    class BundledRemoteContract:
        name = "remote"

        def render(self, *_args, **_kwargs):
            pytest.fail("remote rendering must not run before worker qualification")

    with pytest.raises(RuntimeError, match="does not provide it"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=BundledRemoteContract(),
            resolution=(64, 48),
            output_dir=tmp_path / "evidence",
            source_digest=scene.source_digest,
        )
    assert not (tmp_path / "evidence").exists()


def test_rig_document_binds_evidence_and_requires_descriptor_bound_publication(tmp_path):
    stage, paths = _rig_stage()
    scene = build_scene_analysis_ir(stage)
    first_evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_QualifiedBackend(),
        resolution=(640, 480),
        output_dir=tmp_path / "a" / "rig_evidence",
        artifact_base_dir=tmp_path / "a",
        source_digest=scene.source_digest,
    )
    second_evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_QualifiedBackend(),
        resolution=(640, 480),
        output_dir=tmp_path / "b" / "rig_evidence",
        artifact_base_dir=tmp_path / "b",
        source_digest=scene.source_digest,
    )
    assert first_evidence == second_evidence
    first = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(640, 480),
        verification=first_evidence,
    )
    second = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(640, 480),
        verification=second_evidence,
    )
    assert first == second
    assert len({camera["definition_digest"] for camera in first["cameras"]}) == 2
    assert "rig JSON" in first["conventions"]["verification_artifact_paths"]
    assert first["verification"]["evidence_digest"] == first_evidence["evidence_digest"]

    for document, output in (
        (first, tmp_path / "a" / "rig.json"),
        (second, tmp_path / "b" / "rig.json"),
    ):
        with pytest.raises(ValueError, match="descriptor-bound verification generation"):
            publish_rig_document(document, output)
        assert not output.exists()

    tampered = copy.deepcopy(first_evidence)
    tampered["artifacts"][0]["relative_path"] = "../outside.png"
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = canonical_json_digest(tampered)
    with pytest.raises(ValueError, match="unsafe or duplicate relative path"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(640, 480),
            verification=tampered,
        )

    windows_escape = copy.deepcopy(first_evidence)
    windows_escape["artifacts"][0]["relative_path"] = "..\\outside.png"
    windows_escape.pop("evidence_digest")
    windows_escape["evidence_digest"] = canonical_json_digest(windows_escape)
    with pytest.raises(ValueError, match="unsafe or duplicate relative path"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(640, 480),
            verification=windows_escape,
        )

    noncanonical = copy.deepcopy(first_evidence)
    noncanonical["artifacts"][0]["relative_path"] = noncanonical["artifacts"][0][
        "relative_path"
    ].replace("/", "//", 1)
    noncanonical.pop("evidence_digest")
    noncanonical["evidence_digest"] = canonical_json_digest(noncanonical)
    with pytest.raises(ValueError, match="unsafe or duplicate relative path"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(640, 480),
            verification=noncanonical,
        )

    unsafe_document = copy.deepcopy(first)
    unsafe_document["verification"]["artifacts"][0]["relative_path"] = "../escape.png"
    with pytest.raises(ValueError, match="unsafe or duplicate relative path"):
        publish_rig_document(unsafe_document, tmp_path / "unsafe" / "rig.json")


def test_rig_export_rejects_duplicate_stable_camera_ids():
    stage, paths = _rig_stage()
    stage.GetPrimAtPath(paths[1]).SetCustomDataByKey("usdCameraStableId", "camera-001")
    scene = build_scene_analysis_ir(stage)
    with pytest.raises(ValueError, match="duplicate stable camera IDs"):
        build_rig_document(stage, scene, "/World/Rig", resolution=(640, 480))


def test_rig_document_preserves_detached_per_camera_visibility(tmp_path):
    stage, paths = _rig_stage()
    scene = build_scene_analysis_ir(stage)
    coverage = {
        "schema": "usd-cli.camera-coverage.v1",
        "source_digest": scene.source_digest,
        "analysis_policy": scene.policy.as_dict(),
        "analysis_policy_digest": scene.policy.digest,
        "overlap_histogram": {"0": 2, "1": 6},
        "cameras": [
            {
                "camera": path,
                "visible_cells": index,
                "visibility": {
                    "schema": "usd-cli.visibility-grid.v1",
                    "regions": [{"outline": [[float(index), 0.0, 0.0]], "holes": []}],
                },
            }
            for index, path in enumerate(paths, start=1)
        ],
    }

    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(640, 480),
        coverage_report=coverage,
    )
    assert document["coverage"]["cameras"] == coverage["cameras"]
    assert document["cameras"][0]["visibility"] == coverage["cameras"][0]["visibility"]
    coverage["cameras"][0]["visibility"]["regions"].clear()
    assert document["cameras"][0]["visibility"]["regions"]

    changed_policy = copy.deepcopy(document)
    changed_policy["coverage"]["analysis_policy_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="coverage analysis policy digest"):
        publish_rig_document(changed_policy, tmp_path / "changed-policy-rig.json")


def test_camera_analysis_reports_use_schema_agnostic_atomic_publisher(tmp_path):
    report = {
        "schema": "usd-cli.camera-coverage.v1",
        "coverage_fraction": 0.75,
    }
    path, digest = publish_json_document(report, tmp_path / "coverage.json")

    assert json.loads(Path(path).read_text()) == report
    assert digest.startswith("sha256:")
    with pytest.raises(ValueError, match="unsupported schema"):
        publish_rig_document(report, tmp_path / "not-a-rig.json")


def test_ovrtx_worker_exposes_actual_runtime_identity_without_changing_render_contract():
    from usd_core.render.ovrtx import _DAEMON_SCRIPT, OvRTXRenderBackend

    assert '"camera_observation_v1"' in _DAEMON_SCRIPT
    assert '"warp_dlpack_reduction_v1"' in _DAEMON_SCRIPT
    assert '"semantic_overlay_v1"' in _DAEMON_SCRIPT
    assert '"stage_transport": "ovstage_attached_ordinals"' in _DAEMON_SCRIPT
    assert "WORKER_PROTOCOL_VERSION = 3" in _DAEMON_SCRIPT
    assert '_distribution_version("ovstage")' in _DAEMON_SCRIPT
    assert "wp.from_dlpack(mapped)" in _DAEMON_SCRIPT
    assert "mapped.unmap(stream=stream.cuda_stream)" in _DAEMON_SCRIPT

    class FakeDaemon:
        def alive(self):
            return True

        def runtime_identity(self):
            return {"runtime_versions": dict(QUALIFIED_RUNTIME_VERSIONS)}

    backend = OvRTXRenderBackend()
    backend._daemon = FakeDaemon()
    assert backend.runtime_identity()["runtime_versions"] == QUALIFIED_RUNTIME_VERSIONS


def test_typed_observation_evidence_binds_metric_semantic_and_warp_reductions(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    floor = UsdGeom.Cube.Define(stage, "/World/Floor").GetPrim().GetPath().pathString
    scene = build_scene_analysis_ir(stage)
    evidence, artifacts = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(64, 48),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={floor: "floor_scope"},
        configuration={"seed": 17, "analytic_backend": {"newton": "1.5.0"}},
    )

    assert evidence["schema_version"] == 2
    assert evidence["scope"] == "ovrtx_metric_semantic_aov_evidence"
    assert evidence["analysis_time"] == USD_DEFAULT_ANALYSIS_TIME
    assert evidence["assertions"]["metric_distance_verified"] is True
    assert evidence["assertions"]["semantic_identity_verified"] is True
    assert evidence["assertions"]["warp_dlpack_reduction_verified"] is True
    assert evidence["assertions"]["newton_ovrtx_parity_verified"] is False
    assert set(evidence["observations"][0]["aovs"]) == set(OBSERVATION_AOVS)
    assert len(artifacts) == 1

    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(64, 48),
        verification=evidence,
    )
    assert document["verification"]["evidence_digest"] == evidence["evidence_digest"]


def test_verification_clone_hides_policy_exclusions_and_helpers_only(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    UsdGeom.Cube.Define(stage, "/World/Floor")
    UsdGeom.Cube.Define(stage, "/World/Excluded")
    UsdGeom.Cube.Define(stage, "/World/Helper")
    policy = SceneAnalysisPolicy(
        scope_paths=("/World/Floor",),
        exclude_paths=("/World/Excluded",),
        helper_paths=("/World/Helper",),
        floor_paths=("/World/Floor",),
    )
    scene = build_scene_analysis_ir(stage, policy=policy)
    source_before = stage.GetRootLayer().ExportToString()

    class PolicyInspectingBackend(_ObservationBackend):
        clone_checked = False

        def observe(self, clone, *args, **kwargs):
            assert (
                UsdGeom.Imageable(
                    clone.GetPrimAtPath("/World/Floor")
                ).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            )
            for hidden in ("/World/Excluded", "/World/Helper"):
                assert (
                    UsdGeom.Imageable(clone.GetPrimAtPath(hidden)).ComputeVisibility()
                    == UsdGeom.Tokens.invisible
                )
            assert (
                UsdGeom.Imageable(clone.GetPrimAtPath(paths[0])).ComputeVisibility()
                != UsdGeom.Tokens.invisible
            )
            self.clone_checked = True
            return super().observe(clone, *args, **kwargs)

    backend = PolicyInspectingBackend()
    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=backend,
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={"/World/Floor": "floor_scope"},
        analysis_scene=scene,
        analysis_policy=policy,
    )

    assert backend.clone_checked is True
    assert evidence["analysis_policy"] == policy.as_dict()
    assert evidence["analysis_policy_digest"] == policy.digest
    assert evidence["assertions"]["analysis_world_alignment_verified"] is True
    assert evidence["clone_visibility_overlay"] == {
        "scope": "verification_clone_only",
        "policy_digest": policy.digest,
        "actions": [
            {
                "path": "/World/Excluded",
                "operation": "set_visibility_invisible",
                "reason": "absent_from_analysis_world",
            },
            {
                "path": "/World/Helper",
                "operation": "set_visibility_invisible",
                "reason": "helper_geometry",
            },
        ],
        "expanded_instance_roots": [],
    }
    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=evidence,
    )
    assert document["source"]["analysis_policy_digest"] == policy.digest
    assert stage.GetRootLayer().ExportToString() == source_before


def test_verification_fails_closed_for_hidden_gprim_ancestor(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    UsdGeom.Cube.Define(stage, "/World/Parent")
    UsdGeom.Cube.Define(stage, "/World/Parent/AllowedChild")
    policy = SceneAnalysisPolicy(include_paths=("/World/Parent/AllowedChild",))
    scene = build_scene_analysis_ir(stage, policy=policy)
    backend = _ObservationBackend()

    with pytest.raises(ValueError, match="excluded ancestor Gprim"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=backend,
            resolution=(16, 12),
            output_dir=tmp_path / "evidence",
            artifact_base_dir=tmp_path,
            source_digest=scene.source_digest,
            analysis_scene=scene,
            analysis_policy=policy,
        )
    assert not backend.render_called
    assert not (tmp_path / "evidence").exists()


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (
            SceneAnalysisPolicy(exclude_paths=("/World/ExcludedInstances",)),
            "absent_from_analysis_world",
        ),
        (
            SceneAnalysisPolicy(helper_paths=("/World/ExcludedInstances",)),
            "helper_geometry",
        ),
    ],
)
def test_verification_clone_hides_omitted_point_instancer(
    tmp_path, policy, reason
):
    stage, paths = _rig_stage(cameras=1)
    prototype = UsdGeom.Cube.Define(stage, "/World/Prototype")
    instancer = UsdGeom.PointInstancer.Define(stage, "/World/ExcludedInstances")
    instancer.CreatePrototypesRel().SetTargets([prototype.GetPath()])
    instancer.CreateProtoIndicesAttr().Set([0])
    instancer.CreatePositionsAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0)])
    scene = build_scene_analysis_ir(stage, policy=policy)

    class InstancerInspectingBackend(_ObservationBackend):
        def observe(self, clone, *args, **kwargs):
            assert (
                UsdGeom.Imageable(
                    clone.GetPrimAtPath("/World/ExcludedInstances")
                ).ComputeVisibility()
                == UsdGeom.Tokens.invisible
            )
            return super().observe(clone, *args, **kwargs)

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=InstancerInspectingBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        analysis_scene=scene,
        analysis_policy=policy,
    )

    assert {
        "path": "/World/ExcludedInstances",
        "operation": "set_visibility_invisible",
        "reason": reason,
    } in evidence["clone_visibility_overlay"]["actions"]


def test_verification_generation_hashes_the_opened_directory_during_path_swap(
    tmp_path,
):
    evidence_root = tmp_path / "rig_evidence"
    generation = evidence_root / "generation"
    generation.mkdir(parents=True)
    (generation / "camera.png").write_bytes(b"unexpected-original-bytes")
    matching = tmp_path / "matching-generation"
    matching.mkdir()
    expected = b"expected-evidence-bytes"
    (matching / "camera.png").write_bytes(expected)
    verification = {
        "artifacts": [
            {
                "relative_path": "rig_evidence/generation/camera.png",
                "sha256": "sha256:" + hashlib.sha256(expected).hexdigest(),
                "size_bytes": len(expected),
            }
        ]
    }
    generation_fd = os.open(
        generation,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    held = evidence_root / "held-generation"
    generation.rename(held)
    generation.symlink_to(matching, target_is_directory=True)
    try:
        with pytest.raises(ValueError, match="differs from staged evidence"):
            validate_verification_generation(
                verification,
                artifact_base_dir=tmp_path,
                generation_dir_fd=generation_fd,
                generation_relative_path="rig_evidence/generation",
            )
    finally:
        os.close(generation_fd)
        generation.unlink()
        held.rename(generation)


def test_verification_clone_makes_admitted_triangle_two_sided(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    mesh = UsdGeom.Mesh.Define(stage, "/World/Triangle")
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreatePointsAttr().Set(
        [Gf.Vec3f(-1.0, 0.0, 0.0), Gf.Vec3f(1.0, 0.0, 0.0), Gf.Vec3f(0.0, 1.0, 0.0)]
    )
    mesh.CreateFaceVertexCountsAttr().Set([3])
    mesh.CreateFaceVertexIndicesAttr().Set([0, 1, 2])
    assert mesh.GetDoubleSidedAttr().Get() is False
    scene = build_scene_analysis_ir(stage)

    class SidednessInspectingBackend(_ObservationBackend):
        def observe(self, clone, *args, **kwargs):
            clone_mesh = UsdGeom.Mesh(clone.GetPrimAtPath("/World/Triangle"))
            assert clone_mesh.GetDoubleSidedAttr().Get() is True
            return super().observe(clone, *args, **kwargs)

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=SidednessInspectingBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        analysis_scene=scene,
    )

    assert evidence["clone_visibility_overlay"]["actions"] == [
        {
            "path": "/World/Triangle",
            "operation": "set_double_sided_true",
            "reason": "analytic_meshes_are_two_sided",
        }
    ]
    build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=evidence,
    )
    assert mesh.GetDoubleSidedAttr().Get() is False

    omitted = copy.deepcopy(evidence)
    omitted["clone_visibility_overlay"]["actions"] = []
    omitted.pop("evidence_digest")
    omitted["evidence_digest"] = canonical_json_digest(omitted)
    with pytest.raises(ValueError, match="actions differ from the exported"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=omitted,
        )


def test_cancellation_after_analytic_prep_prevents_ovrtx_dispatch(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)
    event = threading.Event()

    class CancellingAnalyticBackend:
        def render_cameras(self, request):
            event.set()
            return SimpleNamespace(
                depth_m=np.ones((1, request.height, request.width), dtype=np.float32),
                shape_ids=np.zeros((1, request.height, request.width), dtype=np.int64),
            )

    class NeverDispatchedBackend(_ObservationBackend):
        def observe(self, *_args, **_kwargs):
            pytest.fail("cancelled analytic preparation must not dispatch OVRTX")

    output = tmp_path / "evidence"
    with cancellation_scope(event), pytest.raises(CameraAnalysisCancelled):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=NeverDispatchedBackend(),
            resolution=(16, 12),
            output_dir=output,
            artifact_base_dir=tmp_path,
            source_digest=scene.source_digest,
            analytic_backend=CancellingAnalyticBackend(),
            analysis_scene=scene,
        )
    assert not output.exists() or not any(output.rglob("*"))


def test_observation_verification_expands_nested_instance_proxies_on_clone_only(
    tmp_path,
):
    stage, paths = _rig_stage(cameras=1)
    UsdGeom.Xform.Define(stage, "/World/LeafAsset")
    UsdGeom.Cube.Define(stage, "/World/LeafAsset/Target")
    UsdGeom.Xform.Define(stage, "/World/MiddleAsset")
    inner = UsdGeom.Xform.Define(stage, "/World/MiddleAsset/InnerInstance").GetPrim()
    assert inner.GetReferences().AddInternalReference("/World/LeafAsset")
    assert inner.SetInstanceable(True)
    outer = UsdGeom.Xform.Define(stage, "/World/OuterInstance").GetPrim()
    assert outer.GetReferences().AddInternalReference("/World/MiddleAsset")
    assert outer.SetInstanceable(True)
    target = "/World/OuterInstance/InnerInstance/Target"
    source_target = stage.GetPrimAtPath(target)
    assert source_target.IsInstanceProxy()
    root_before = stage.GetRootLayer().ExportToString()
    session_before = stage.GetSessionLayer().ExportToString()
    scene = build_scene_analysis_ir(stage)

    class CloneInspectingBackend(_ObservationBackend):
        clone_checked = False

        def observe(self, clone, *args, semantic_labels, **kwargs):
            assert not clone.GetPrimAtPath(target).IsInstanceProxy()
            assert not clone.GetPrimAtPath("/World/OuterInstance").IsInstanceable()
            assert not clone.GetPrimAtPath(
                "/World/OuterInstance/InnerInstance"
            ).IsInstanceable()
            assert semantic_labels == {target: semantic_label(target, "look_at_target")}
            self.clone_checked = True
            return super().observe(
                clone, *args, semantic_labels=semantic_labels, **kwargs
            )

    backend = CloneInspectingBackend()
    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=backend,
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={target: "look_at_target"},
    )

    expansion = {
        "scope": "verification_clone_only",
        "operation": "set_instanceable_false",
        "expanded_roots": [
            "/World/OuterInstance",
            "/World/OuterInstance/InnerInstance",
        ],
    }
    assert backend.clone_checked is True
    assert evidence["clone_instance_expansion"] == expansion
    assert evidence["semantic_assignments"] == [
        {
            "path": target,
            "role": "look_at_target",
            "label": semantic_label(target, "look_at_target"),
        }
    ]
    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=evidence,
    )
    assert document["verification"]["clone_instance_expansion"] == expansion
    assert stage.GetRootLayer().ExportToString() == root_before
    assert stage.GetSessionLayer().ExportToString() == session_before
    assert stage.GetPrimAtPath(target).IsInstanceProxy()
    assert stage.GetPrimAtPath("/World/OuterInstance").IsInstance()

    reordered = copy.deepcopy(evidence)
    reordered["clone_instance_expansion"]["expanded_roots"].reverse()
    reordered.pop("evidence_digest")
    reordered["evidence_digest"] = canonical_json_digest(reordered)
    with pytest.raises(ValueError, match="nested instance roots out of order"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=reordered,
        )


def test_verified_camera_without_authored_id_uses_shared_path_hash(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    camera_prim = stage.GetPrimAtPath(paths[0])
    camera_prim.ClearCustomDataByKey("usdCameraStableId")
    expected_id, expected_source = camera_stable_identity(camera_prim, paths[0])
    scene = build_scene_analysis_ir(stage)
    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
    )
    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=evidence,
    )

    assert expected_id.startswith("camera-path-sha256:")
    assert expected_source == "derived_prim_path"
    assert evidence["camera_definitions"][0]["stable_id"] == expected_id
    assert document["cameras"][0]["id"] == expected_id
    assert document["cameras"][0]["id_source"] == expected_source


def test_observation_verification_rejects_reused_artifact_path(tmp_path):
    stage, paths = _rig_stage(cameras=2)
    scene = build_scene_analysis_ir(stage)

    class ReusedArtifactBackend(_ObservationBackend):
        def observe(self, *args, **kwargs):
            results = super().observe(*args, **kwargs)
            results[1].artifacts["LdrColor"] = results[0].artifacts["LdrColor"]
            return results

    with pytest.raises(RuntimeError, match="reused artifact path"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=ReusedArtifactBackend(),
            resolution=(16, 12),
            output_dir=tmp_path / "evidence",
            source_digest=scene.source_digest,
        )


def test_metric_verification_rejects_all_zero_distance_summary(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)

    class ZeroMetricBackend(_ObservationBackend):
        def observe(self, *args, **kwargs):
            results = super().observe(*args, **kwargs)
            for result in results:
                result.aovs["DistanceToCameraSD"]["statistics"].update(
                    {"nonzero_count": 0, "minimum": 0.0, "maximum": 0.0}
                )
            return results

    with pytest.raises(RuntimeError, match="no positive finite DistanceToCameraSD"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=ZeroMetricBackend(),
            resolution=(16, 12),
            output_dir=tmp_path / "zero_metric_evidence",
            artifact_base_dir=tmp_path,
            source_digest=scene.source_digest,
        )


def test_semantic_verification_rejects_requested_label_with_zero_pixels(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    target = UsdGeom.Cube.Define(stage, "/World/Target").GetPath().pathString
    scene = build_scene_analysis_ir(stage)
    valid, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "valid_evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={target: "look_at_target"},
    )

    tampered = copy.deepcopy(valid)
    tampered["observations"][0]["aovs"]["SemanticSegmentation"]["statistics"][
        "semantic_labels"
    ][0]["pixels"] = 0
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = canonical_json_digest(tampered)
    with pytest.raises(ValueError, match="does not support its semantic assertion"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=tampered,
        )

    background_id = copy.deepcopy(valid)
    background_id["observations"][0]["aovs"]["SemanticSegmentation"]["statistics"][
        "semantic_labels"
    ][0]["id"] = 0
    background_id.pop("evidence_digest")
    background_id["evidence_digest"] = canonical_json_digest(background_id)
    with pytest.raises(ValueError, match="does not support its semantic assertion"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=background_id,
        )

    class ZeroPixelBackend(_ObservationBackend):
        def observe(self, *args, **kwargs):
            results = super().observe(*args, **kwargs)
            for result in results:
                labels = result.aovs["SemanticSegmentation"]["statistics"][
                    "semantic_labels"
                ]
                labels[0]["pixels"] = 0
            return results

    with pytest.raises(RuntimeError, match="could not bind"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=ZeroPixelBackend(),
            resolution=(16, 12),
            output_dir=tmp_path / "zero_pixel_evidence",
            artifact_base_dir=tmp_path,
            source_digest=scene.source_digest,
            semantic_roles={target: "look_at_target"},
        )

    class BackgroundIdBackend(_ObservationBackend):
        def observe(self, *args, **kwargs):
            results = super().observe(*args, **kwargs)
            for result in results:
                labels = result.aovs["SemanticSegmentation"]["statistics"][
                    "semantic_labels"
                ]
                labels[0]["id"] = 0
            return results

    with pytest.raises(RuntimeError, match="could not bind"):
        verify_rig_with_ovrtx(
            stage,
            paths,
            backend=BackgroundIdBackend(),
            resolution=(16, 12),
            output_dir=tmp_path / "background_id_evidence",
            artifact_base_dir=tmp_path,
            source_digest=scene.source_digest,
            semantic_roles={target: "look_at_target"},
        )


def test_rig_rejects_self_digested_observation_claim_inconsistencies(tmp_path):
    stage, paths = _rig_stage(cameras=2)
    scene = build_scene_analysis_ir(stage)
    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
    )

    cases = []
    duplicate_product = copy.deepcopy(evidence)
    duplicate_product["observations"][1]["render_product"] = duplicate_product[
        "observations"
    ][0]["render_product"]
    cases.append((duplicate_product, "invalid or duplicate render product"))

    empty_metric = copy.deepcopy(evidence)
    empty_metric["observations"][0]["aovs"]["DistanceToCameraSD"]["statistics"][
        "valid_count"
    ] = 0
    cases.append((empty_metric, "does not support its metric assertion"))

    zero_metric = copy.deepcopy(evidence)
    zero_metric["observations"][0]["aovs"]["DistanceToCameraSD"]["statistics"].update(
        {"nonzero_count": 0, "minimum": 0.0, "maximum": 0.0}
    )
    cases.append((zero_metric, "does not support its metric assertion"))

    truncated_aov = copy.deepcopy(evidence)
    truncated_aov["observations"][0]["aovs"]["NormalSD"]["statistics"][
        "element_count"
    ] -= 1
    cases.append((truncated_aov, "unqualified AOV reduction"))

    inconsistent_qualification = copy.deepcopy(evidence)
    inconsistent_qualification["runtime_qualification"]["reported_versions"]["warp"] = (
        "forged"
    )
    cases.append((inconsistent_qualification, "runtime qualification did not pass"))

    numeric_time = copy.deepcopy(evidence)
    numeric_time["analysis_time"] = {"kind": "numeric", "time_code": 0.0}
    cases.append((numeric_time, "not bound to USD Default analysis time"))

    for tampered, message in cases:
        tampered.pop("evidence_digest")
        tampered["evidence_digest"] = canonical_json_digest(tampered)
        with pytest.raises(ValueError, match=message):
            build_rig_document(
                stage,
                scene,
                "/World/Rig",
                resolution=(16, 12),
                verification=tampered,
            )


def test_parity_treats_negative_ovrtx_depth_as_a_miss(tmp_path):
    depth_path = tmp_path / "depth.npy"
    semantic_path = tmp_path / "semantic.npy"
    np.save(
        depth_path,
        np.asarray([[[-1.0], [1.0]]], dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        semantic_path,
        np.zeros((1, 2, 1), dtype=np.uint32),
        allow_pickle=False,
    )
    analytic = SimpleNamespace(
        depth_m=np.asarray([[[-1.0, 1.0]]], dtype=np.float32),
        shape_ids=np.asarray([[[-1, 0]]], dtype=np.int64),
    )

    parity = _newton_ovrtx_parity(
        "/World/Rig/Camera_001",
        analytic_observation=analytic,
        scene=SimpleNamespace(shape_path_by_id={}),
        aovs={"SemanticSegmentation": {"statistics": {"semantic_labels": []}}},
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={},
    )

    assert parity["depth"]["ovrtx_valid_pixels"] == 1
    assert parity["depth"]["valid_mask_iou"] == pytest.approx(1.0)
    assert parity["passed"] is True


def test_parity_rejects_requested_semantics_when_both_masks_are_empty(tmp_path):
    depth_path = tmp_path / "empty_semantic_depth.npy"
    semantic_path = tmp_path / "empty_semantic.npy"
    np.save(depth_path, np.ones((2, 2, 1), dtype=np.float32), allow_pickle=False)
    np.save(semantic_path, np.zeros((2, 2, 1), dtype=np.uint32), allow_pickle=False)
    target = "/World/Target"
    label = semantic_label(target, "look_at_target")

    parity = _newton_ovrtx_parity(
        "/World/Rig/Camera_001",
        analytic_observation=SimpleNamespace(
            depth_m=np.ones((1, 2, 2), dtype=np.float32),
            shape_ids=np.zeros((1, 2, 2), dtype=np.int64),
        ),
        scene=SimpleNamespace(shape_path_by_id={0: "/World/Other"}),
        aovs={
            "SemanticSegmentation": {
                "statistics": {
                    "semantic_labels": [
                        {"id": 1, "label": f"usd_cli: {label};", "pixels": 1}
                    ]
                }
            }
        },
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={target: label},
    )

    assert parity["semantics"]["ovrtx_target_pixels"] == 0
    assert parity["semantics"]["union_pixels"] == 0
    assert parity["semantics"]["mask_iou"] == pytest.approx(0.0)
    assert parity["semantics"]["passed"] is False
    assert parity["passed"] is False


def test_parity_rejects_swapped_requested_semantic_labels(tmp_path):
    depth_path = tmp_path / "swapped_label_depth.npy"
    semantic_path = tmp_path / "swapped_label_semantic.npy"
    np.save(depth_path, np.ones((1, 2, 1), dtype=np.float32), allow_pickle=False)
    # The aggregate requested-object mask is perfect, but A and B are exchanged.
    np.save(
        semantic_path,
        np.asarray([[[2], [1]]], dtype=np.uint32),
        allow_pickle=False,
    )
    path_a, path_b = "/World/A", "/World/B"
    label_a = semantic_label(path_a, "target")
    label_b = semantic_label(path_b, "floor")

    parity = _newton_ovrtx_parity(
        "/World/Rig/Camera_001",
        analytic_observation=SimpleNamespace(
            depth_m=np.ones((1, 1, 2), dtype=np.float32),
            shape_ids=np.asarray([[[0, 1]]], dtype=np.int64),
        ),
        scene=SimpleNamespace(shape_path_by_id={0: path_a, 1: path_b}),
        aovs={
            "SemanticSegmentation": {
                "statistics": {
                    "semantic_labels": [
                        {"id": 1, "label": f"usd_cli: {label_a};", "pixels": 1},
                        {"id": 2, "label": f"usd_cli: {label_b};", "pixels": 1},
                    ]
                }
            }
        },
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={path_a: label_a, path_b: label_b},
    )

    assert parity["semantics"]["mask_iou"] == pytest.approx(1.0)
    assert [item["mask_iou"] for item in parity["semantics"]["labels"]] == [
        pytest.approx(0.0),
        pytest.approx(0.0),
    ]
    assert parity["semantics"]["passed"] is False
    assert parity["passed"] is False


def test_nested_floor_label_remains_part_of_ancestor_target_mask(tmp_path):
    depth_path = tmp_path / "nested_depth.npy"
    semantic_path = tmp_path / "nested_semantic.npy"
    np.save(depth_path, np.ones((1, 2, 1), dtype=np.float32), allow_pickle=False)
    np.save(
        semantic_path,
        np.asarray([[[1], [2]]], dtype=np.uint32),
        allow_pickle=False,
    )
    target = "/World/Object"
    body = "/World/Object/Body"
    floor = "/World/Object/Floor"
    target_label = semantic_label(target, "target")
    floor_label = semantic_label(floor, "target_and_floor")

    parity = _newton_ovrtx_parity(
        "/World/Rig/Camera_001",
        analytic_observation=SimpleNamespace(
            depth_m=np.ones((1, 1, 2), dtype=np.float32),
            shape_ids=np.asarray([[[0, 1]]], dtype=np.int64),
        ),
        scene=SimpleNamespace(shape_path_by_id={0: body, 1: floor}),
        aovs={
            "SemanticSegmentation": {
                "statistics": {
                    "semantic_labels": [
                        {
                            "id": 1,
                            "label": f"usd_cli: {target_label};",
                            "pixels": 1,
                        },
                        {
                            "id": 2,
                            "label": f"usd_cli: {floor_label};",
                            "pixels": 1,
                        },
                    ]
                }
            }
        },
        artifact_map={
            "DistanceToCameraSD": str(depth_path),
            "SemanticSegmentation": str(semantic_path),
        },
        semantic_labels={target: target_label, floor: floor_label},
    )

    assert [
        item["mask_iou"] for item in parity["semantics"]["labels"]
    ] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert parity["passed"] is True


def test_fully_nested_floor_uses_one_combined_semantic_assignment():
    floor = "/World/Object/Floor"
    scene = SimpleNamespace(
        shapes=(SimpleNamespace(prim_path=floor, analysis_role="target"),)
    )

    roles = verification_semantic_roles(
        scene,
        target_path="/World/Object",
        floor_paths=[floor],
    )

    assert roles == {floor: "target_and_floor"}


def test_rig_recomputes_declared_parity_tolerances(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    scene = build_scene_analysis_ir(stage)

    class MatchingAnalyticBackend:
        def render_cameras(self, request):
            return SimpleNamespace(
                depth_m=np.ones((1, request.height, request.width), dtype=np.float32),
                shape_ids=np.zeros((1, request.height, request.width), dtype=np.int64),
            )

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(64, 48),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        analytic_backend=MatchingAnalyticBackend(),
        analysis_scene=scene,
    )
    assert evidence["assertions"]["newton_ovrtx_parity_verified"] is True
    depth = evidence["newton_ovrtx_parity"][0]["depth"]
    assert depth["median_distance_m"] == pytest.approx(1.0)
    expected_mean, expected_percentile = parity_v1_depth_tolerances(
        depth["median_distance_m"]
    )
    assert depth["mean_abs_tolerance_m"] == pytest.approx(expected_mean)
    assert depth["percentile_95_tolerance_m"] == pytest.approx(expected_percentile)
    assert depth["minimum_valid_mask_iou"] == PARITY_V1_MINIMUM_MASK_IOU

    tampered = copy.deepcopy(evidence)
    tampered_depth = tampered["newton_ovrtx_parity"][0]["depth"]
    tampered_depth["mean_abs_error_m"] = tampered_depth["mean_abs_tolerance_m"] + 1.0
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = canonical_json_digest(tampered)
    with pytest.raises(ValueError, match="parity record did not pass"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(64, 48),
            verification=tampered,
        )

    for field, value in (
        ("minimum_valid_mask_iou", 0.10),
        ("mean_abs_tolerance_m", expected_mean + 1.0),
        ("percentile_95_tolerance_m", expected_percentile + 1.0),
    ):
        self_declared = copy.deepcopy(evidence)
        self_declared["newton_ovrtx_parity"][0]["depth"][field] = value
        self_declared.pop("evidence_digest")
        self_declared["evidence_digest"] = canonical_json_digest(self_declared)
        with pytest.raises(ValueError, match="parity record did not pass"):
            build_rig_document(
                stage,
                scene,
                "/World/Rig",
                resolution=(64, 48),
                verification=self_declared,
            )

    missing_median = copy.deepcopy(evidence)
    missing_median["newton_ovrtx_parity"][0]["depth"].pop("median_distance_m")
    missing_median.pop("evidence_digest")
    missing_median["evidence_digest"] = canonical_json_digest(missing_median)
    with pytest.raises(ValueError, match="parity record did not pass"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(64, 48),
            verification=missing_median,
        )

    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(64, 48),
        verification=evidence,
    )
    document["verification"]["newton_ovrtx_parity"][0]["depth"][
        "minimum_valid_mask_iou"
    ] = 0.10
    document["verification"].pop("evidence_digest")
    document["verification"]["evidence_digest"] = canonical_json_digest(
        document["verification"]
    )
    with pytest.raises(ValueError, match="parity record did not pass"):
        publish_rig_document(document, tmp_path / "tampered-policy-rig.json")


def test_rig_rejects_self_declared_semantic_parity_threshold(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    target = UsdGeom.Cube.Define(stage, "/World/Target").GetPath().pathString
    scene = build_scene_analysis_ir(stage)
    target_shape_id = next(
        shape_id for shape_id, path in scene.shape_path_by_id.items() if path == target
    )

    class MatchingSemanticAnalyticBackend:
        def render_cameras(self, request):
            return SimpleNamespace(
                depth_m=np.ones((1, request.height, request.width), dtype=np.float32),
                shape_ids=np.full(
                    (1, request.height, request.width),
                    target_shape_id,
                    dtype=np.int64,
                ),
            )

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={target: "look_at_target"},
        analytic_backend=MatchingSemanticAnalyticBackend(),
        analysis_scene=scene,
    )
    tampered = copy.deepcopy(evidence)
    semantics = tampered["newton_ovrtx_parity"][0]["semantics"]
    label = semantics["labels"][0]
    for record in (semantics, label):
        record.update(
            {
                "mask_iou": 1.0 / 3.0,
                "minimum_mask_iou": 0.10,
                "ovrtx_target_pixels": 2,
                "newton_target_pixels": 2,
                "intersection_pixels": 1,
                "union_pixels": 3,
                "passed": True,
            }
        )
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = canonical_json_digest(tampered)

    with pytest.raises(ValueError, match="parity record did not pass"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=tampered,
        )


def test_rig_rejects_self_digested_empty_semantic_parity(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    target = UsdGeom.Cube.Define(stage, "/World/Target").GetPath().pathString
    scene = build_scene_analysis_ir(stage)
    target_shape_id = next(
        shape_id for shape_id, path in scene.shape_path_by_id.items() if path == target
    )

    class MatchingSemanticAnalyticBackend:
        def render_cameras(self, request):
            return SimpleNamespace(
                depth_m=np.ones((1, request.height, request.width), dtype=np.float32),
                shape_ids=np.full(
                    (1, request.height, request.width),
                    target_shape_id,
                    dtype=np.int64,
                ),
            )

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=_ObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={target: "look_at_target"},
        analytic_backend=MatchingSemanticAnalyticBackend(),
        analysis_scene=scene,
    )
    assert evidence["newton_ovrtx_parity"][0]["semantics"]["union_pixels"] > 0

    tampered = copy.deepcopy(evidence)
    semantics = tampered["newton_ovrtx_parity"][0]["semantics"]
    semantics.update(
        {
            "mask_iou": 1.0,
            "ovrtx_target_pixels": 0,
            "newton_target_pixels": 0,
            "intersection_pixels": 0,
            "union_pixels": 0,
            "passed": True,
        }
    )
    tampered.pop("evidence_digest")
    tampered["evidence_digest"] = canonical_json_digest(tampered)
    with pytest.raises(ValueError, match="parity record did not pass"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=tampered,
        )

    wrong_label_binding = copy.deepcopy(evidence)
    wrong_label_binding["newton_ovrtx_parity"][0]["semantics"]["labels"][0]["path"] = (
        "/World/Other"
    )
    wrong_label_binding.pop("evidence_digest")
    wrong_label_binding["evidence_digest"] = canonical_json_digest(wrong_label_binding)
    with pytest.raises(ValueError, match="parity record did not pass"):
        build_rig_document(
            stage,
            scene,
            "/World/Rig",
            resolution=(16, 12),
            verification=wrong_label_binding,
        )


def test_rig_rejects_self_digested_parity_semantic_id_swap(tmp_path):
    stage, paths = _rig_stage(cameras=1)
    path_a = UsdGeom.Cube.Define(stage, "/World/A").GetPath().pathString
    path_b = UsdGeom.Cube.Define(stage, "/World/B").GetPath().pathString
    scene = build_scene_analysis_ir(stage)
    shape_id_by_path = {
        path: shape_id for shape_id, path in scene.shape_path_by_id.items()
    }

    class SplitSemanticObservationBackend(_ObservationBackend):
        def observe(self, stage, cameras, width, height, out_dir, **kwargs):
            results = super().observe(stage, cameras, width, height, out_dir, **kwargs)
            split = width // 2
            for result in results:
                semantic_raster = np.ones((height, width, 1), dtype=np.uint32)
                semantic_raster[:, split:, :] = 2
                np.save(
                    result.artifacts["SemanticSegmentation"],
                    semantic_raster,
                    allow_pickle=False,
                )
                rendered_labels = result.aovs["SemanticSegmentation"]["statistics"][
                    "semantic_labels"
                ]
                assert [item["id"] for item in rendered_labels] == [1, 2]
                rendered_labels[0]["pixels"] = height * split
                rendered_labels[1]["pixels"] = height * (width - split)
            return results

    class SplitSemanticAnalyticBackend:
        def render_cameras(self, request):
            split = request.width // 2
            shape_ids = np.full(
                (1, request.height, request.width),
                shape_id_by_path[path_a],
                dtype=np.int64,
            )
            shape_ids[:, :, split:] = shape_id_by_path[path_b]
            return SimpleNamespace(
                depth_m=np.ones((1, request.height, request.width), dtype=np.float32),
                shape_ids=shape_ids,
            )

    evidence, _ = verify_rig_with_ovrtx(
        stage,
        paths,
        backend=SplitSemanticObservationBackend(),
        resolution=(16, 12),
        output_dir=tmp_path / "evidence",
        artifact_base_dir=tmp_path,
        source_digest=scene.source_digest,
        semantic_roles={path_a: "target", path_b: "floor"},
        analytic_backend=SplitSemanticAnalyticBackend(),
        analysis_scene=scene,
    )
    document = build_rig_document(
        stage,
        scene,
        "/World/Rig",
        resolution=(16, 12),
        verification=evidence,
    )
    parity_labels = document["verification"]["newton_ovrtx_parity"][0]["semantics"][
        "labels"
    ]
    parity_labels[0]["semantic_id"], parity_labels[1]["semantic_id"] = (
        parity_labels[1]["semantic_id"],
        parity_labels[0]["semantic_id"],
    )
    verification = document["verification"]
    verification.pop("evidence_digest")
    verification["evidence_digest"] = canonical_json_digest(verification)

    output = tmp_path / "tampered-rig.json"
    with pytest.raises(ValueError, match="parity record did not pass"):
        publish_rig_document(document, output)
    assert not output.exists()


def test_local_ovrtx_render_uses_disposable_aspect_overlay(monkeypatch, tmp_path):
    from usd_core.render.ovrtx import OvRTXRenderBackend

    stage, paths = _rig_stage(cameras=1)
    camera = UsdGeom.Camera(stage.GetPrimAtPath(paths[0]))
    camera.GetHorizontalApertureAttr().Set(40.0)
    camera.GetVerticalApertureAttr().Set(10.0)
    root_before = stage.GetRootLayer().ExportToString()
    backend = OvRTXRenderBackend()

    def fake_worker(command, params):
        assert command == "render"
        assert params["usd_time"] == 0.0
        combined = Path(params["usd_path"]).read_text()
        assert "camera_aspect" in combined
        assert "metersPerUnit = 1" in combined
        assert 'upAxis = "Z"' in combined
        return [{"camera": paths[0], "path": params["out_paths"][0]}]

    monkeypatch.setattr(backend, "_run_worker", fake_worker)
    backend.render(stage, paths, 100, 50, tmp_path, names=["camera"])

    assert camera.GetVerticalApertureAttr().Get() == pytest.approx(10.0)
    assert stage.GetRootLayer().ExportToString() == root_before
    assert not list(tmp_path.glob(".usd-cli_render_*"))
    with pytest.raises(ValueError, match="unsafe OVRTX output"):
        backend.render(stage, paths, 100, 50, tmp_path, names=["../escape"])


def test_local_ovrtx_aspect_overlay_rejects_instance_proxy_camera(tmp_path):
    from usd_core.render.ovrtx import _write_camera_aspect_overlay

    stage = Usd.Stage.CreateInMemory()
    stage.CreateClassPrim("/_CameraPrototype")
    UsdGeom.Camera.Define(stage, "/_CameraPrototype/Camera")
    instance = stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_CameraPrototype")
    instance.SetInstanceable(True)
    camera_path = "/World/Instance/Camera"
    assert stage.GetPrimAtPath(camera_path).IsInstanceProxy()

    with pytest.raises(ValueError, match="instance proxy"):
        _write_camera_aspect_overlay(
            stage,
            [camera_path],
            100,
            50,
            tmp_path / "camera_aspect.usda",
        )
    assert not (tmp_path / "camera_aspect.usda").exists()


def test_local_ovrtx_render_rejects_proxy_camera_without_temp_artifacts(
    monkeypatch, tmp_path
):
    from usd_core.render.ovrtx import OvRTXRenderBackend

    source_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(source_path))
    stage.CreateClassPrim("/_CameraPrototype")
    UsdGeom.Camera.Define(stage, "/_CameraPrototype/Camera")
    instance = stage.DefinePrim("/World/Instance", "Xform")
    instance.GetReferences().AddInternalReference("/_CameraPrototype")
    instance.SetInstanceable(True)
    stage.GetRootLayer().Save()
    camera_path = "/World/Instance/Camera"
    assert stage.GetPrimAtPath(camera_path).IsInstanceProxy()

    backend = OvRTXRenderBackend()
    worker = Mock(side_effect=AssertionError("worker must not run"))
    monkeypatch.setattr(backend, "_run_worker", worker)
    output_dir = tmp_path / "renders"

    with pytest.raises(ValueError, match="instance proxy"):
        backend.render(stage, [camera_path], 100, 50, output_dir)

    worker.assert_not_called()
    assert not list(tmp_path.rglob(".usd-cli_render_*"))


def test_local_ovrtx_observation_distinguishes_default_and_numeric_zero_time(
    monkeypatch, tmp_path
):
    from usd_core.render.ovrtx import (
        CAMERA_OBSERVATION_CAPABILITY,
        WARP_DLPACK_REDUCTION_CAPABILITY,
        OvRTXRenderBackend,
    )

    stage, paths = _rig_stage(cameras=1)
    backend = OvRTXRenderBackend(num_sensor_updates=1)
    captured_times = []

    monkeypatch.setattr(
        backend,
        "runtime_identity",
        lambda: {
            "capabilities": [
                CAMERA_OBSERVATION_CAPABILITY,
                WARP_DLPACK_REDUCTION_CAPABILITY,
            ]
        },
    )

    def fake_worker(command, params):
        assert command == "observe"
        captured_times.append(params["usd_time"])
        return [
            {
                "camera": paths[0],
                "render_product": params["product_paths"][0],
                "aovs": {
                    "DistanceToCameraSD": {
                        "shape": [4, 4, 1],
                        "dtype": "float32",
                        "statistics": {
                            "element_count": 16,
                            "valid_count": 16,
                            "reduction": "warp_cuda_dlpack_v1",
                        },
                    }
                },
                "artifacts": {},
            }
        ]

    monkeypatch.setattr(backend, "_run_worker", fake_worker)
    for index, frame in enumerate((None, 0.0)):
        backend.observe(
            stage,
            paths,
            4,
            4,
            tmp_path / f"observe-{index}",
            mode="quality",
            aovs=("DistanceToCameraSD",),
            artifact_aovs=(),
            names=["camera"],
            frame=frame,
        )

    assert captured_times == [None, 0.0]
