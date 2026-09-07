# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic VoMP configuration and mass-invariant tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from content_agent_workflows.physics import (
    PhysicsVompMassConfig,
    PhysicsVompMassResult,
    resolve_vomp_target_prim,
    verify_vomp_mass_properties,
)


def _config(tmp_path: Path, *, target: str | None = None) -> PhysicsVompMassConfig:
    return PhysicsVompMassConfig(
        runtime_root=tmp_path / "VoMP",
        target_prim_path=target,
    )


def test_resolve_vomp_target_uses_the_sole_mass_authoring_path(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    target = resolve_vomp_target_prim(
        config,
        [
            {"mass_authoring_path": "/World/Body"},
            {"mass_authoring_path": "/World/Body"},
        ],
    )

    assert target == "/World/Body"


def test_resolve_vomp_target_requires_an_explicit_choice_when_ambiguous(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    with pytest.raises(RuntimeError, match="ambiguous"):
        resolve_vomp_target_prim(
            config,
            [
                {"mass_authoring_path": "/World/BodyA"},
                {"mass_authoring_path": "/World/BodyB"},
            ],
        )

    explicit = _config(tmp_path, target="/World/BodyB")
    assert resolve_vomp_target_prim(explicit, []) == "/World/BodyB"


def test_resolve_vomp_target_rejects_explicit_non_body_path(tmp_path: Path) -> None:
    config = _config(tmp_path, target="/World/StaticFixture")

    with pytest.raises(RuntimeError, match="not an accepted mass-authoring path"):
        resolve_vomp_target_prim(
            config,
            [{"mass_authoring_path": "/World/Body"}],
        )


@pytest.mark.parametrize("seed", [-1, 2**32])
def test_vomp_seed_is_bounded_for_numpy(seed: int, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="seed"):
        PhysicsVompMassConfig(runtime_root=tmp_path / "VoMP", seed=seed)


def test_verify_vomp_mass_properties_detects_post_authoring_mass_drift(
    tmp_path: Path,
) -> None:
    pxr = pytest.importorskip("pxr")
    del pxr
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    provenance_payload = {
        "schemaVersion": 1,
        "source": {
            "npzSha256": "1" * 64,
            "npzSchema": "physics-agent.vomp.v1",
            "usdSha256": "2" * 64,
        },
        "association": {"targetPrimPath": "/World"},
        "rigidBodyMassProperties": {
            "massKg": 10.0,
            "centerOfMassLocalM": [0.02, 0.03, 0.04],
            "diagonalInertiaKgM2": [0.0002, 0.0004, 0.0006],
            "principalAxesWxyz": [1.0, 0.0, 0.0, 0.0],
        },
        "voxelField": {"declaredComplete": True, "sampleCount": 8},
        "evidence": {
            "rendering": {
                "rendererMetadata": {
                    "renderer": "RTX",
                    "samplesPerPixel": 4,
                    "optional": None,
                },
                "viewCount": 150,
            }
        },
    }
    embedded_provenance = json.loads(json.dumps(provenance_payload))
    embedded_rendering = embedded_provenance["evidence"]["rendering"]
    renderer_metadata = embedded_rendering.pop("rendererMetadata")
    embedded_rendering["rendererMetadataJson"] = json.dumps(
        renderer_metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    usd_path = tmp_path / "vomp.usda"
    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 2.0)
    body = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateRigidBodyEnabledAttr(True)
    mass_api = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass_api.CreateMassAttr(5.0)
    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(2.0, 3.0, 4.0))
    mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 2.0, 3.0))
    mass_api.CreatePrincipalAxesAttr(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
    body.GetPrim().SetCustomDataByKey(
        "physicsAgentVomp",
        embedded_provenance,
    )
    stage.GetRootLayer().Save()

    provenance_path = tmp_path / "provenance.json"
    provenance_bytes = json.dumps(provenance_payload, sort_keys=True).encode()
    provenance_path.write_bytes(provenance_bytes)
    result = PhysicsVompMassResult(
        target_prim_path="/World",
        input_usd_path=str(tmp_path / "input.usda"),
        output_usd_path=str(usd_path),
        output_usd_sha256="a" * 64,
        provenance_path=str(provenance_path),
        provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
        evidence_dir=str(tmp_path / "evidence"),
        evidence_manifest_path=str(tmp_path / "evidence" / "manifest.json"),
        vomp_npz_path=str(tmp_path / "vomp.npz"),
        worker_manifest_path=str(tmp_path / "worker.json"),
        worker_log_path=str(tmp_path / "worker.log"),
        sample_count=8,
        mass_kg=10.0,
        center_of_mass_local_m=(0.02, 0.03, 0.04),
        diagonal_inertia_kg_m2=(0.0002, 0.0004, 0.0006),
        principal_axes_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    verification = verify_vomp_mass_properties(usd_path, result)
    assert verification["verified"] is True
    assert verification["mass_kg"] == pytest.approx(10.0)

    missing_sample_payload = json.loads(json.dumps(provenance_payload))
    missing_sample_payload.pop("voxelField")
    missing_sample_bytes = json.dumps(missing_sample_payload, sort_keys=True).encode()
    provenance_path.write_bytes(missing_sample_bytes)
    missing_sample_result = result.model_copy(
        update={
            "provenance_sha256": hashlib.sha256(missing_sample_bytes).hexdigest(),
        }
    )
    with pytest.raises(RuntimeError, match="sample count changed"):
        verify_vomp_mass_properties(usd_path, missing_sample_result)
    provenance_path.write_bytes(provenance_bytes)

    provenance_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="external provenance digest changed"):
        verify_vomp_mass_properties(usd_path, result)
    provenance_path.write_bytes(provenance_bytes)

    reopened = Usd.Stage.Open(str(usd_path))
    reopened_target = reopened.GetPrimAtPath("/World")
    tampered_provenance = reopened_target.GetCustomDataByKey("physicsAgentVomp")
    tampered_provenance["source"]["usdSha256"] = "3" * 64
    reopened_target.SetCustomDataByKey("physicsAgentVomp", tampered_provenance)
    reopened.GetRootLayer().Save()

    with pytest.raises(RuntimeError, match="complete external provenance"):
        verify_vomp_mass_properties(usd_path, result)

    reopened_target.SetCustomDataByKey("physicsAgentVomp", embedded_provenance)
    UsdPhysics.MassAPI(reopened_target).GetMassAttr().Set(6.0)
    reopened.GetRootLayer().Save()

    with pytest.raises(RuntimeError, match="mass changed"):
        verify_vomp_mass_properties(usd_path, result)
