# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite IsaacLab cube-bounce trial implementing the Physics Agent protocol."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

EXPECTED_OBJECTIVE = {
    "name": "bounce_height_error",
    "unit": "m",
    "direction": "minimize",
}
MAX_TRIAL_STEPS = 10_000

PoseSample = tuple[
    float,
    tuple[float, float, float],
    tuple[float, float, float, float],
]


def _validated_trial_steps(value: Any) -> int:
    """Return a finite loop bound for the reference IsaacLab trial."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(
            f"trial.steps must be an integer between 1 and {MAX_TRIAL_STEPS}"
        )
    if value < 1 or value > MAX_TRIAL_STEPS:
        raise ValueError(
            f"trial.steps must be an integer between 1 and {MAX_TRIAL_STEPS}"
        )
    return int(value)


def _load_prevalidated_request(request_path: str) -> tuple[dict[str, Any], int]:
    """Load a request and bound its trial loop before IsaacLab startup."""

    request = json.loads(Path(request_path).read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise ValueError("request must be an object")
    trial = request.get("trial")
    if not isinstance(trial, dict):
        raise ValueError("request trial must be an object")
    return request, _validated_trial_steps(trial.get("steps"))


def _start_frame_capture(
    evidence: dict[str, Any], artifacts_dir: Path, *, purpose: str
) -> tuple[Any, Path]:
    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from isaaclab_physx.renderers import IsaacRtxRendererCfg

    width = int(evidence["width"])
    height = int(evidence["height"])
    camera = Camera(
        cfg=CameraCfg(
            prim_path="/World/EvidenceCamera",
            update_period=0,
            height=height,
            width=width,
            data_types=["rgb"],
            renderer_cfg=IsaacRtxRendererCfg(),
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1000.0),
            ),
        )
    )

    frame_dir = artifacts_dir / f"{purpose}_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    return camera, frame_dir


def _capture_frame(frame_dir: Path, camera: Any, *, index: int) -> Path | None:
    import numpy as np
    from PIL import Image

    data = camera.data.output.get("rgb")
    if data is None:
        return None
    if hasattr(data, "detach"):
        data = data.detach().cpu().numpy()
    frame = np.asarray(data)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim != 3 or frame.shape[2] < 3:
        return None
    rgb = np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8)
    frame_path = frame_dir / f"frame_{index:04d}.png"
    Image.fromarray(rgb, mode="RGB").save(frame_path, format="PNG")
    return frame_path


def _finish_frame_capture(
    evidence: dict[str, Any],
    artifacts_dir: Path,
    frame_paths: list[Path],
    *,
    purpose: str,
) -> Path:
    if len(frame_paths) < int(evidence["min_frames"]):
        raise RuntimeError(
            "rendered evidence has too few frames: "
            f"{len(frame_paths)} < {int(evidence['min_frames'])}"
        )
    fps = float(evidence["fps"])
    manifest_path = artifacts_dir / f"{purpose}_frames.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "physics-agent.qualification-frames.v1",
                "renderer": str(evidence["renderer"]),
                "width": int(evidence["width"]),
                "height": int(evidence["height"]),
                "fps": fps,
                "frames": [
                    {
                        "path": path.relative_to(artifacts_dir).as_posix(),
                        "timestamp_seconds": index / fps,
                    }
                    for index, path in enumerate(frame_paths)
                ],
            },
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return manifest_path


def _camera_transform(position: list[float], target: list[float]) -> Any:
    """Build a Z-up USD camera transform from eye and target points."""

    from pxr import Gf

    eye = Gf.Vec3d(*position)
    look_at = Gf.Vec3d(*target)
    forward = (eye - look_at).GetNormalized()
    up = Gf.Vec3d(0.0, 0.0, 1.0)
    if abs(Gf.Dot(forward, up)) > 0.999:
        up = Gf.Vec3d(0.0, 1.0, 0.0)
    right = Gf.Cross(up, forward).GetNormalized()
    up = Gf.Cross(forward, right).GetNormalized()
    rotation = Gf.Matrix4d(
        right[0],
        right[1],
        right[2],
        0.0,
        up[0],
        up[1],
        up[2],
        0.0,
        forward[0],
        forward[1],
        forward[2],
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return rotation * Gf.Matrix4d().SetTranslate(eye)


def _write_recording(
    recording: dict[str, Any],
    artifacts_dir: Path,
    poses: list[PoseSample],
) -> tuple[Path, int]:
    """Author the exact evaluated rollout as a time-sampled USD."""

    from pxr import Gf, Usd, UsdGeom, UsdLux

    if len(poses) < 2:
        raise RuntimeError("rollout recording requires at least two pose samples")
    recording_path = artifacts_dir / "recording.usd"
    stage = Usd.Stage.CreateNew(str(recording_path))
    if stage is None:
        raise RuntimeError(f"could not create rollout recording: {recording_path}")
    fps = float(recording["fps"])
    stage.SetTimeCodesPerSecond(fps)
    stage.SetFramesPerSecond(fps)
    samples_by_frame: dict[
        int,
        tuple[
            tuple[float, float, float],
            tuple[float, float, float, float],
        ],
    ] = {}
    for sample_time, position, quaternion in poses:
        samples_by_frame[int(round(sample_time * fps))] = (position, quaternion)
    frames = sorted(samples_by_frame)
    stage.SetStartTimeCode(frames[0])
    stage.SetEndTimeCode(frames[-1])
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    UsdGeom.Xform.Define(stage, "/World")
    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.GetSizeAttr().Set(0.2)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.1, 0.45, 0.9)])
    xformable = UsdGeom.Xformable(cube.GetPrim())
    translate = xformable.AddTranslateOp()
    orient = xformable.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    for frame in frames:
        position, quaternion_wxyz = samples_by_frame[frame]
        translate.Set(Gf.Vec3d(*position), frame)
        w, x, y, z = quaternion_wxyz
        orient.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)), frame)

    ground = UsdGeom.Cube.Define(stage, "/World/Ground")
    ground.GetSizeAttr().Set(1.0)
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.18, 0.2, 0.24)])
    ground_xform = UsdGeom.Xformable(ground.GetPrim())
    ground_xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -0.025))
    ground_xform.AddScaleOp().Set(Gf.Vec3f(4.0, 4.0, 0.05))
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2000.0)
    camera = UsdGeom.Camera.Define(stage, "/World/PlaybackCamera")
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    camera_xform = UsdGeom.Xformable(camera.GetPrim()).AddTransformOp()
    camera_xform.Set(
        _camera_transform(
            recording["camera"]["position"], recording["camera"]["target"]
        )
    )
    stage.GetRootLayer().Save()
    return recording_path, len(frames)


def _build_result(
    request: dict[str, Any],
    *,
    artifacts_dir: Path,
    restitution: float,
    trajectory: list[float],
    contacted: bool,
    bounce_height: float,
    captured_frames: int,
    evidence_manifest_path: Path | None,
    recorded_poses: list[PoseSample],
) -> dict[str, Any]:
    """Assemble protocol output and exact-rollout artifacts from simulation data."""

    trajectory_path = artifacts_dir / "trajectory.json"
    trajectory_path.write_text(
        json.dumps(
            {
                "dt_s": float(request["trial"]["dt_s"]),
                "z_m": trajectory,
                "contacted": contacted,
                "bounce_height_m": bounce_height,
            }
        ),
        encoding="utf-8",
    )
    artifacts = {"trajectory": str(trajectory_path)}
    metadata: dict[str, Any] = {"applied_params": {"restitution": restitution}}
    recording = request.get("recording")
    if recording is not None:
        recording_path, recording_frame_count = _write_recording(
            recording, artifacts_dir, recorded_poses
        )
        artifacts[str(recording["artifact_name"])] = str(recording_path)
        metadata["recording"] = {
            "media_type": str(recording["media_type"]),
            "fps": float(recording["fps"]),
            "frame_count": recording_frame_count,
        }
    evidence = request.get("evidence")
    if evidence is not None:
        if evidence_manifest_path is None:
            raise RuntimeError("evidence request did not produce PNG frames")
        artifacts[str(evidence["artifact_name"])] = str(evidence_manifest_path)
        metadata["evidence"] = {
            "renderer": str(evidence["renderer"]),
            "width": int(evidence["width"]),
            "height": int(evidence["height"]),
            "fps": float(evidence["fps"]),
            "frame_count": captured_frames,
        }
    success = contacted and math.isfinite(bounce_height)
    result = {
        "status": "ok",
        "success": success,
        "metrics": {
            "contacted": contacted,
            "bounce_height_m": bounce_height,
        },
        "metadata": metadata,
        "artifacts": artifacts,
    }
    if success:
        result["objective"] = {
            **EXPECTED_OBJECTIVE,
            "value": abs(
                bounce_height - float(request["trial"]["target_bounce_height_m"])
            ),
        }
    return result


def _run_trial(
    request: dict[str, Any],
    runtime_args: argparse.Namespace,
    *,
    validated_steps: int | None = None,
) -> dict[str, Any]:
    params = request["params"]
    trial = request["trial"]
    if not isinstance(params, dict) or not isinstance(trial, dict):
        raise ValueError("request params and trial must be objects")
    objective_contract = request.get("objective")
    if objective_contract != EXPECTED_OBJECTIVE:
        raise ValueError("unexpected objective contract")

    restitution = float(params["restitution"])
    seed = int(request["seed"])
    drop_height = float(trial["drop_height_m"])
    steps = (
        _validated_trial_steps(trial.get("steps"))
        if validated_steps is None
        else validated_steps
    )
    dt = float(trial["dt_s"])
    evidence = request.get("evidence")
    if evidence is not None and not isinstance(evidence, dict):
        raise ValueError("request evidence must be an object")
    recording = request.get("recording")
    if recording is not None and not isinstance(recording, dict):
        raise ValueError("request recording must be an object")

    import isaaclab.sim as sim_utils
    import torch
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from isaaclab.sim import SimulationContext

    torch.manual_seed(seed)

    simulation = SimulationContext(
        sim_utils.SimulationCfg(dt=dt, device=runtime_args.device)
    )
    material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.5,
        dynamic_friction=0.5,
        restitution=restitution,
        friction_combine_mode="average",
        restitution_combine_mode="max",
    )
    ground = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.0,
            restitution_combine_mode="max",
        )
    )
    ground.func("/World/defaultGroundPlane", ground)
    light = sim_utils.DomeLightCfg(intensity=2000.0)
    light.func("/World/Light", light)
    cube_cfg = RigidObjectCfg(
        prim_path="/World/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.45, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, drop_height)),
    )
    cube = RigidObject(cfg=cube_cfg)
    artifacts_dir = Path(str(request["artifacts_dir"]))
    frame_capture = (
        _start_frame_capture(evidence, artifacts_dir, purpose=str(request["purpose"]))
        if evidence is not None
        else None
    )
    simulation.reset()
    if frame_capture is not None:
        camera_config = evidence["camera"]
        frame_capture[0].set_world_poses_from_view(
            torch.tensor([camera_config["position"]], device=runtime_args.device),
            torch.tensor([camera_config["target"]], device=runtime_args.device),
        )

    capture_interval = (
        max(1, int(round(1.0 / (dt * float(evidence["fps"])))))
        if evidence is not None
        else 1
    )
    captured_frames = 0
    captured_frame_paths: list[Path] = []
    evidence_manifest_path: Path | None = None
    recorded_poses: list[PoseSample] = []

    contacted = False
    bounce_height = 0.0
    trajectory: list[float] = []
    try:
        for step in range(steps):
            cube.write_data_to_sim()
            capture_frame = frame_capture is not None and step % capture_interval == 0
            simulation.step(render=capture_frame)
            cube.update(dt)
            z = float(cube.data.root_pos_w[0, 2].item())
            trajectory.append(z)
            if recording is not None:
                position = cube.data.root_pos_w[0].detach().cpu().tolist()
                quaternion = cube.data.root_quat_w[0].detach().cpu().tolist()
                recorded_poses.append(
                    (
                        (step + 1) * dt,
                        (float(position[0]), float(position[1]), float(position[2])),
                        (
                            float(quaternion[0]),
                            float(quaternion[1]),
                            float(quaternion[2]),
                            float(quaternion[3]),
                        ),
                    )
                )
            if z <= 0.115:
                contacted = True
            elif contacted:
                bounce_height = max(bounce_height, z)
            if capture_frame:
                frame_capture[0].update(dt)
                captured = _capture_frame(
                    frame_capture[1], frame_capture[0], index=captured_frames
                )
                if captured is not None:
                    captured_frame_paths.append(captured)
                    captured_frames += 1
    finally:
        if frame_capture is not None:
            evidence_manifest_path = _finish_frame_capture(
                evidence,
                artifacts_dir,
                captured_frame_paths,
                purpose=str(request["purpose"]),
            )

    return _build_result(
        request,
        artifacts_dir=artifacts_dir,
        restitution=restitution,
        trajectory=trajectory,
        contacted=contacted,
        bounce_height=bounce_height,
        captured_frames=captured_frames,
        evidence_manifest_path=evidence_manifest_path,
        recorded_poses=recorded_poses,
    )


def main() -> None:
    preflight_parser = argparse.ArgumentParser(add_help=False)
    preflight_parser.add_argument("--request")
    preflight_args, _ = preflight_parser.parse_known_args()
    preflight = (
        _load_prevalidated_request(preflight_args.request)
        if preflight_args.request is not None
        else None
    )

    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--result", required=True)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    request, validated_steps = (
        preflight if preflight is not None else _load_prevalidated_request(args.request)
    )
    evidence_request = request.get("evidence")
    if evidence_request is not None:
        if not isinstance(evidence_request, dict) or not evidence_request.get(
            "required"
        ):
            raise ValueError("request evidence must be a required object")
        args.enable_cameras = True
    args.headless = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app
    try:
        result = _run_trial(request, args, validated_steps=validated_steps)
        Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
