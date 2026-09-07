#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render registered OVRTX color, normal, and depth evidence through usd-cli."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
from content_agent_workflows.common import usd_cli as usd_cli_backend
from content_agent_workflows.common.usd_cli import (
    UsdCliPackageRoute,
    controlled_usd_cli_telemetry_env,
    run_bounded_usd_cli_subprocess,
)
from content_agent_workflows.common.usd_cli_session import (
    validated_ovrtx_render_metadata,
)
from PIL import Image
from pxr import Gf, Usd, UsdGeom
from usd_core.camera import fit_distance, look_at_matrix, orbit_position

from world_understanding.utils.artifacts import (
    append_bytes_to_confined,
    open_confined_directory,
    open_regular_file_no_follow,
    write_bytes_to_confined,
)

DEFAULT_RENDER_MODE = "quality"
DEFAULT_Z_UP_CAMERAS = (
    "+x-y+z",
    "-x-y+z",
    "+x+y+z",
    "-x+y+z",
    "+x-y-z",
    "-x-y-z",
    "+x+y-z",
    "-x+y-z",
)
DEFAULT_Y_UP_CAMERAS = (
    "+x+y-z",
    "-x+y-z",
    "+x+y+z",
    "-x+y+z",
    "+x-y-z",
    "-x-y-z",
    "+x-y+z",
    "-x-y+z",
)

_BINARY_OPEN_FLAG = getattr(os, "O_BINARY", 0)
_NO_INHERIT_OPEN_FLAG = getattr(
    os,
    "O_CLOEXEC",
    getattr(os, "O_NOINHERIT", 0),
)


def _is_private_regular_file(metadata: os.stat_result) -> bool:
    """Validate portable invariants plus POSIX ownership and mode bits."""

    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        return False
    if os.name != "posix":
        # Windows privacy is ACL-based. The confined artifact backend rejects
        # reparse points and pins the file identity; the run root owns its ACL.
        return True
    return metadata.st_uid == os.geteuid() and stat.S_IMODE(metadata.st_mode) == 0o600


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--focus", default="/World/FusedMesh")
    parser.add_argument("--camera", action="append", dest="cameras")
    parser.add_argument(
        "--camera-json",
        action="append",
        type=Path,
        help="Repeatable saved workflow camera JSON; cannot mix with --camera.",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_name(value: str) -> str:
    name = value.replace("+", "plus_").replace("-", "minus_")
    return "".join(
        character if character.isalnum() or character == "_" else "_"
        for character in name
    ).strip("_")


def _open_stage(scene: Path) -> tuple[Usd.Stage, bool]:
    # Evidence framing must see the same composed asset that usd-cli renders.
    # The staged input may carry the focus mesh behind a payload, especially
    # when --asset-root preserves a complete asset tree.
    stage = Usd.Stage.Open(str(scene), load=Usd.Stage.LoadAll)
    if stage is None:
        raise ValueError(f"Could not open USD: {scene}")
    return stage, UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y


def _default_cameras(stage: Usd.Stage) -> tuple[str, ...]:
    return (
        DEFAULT_Y_UP_CAMERAS
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
        else DEFAULT_Z_UP_CAMERAS
    )


def _project_dir(output_dir: Path) -> Path:
    for candidate in (output_dir, *output_dir.parents):
        if (candidate / "request.json").is_file():
            return candidate
    return Path.cwd().resolve()


def _session_id(project_dir: Path, scene: Path) -> str:
    request_path = project_dir / "request.json"
    if request_path.is_file():
        request = json.loads(request_path.read_text(encoding="utf-8"))
        runtime = request.get("runtime")
        if isinstance(runtime, dict):
            value = runtime.get("usd_cli_session_id")
            if isinstance(value, str) and value:
                return value
    digest = hashlib.sha256(str(scene).encode("utf-8")).hexdigest()[:20]
    return f"workflow-{digest}"


def _package_owned_route() -> UsdCliPackageRoute:
    # This helper is staged into an isolated run directory, so its own
    # ``__file__`` is not below the repository that owns usd-cli. Resolve from
    # the installed workflow package instead; that package is the authority
    # already supplying the attestation and subprocess helpers used here.
    module_path = Path(usd_cli_backend.__file__).resolve()
    repository_root = usd_cli_backend.find_usd_cli_repository_root(module_path)
    return usd_cli_backend.resolve_package_owned_usd_cli_route(repository_root)


@dataclass(slots=True)
class _MeshEvidenceUsdCliSession:
    """Authenticated client for the launcher's live session with sibling receipts.

    The launcher-owned ``WorkflowUsdCliSession`` keeps its receipt journal pinned to
    one process.  This helper therefore joins that named live sidecar but writes a
    separately named, identity-pinned journal so it cannot invalidate the launcher's
    checkpoint when running in the isolated child process.
    """

    project_dir: Path
    session_id: str
    route: UsdCliPackageRoute
    telemetry_file: Path
    receipt_dir: Path
    _journal_identity: tuple[int, int] = field(init=False, repr=False)
    _journal_digest: str = field(init=False, repr=False)
    _journal_size: int = field(init=False, repr=False)
    _checkpoint_digest: str = field(init=False, repr=False)

    @classmethod
    def create(
        cls,
        *,
        project_dir: Path,
        session_id: str,
        evidence_dir: Path,
    ) -> _MeshEvidenceUsdCliSession:
        receipt_dir = evidence_dir / ".usd_cli_receipts"
        receipt_dir.mkdir(mode=0o700)
        journal_path = receipt_dir / "mesh_evidence_commands.jsonl"
        descriptor = os.open(
            journal_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | _NO_INHERIT_OPEN_FLAG
            | _BINARY_OPEN_FLAG,
            0o600,
        )
        try:
            journal_stat = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        session = cls(
            project_dir=project_dir.resolve(strict=True),
            session_id=session_id,
            route=_package_owned_route(),
            telemetry_file=receipt_dir / "usd_cli_telemetry.jsonl",
            receipt_dir=receipt_dir,
        )
        session._journal_identity = (journal_stat.st_dev, journal_stat.st_ino)
        session._journal_digest = hashlib.sha256(b"").hexdigest()
        session._journal_size = 0
        session._write_checkpoint()
        return session

    @property
    def journal_path(self) -> Path:
        return self.receipt_dir / "mesh_evidence_commands.jsonl"

    @property
    def checkpoint_path(self) -> Path:
        return self.receipt_dir / "mesh_evidence_commands.checkpoint.json"

    def _verified_journal(self) -> bytes:
        with open_regular_file_no_follow(self.journal_path) as (
            stream,
            journal_stat,
        ):
            journal = stream.read()
        if (
            not _is_private_regular_file(journal_stat)
            or (journal_stat.st_dev, journal_stat.st_ino) != self._journal_identity
            or len(journal) != self._journal_size
            or hashlib.sha256(journal).hexdigest() != self._journal_digest
        ):
            raise RuntimeError("mesh-evidence usd-cli receipt journal changed")
        return journal

    def _write_checkpoint(self) -> None:
        checkpoint = {
            "schema_version": (
                "content-agent-workflows.mesh-evidence-usd-cli-checkpoint.v1"
            ),
            "workflow": "mesh-segmentation",
            "session_id": self.session_id,
            "receipt_device": self._journal_identity[0],
            "receipt_inode": self._journal_identity[1],
            "receipt_sha256": self._journal_digest,
            "receipt_size_bytes": self._journal_size,
            "usd_cli_source_revision": self.route.source_revision,
        }
        encoded = (json.dumps(checkpoint, sort_keys=True) + "\n").encode()
        with open_confined_directory(self.receipt_dir) as receipt_root:
            if not write_bytes_to_confined(
                receipt_root,
                self.checkpoint_path.name,
                encoded,
                overwrite=True,
                file_mode=0o600,
            ):  # pragma: no cover - overwrite always publishes
                raise RuntimeError("could not seal the usd-cli receipt checkpoint")
        self._checkpoint_digest = hashlib.sha256(encoded).hexdigest()

    def _verify_checkpoint(self) -> None:
        with open_regular_file_no_follow(self.checkpoint_path) as (
            stream,
            checkpoint_stat,
        ):
            encoded = stream.read()
        try:
            checkpoint = json.loads(encoded)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("mesh-evidence usd-cli checkpoint is invalid") from exc
        if (
            not _is_private_regular_file(checkpoint_stat)
            or hashlib.sha256(encoded).hexdigest() != self._checkpoint_digest
            or not isinstance(checkpoint, dict)
            or checkpoint.get("session_id") != self.session_id
            or checkpoint.get("receipt_device") != self._journal_identity[0]
            or checkpoint.get("receipt_inode") != self._journal_identity[1]
            or checkpoint.get("receipt_sha256") != self._journal_digest
            or checkpoint.get("receipt_size_bytes") != self._journal_size
        ):
            raise RuntimeError("mesh-evidence usd-cli checkpoint changed")

    def _append_receipt(self, receipt: dict[str, Any]) -> None:
        previous = self._verified_journal()
        encoded = (json.dumps(receipt, sort_keys=True) + "\n").encode()
        with open_confined_directory(self.receipt_dir) as receipt_root:
            append_bytes_to_confined(
                receipt_root,
                self.journal_path.name,
                encoded,
                file_mode=0o600,
            )
        combined = previous + encoded
        self._journal_size = len(combined)
        self._journal_digest = hashlib.sha256(combined).hexdigest()
        self._verified_journal()
        self._write_checkpoint()

    def _render_artifact_bindings(
        self, payload: dict[str, Any]
    ) -> list[dict[str, Any]]:
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise RuntimeError("usd-cli render response omitted artifact bindings")
        bindings: list[dict[str, Any]] = []
        for index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                raise RuntimeError(f"usd-cli render artifact {index} is invalid")
            label = artifact.get("label")
            raw_path = artifact.get("path")
            if not isinstance(label, str) or not label or not isinstance(raw_path, str):
                raise RuntimeError(f"usd-cli render artifact {index} is incomplete")
            path = Path(raw_path).expanduser()
            if path.is_symlink():
                raise RuntimeError(f"usd-cli render artifact is a symlink: {path}")
            path = path.resolve(strict=True)
            try:
                path.relative_to(self.project_dir)
            except ValueError as exc:
                raise RuntimeError(
                    f"usd-cli render artifact escaped the workflow project: {path}"
                ) from exc
            metadata = path.stat(follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise RuntimeError(
                    f"usd-cli render artifact is not a unique regular file: {path}"
                )
            bindings.append(
                {
                    "label": label,
                    "path": str(path),
                    "sha256": _digest(path),
                    "size_bytes": metadata.st_size,
                }
            )
        return bindings

    def run_json(
        self,
        arguments: list[str],
        *,
        timeout_seconds: float = 1800.0,
    ) -> dict[str, Any]:
        self._verified_journal()
        self._verify_checkpoint()
        environment = controlled_usd_cli_telemetry_env(
            route=self.route,
            telemetry_file=self.telemetry_file,
            attrs=(f"wu.workflow=mesh-segmentation,wu.session={self.session_id}"),
        )
        environment["USD_CLI_LOCK_RENDER_CONFIG"] = "1"
        if environment.get("CONTENT_WORKFLOW_PARENT_USD_CLI_MANAGED") == "1":
            environment["USD_CLI_NO_DAEMON"] = "1"
            environment["USD_CLI_LOCAL_GPU_FORBIDDEN"] = "1"
        command = [
            str(self.route.wrapper),
            "--json",
            "--session",
            self.session_id,
            *arguments,
        ]
        started_ns = time.time_ns()
        try:
            completed = run_bounded_usd_cli_subprocess(
                command,
                cwd=self.project_dir,
                env=environment,
                timeout=timeout_seconds,
                check=False,
            )
            payload = json.loads(completed.stdout)
        except Exception as exc:
            self._append_receipt(
                {
                    "schema_version": (
                        "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
                    ),
                    "workflow": "mesh-segmentation",
                    "session_id": self.session_id,
                    "started_unix_nano": started_ns,
                    "completed_unix_nano": time.time_ns(),
                    "arguments": arguments,
                    "tool": {
                        "name": "usd-cli",
                        "source_revision": self.route.source_revision,
                    },
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            raise RuntimeError(
                f"usd-cli command could not complete for {arguments!r}: {exc}"
            ) from exc
        is_probe = (
            isinstance(payload, dict)
            and payload.get("schema_version") == "usd-cli.render-probe.v1"
            and payload.get("ready") is True
        )
        succeeded = (
            completed.returncode == 0
            and isinstance(payload, dict)
            and (payload.get("ok") is True or is_probe)
        )
        artifact_bindings: list[dict[str, Any]] = []
        binding_error: Exception | None = None
        if succeeded and arguments[0] == "render":
            try:
                artifact_bindings = self._render_artifact_bindings(payload)
            except Exception as exc:
                binding_error = exc
                succeeded = False
        self._append_receipt(
            {
                "schema_version": (
                    "content-agent-workflows.mesh-evidence-usd-cli-receipt.v1"
                ),
                "workflow": "mesh-segmentation",
                "session_id": self.session_id,
                "started_unix_nano": started_ns,
                "completed_unix_nano": time.time_ns(),
                "arguments": arguments,
                "tool": {
                    "name": "usd-cli",
                    "source_revision": self.route.source_revision,
                },
                "returncode": completed.returncode,
                "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
                "stderr": completed.stderr,
                "response": payload if isinstance(payload, dict) else None,
                "artifact_bindings": artifact_bindings,
                "status": "completed" if succeeded else "failed",
                **(
                    {
                        "error_type": type(binding_error).__name__,
                        "error": str(binding_error),
                    }
                    if binding_error is not None
                    else {}
                ),
            }
        )
        if not succeeded:
            detail = binding_error or completed.stderr.strip() or payload
            raise RuntimeError(f"usd-cli command failed for {arguments!r}: {detail}")
        return cast(dict[str, Any], payload)


def _focus_geometry(
    stage: Usd.Stage,
    focus: str,
    *,
    up_axis_y: bool,
) -> tuple[list[float], float]:
    prim = stage.GetPrimAtPath(focus)
    if not prim.IsValid():
        raise ValueError(f"Focus prim does not exist: {focus}")
    bbox = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    ).ComputeWorldBound(prim)
    aligned = bbox.ComputeAlignedBox()
    minimum = aligned.GetMin()
    maximum = aligned.GetMax()
    center = [float((minimum[index] + maximum[index]) * 0.5) for index in range(3)]
    size = [float(maximum[index] - minimum[index]) for index in range(3)]
    return center, fit_distance(size)


def _direction_angles(direction: str, *, up_axis_y: bool) -> tuple[float, float]:
    values: dict[str, float] = {}
    index = 0
    while index < len(direction):
        sign = direction[index]
        if sign not in "+-" or index + 1 >= len(direction):
            raise ValueError(f"Invalid cube-corner direction: {direction}")
        axis = direction[index + 1]
        if axis not in "xyz" or axis in values:
            raise ValueError(f"Invalid cube-corner direction: {direction}")
        values[axis] = 1.0 if sign == "+" else -1.0
        index += 2
    if not values:
        raise ValueError(f"Invalid camera direction: {direction}")
    for axis in "xyz":
        values.setdefault(axis, 0.0)
    length = math.sqrt(sum(value * value for value in values.values()))
    if up_axis_y:
        azimuth = math.degrees(math.atan2(values["x"], values["z"]))
        elevation = math.degrees(math.asin(values["y"] / length))
    else:
        azimuth = math.degrees(math.atan2(values["y"], values["x"]))
        elevation = math.degrees(math.asin(values["z"] / length))
    return azimuth, elevation


def _matrix_rows(matrix: Gf.Matrix4d) -> list[list[float]]:
    return [[float(matrix[row][column]) for column in range(4)] for row in range(4)]


def _camera_payload(
    *,
    target: list[float],
    position: tuple[float, float, float],
    distance: float,
    azimuth: float,
    elevation: float,
    focus: str | None,
    width: int,
    height: int,
    up_axis_y: bool,
    direction: str | None,
) -> dict[str, Any]:
    up = Gf.Vec3d(0, 1, 0) if up_axis_y else Gf.Vec3d(0, 0, 1)
    return {
        "camera_path": "/World/usd_cam",
        "camera_state": {
            "target": target,
            "distance": distance,
            "yaw_degrees": azimuth,
            "pitch_degrees": elevation,
            "focal_length": 60.0,
            "horizontal_aperture": 36.0,
            "last_framed_prim_path": focus,
        },
        "camera_world_transform": _matrix_rows(look_at_matrix(position, target, up)),
        "image_width": width,
        "image_height": height,
        "direction": direction,
        "renderer": "ovrtx",
    }


def _artifact(payload: dict[str, Any], label: str) -> Path:
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("usd-cli render response omitted artifacts")
    for artifact in artifacts:
        actual = artifact.get("label") if isinstance(artifact, dict) else None
        if isinstance(actual, str) and (
            actual == label or (label == "rgb" and actual.startswith("rgb:"))
        ):
            return Path(str(artifact["path"])).resolve(strict=True)
    raise RuntimeError(f"usd-cli render response omitted {label!r}")


def _save_channel(
    source: Path,
    output_dir: Path,
    stem: str,
    name: str,
) -> dict[str, Any]:
    with Image.open(source) as image:
        values = np.asarray(image)
    raw_path = output_dir / f"{stem}_{name}.npy"
    np.save(raw_path, values, allow_pickle=False)
    return {
        "aov": name,
        "evidence_role": "auxiliary_cpu_aov",
        "final_render_evidence": False,
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "raw": str(raw_path),
        "raw_sha256": _digest(raw_path),
        "preview": str(source),
        "preview_sha256": _digest(source),
    }


def _save_metric_depth(
    source: Path,
    preview_source: Path,
    *,
    unit: object,
) -> dict[str, Any]:
    """Preserve a genuine metric depth tensor and its separate display preview."""

    if source.suffix.lower() != ".npy":
        raise RuntimeError(
            "usd-cli did not provide raw metric linear depth; refusing to relabel "
            f"the {source.suffix or 'unknown'} artifact as linear_depth.npy"
        )
    if unit != "meter":
        raise RuntimeError(
            "usd-cli metric linear-depth artifact omitted its meter unit declaration"
        )
    try:
        values = np.load(source, allow_pickle=False, mmap_mode="r")
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            "usd-cli linear-depth artifact is not a valid NPY array"
        ) from exc
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.floating):
        raise RuntimeError(
            "usd-cli linear-depth artifact must be a two-dimensional floating array"
        )
    finite = np.isfinite(values)
    if not finite.any() or np.any(values[finite] <= 0):
        raise RuntimeError(
            "usd-cli linear-depth artifact contains no valid positive metric depths"
        )
    with Image.open(preview_source) as preview:
        if preview.size != (values.shape[1], values.shape[0]):
            raise RuntimeError(
                "usd-cli depth preview dimensions do not match metric linear depth"
            )

    return {
        "aov": "linear_depth",
        "evidence_role": "auxiliary_cpu_aov",
        "final_render_evidence": False,
        "encoding": "camera_space_linear_depth",
        "unit": "meter",
        "shape": list(values.shape),
        "dtype": str(values.dtype),
        "raw": str(source),
        "raw_sha256": _digest(source),
        "preview": str(preview_source),
        "preview_sha256": _digest(preview_source),
        "preview_encoding": "per_frame_normalized_uint8",
    }


def _from_saved_camera(
    payload: dict[str, Any],
) -> tuple[list[float], tuple[float, float, float]]:
    state = payload.get("camera_state")
    transform = payload.get("camera_world_transform")
    if not isinstance(state, dict) or not isinstance(transform, list):
        raise ValueError("Saved camera JSON is missing camera state or transform")
    target = [float(value) for value in state["target"]]
    raw_position = tuple(float(value) for value in transform[3][:3])
    if len(raw_position) != 3:
        raise ValueError("Saved camera JSON has an invalid camera position")
    position = (raw_position[0], raw_position[1], raw_position[2])
    return target, position


def main() -> None:
    args = parse_args()
    if args.cameras and args.camera_json:
        raise ValueError("Use --camera or --camera-json, not both")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("Render dimensions must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scene = args.scene.resolve(strict=True)
    stage, up_axis_y = _open_stage(scene)
    project_dir = _project_dir(output_dir)
    session_id = os.environ.get("CONTENT_WORKFLOW_USD_CLI_SESSION_ID") or _session_id(
        project_dir, scene
    )
    usd_cli_session = _MeshEvidenceUsdCliSession.create(
        project_dir=project_dir,
        session_id=session_id,
        evidence_dir=output_dir,
    )
    # Camera creation is an in-memory stage edit.  A prior evidence invocation
    # may therefore leave this workflow-owned session dirty even though the USD
    # on disk is immutable.  Evidence rendering always starts from the exact
    # on-disk scene, so explicitly discard only that transient camera state.
    usd_cli_session.run_json(["open", str(scene), "--force-reload"])
    usd_cli_session.run_json(
        [
            "render-probe",
            "--require-engine",
            "ovrtx",
            "--output-dir",
            str(output_dir / "probe"),
        ],
    )
    center, distance = _focus_geometry(stage, args.focus, up_axis_y=up_axis_y)
    if args.camera_json:
        specs: list[dict[str, Any]] = [
            {
                "name": path.stem.removesuffix("_camera"),
                "camera": path.resolve(strict=True),
            }
            for path in args.camera_json
        ]
    else:
        specs = [
            {"name": _safe_name(direction), "direction": direction}
            for direction in (args.cameras or _default_cameras(stage))
        ]

    records: list[dict[str, Any]] = []
    for spec in specs:
        direction = spec.get("direction")
        source_camera = spec.get("camera")
        if isinstance(source_camera, Path):
            camera_payload = json.loads(source_camera.read_text(encoding="utf-8"))
            target, position = _from_saved_camera(camera_payload)
            saved_state = camera_payload["camera_state"]
            usd_cli_session.run_json(
                [
                    "camera",
                    "create",
                    "--name",
                    "mesh_evidence",
                    "--at",
                    ",".join(str(value) for value in position),
                    "--look-at",
                    ",".join(str(value) for value in target),
                    "--focal",
                    str(saved_state.get("focal_length", 60.0)),
                    "--aperture",
                    str(saved_state.get("horizontal_aperture", 36.0)),
                ],
            )
            camera_payload["image_width"] = args.width
            camera_payload["image_height"] = args.height
            camera_payload["renderer"] = "ovrtx"
            camera_payload["camera_path"] = "/World/mesh_evidence"
        else:
            azimuth, elevation = _direction_angles(str(direction), up_axis_y=up_axis_y)
            usd_cli_session.run_json(
                [
                    "camera",
                    "orbit",
                    args.focus,
                    "--az",
                    str(azimuth),
                    "--el",
                    str(elevation),
                    "--dist",
                    str(distance),
                ],
            )
            position = orbit_position(
                center,
                distance,
                azimuth,
                elevation,
                up_axis_y=up_axis_y,
            )
            camera_payload = _camera_payload(
                target=center,
                position=position,
                distance=distance,
                azimuth=azimuth,
                elevation=elevation,
                focus=args.focus,
                width=args.width,
                height=args.height,
                up_axis_y=up_axis_y,
                direction=str(direction),
            )

        stem = str(spec["name"])
        render_dir = output_dir / f".usd_cli_{stem}"
        started = time.perf_counter()
        response = usd_cli_session.run_json(
            [
                "render",
                "--photoreal",
                "--depth",
                "--normals",
                "--mode",
                DEFAULT_RENDER_MODE,
                "--res",
                f"{args.width}x{args.height}",
                "--output",
                str(render_dir),
            ],
        )
        image_path = _artifact(response, "rgb")
        render_metadata = validated_ovrtx_render_metadata(response)
        camera_path = output_dir / f"{stem}_camera.json"
        response_path = output_dir / f"{stem}_response.json"
        camera_path.write_text(
            json.dumps(camera_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        response_path.write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        records.append(
            {
                "name": stem,
                "direction": direction,
                "source_camera": str(source_camera) if source_camera else None,
                "image": str(image_path),
                "image_sha256": _digest(image_path),
                "camera": str(camera_path),
                "camera_sha256": _digest(camera_path),
                "response": str(response_path),
                "response_sha256": _digest(response_path),
                "request_seconds": time.perf_counter() - started,
                "renderer": render_metadata["backend"],
                "ovrtx_render_mode": render_metadata["ovrtx_render_mode"],
                "ovrtx_num_sensor_updates": render_metadata["ovrtx_num_sensor_updates"],
                "channels": {
                    "normal": _save_channel(
                        _artifact(response, "normals"),
                        output_dir,
                        stem,
                        "normal",
                    ),
                    "linear_depth": _save_metric_depth(
                        _artifact(response, "linear_depth"),
                        _artifact(response, "depth"),
                        unit=(
                            response["data"].get("linear_depth_unit")
                            if isinstance(response.get("data"), dict)
                            else None
                        ),
                    ),
                },
            }
        )
        print(f"rendered {stem}", flush=True)

    manifest = {
        "schema_version": "mesh-segmentation-render-evidence.v2",
        "scene": str(scene),
        "scene_sha256": _digest(scene),
        "focus": args.focus,
        "width": args.width,
        "height": args.height,
        "renderer": "ovrtx",
        "render_mode": DEFAULT_RENDER_MODE,
        "scene_tool": "usd-cli",
        "usd_cli_session_id": usd_cli_session.session_id,
        "final_render_channels": ["rgb"],
        "auxiliary_cpu_aov_channels": ["normal", "linear_depth"],
        "usd_cli_command_receipts": str(usd_cli_session.journal_path),
        "usd_cli_command_receipts_sha256": _digest(usd_cli_session.journal_path),
        "usd_cli_receipt_checkpoint": str(usd_cli_session.checkpoint_path),
        "usd_cli_receipt_checkpoint_sha256": _digest(usd_cli_session.checkpoint_path),
        "renders": records,
    }
    manifest_path = output_dir / "render_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
