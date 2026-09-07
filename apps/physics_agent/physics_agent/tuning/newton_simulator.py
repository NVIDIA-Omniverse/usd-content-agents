# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""In-process Newton simulator adapter for the tuning runner.

Newton (https://github.com/newton-physics/newton) is NVIDIA's open-source,
GPU-accelerated, differentiable physics engine built on Warp + MuJoCo-warp
kernels. Unlike OvPhysX (which we daemon-isolate because its bundled OpenUSD
collides with the parent's OpenUSD provider), Newton is installed through the
``apps/physics_agent[newton]`` extra and is driven from the parent venv
directly.

This module wraps Newton in a :class:`physics_agent.tuning.simulator.Simulator`
implementation. The scenario evaluators (``drop_settle`` and ``freeform``) call
the same ``simulator.evaluate(...)`` against either OvPhysX or Newton; only the
engine name changes.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from physics_agent.tuning.errors import NewtonUnavailableError

logger = logging.getLogger(__name__)


_DEFAULT_DEVICE_ENV = "PA_NEWTON_DEVICE"
_DEFAULT_MUJOCO_NJMAX = 128


class NewtonSimulator:
    """In-process Newton simulator implementing the :class:`Simulator` protocol.

    The simulator is constructed lazily on first ``evaluate`` call (Newton's
    Warp kernel compilation is heavy; the parent process shouldn't pay that
    cost until a Newton trial is actually requested).

    Device selection comes from the ``PA_NEWTON_DEVICE`` env var (default
    ``"cuda"``). Set to ``"cpu"`` on hosts without CUDA-capable GPUs.
    """

    name = "newton"

    def __init__(self, *, device: str | None = None) -> None:
        self._device = device or os.environ.get(_DEFAULT_DEVICE_ENV, "cuda")

    def _use_mujoco_cpu(self) -> bool:
        device = self._device.strip().lower()
        return device == "cpu" or device.startswith("cpu:")

    @staticmethod
    def _import_newton() -> Any:
        try:
            import newton
        except ImportError as exc:
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                f"(underlying import error: {exc})"
            ) from exc

        missing: list[str] = []
        if not hasattr(newton, "ModelBuilder"):
            missing.append("newton.ModelBuilder")
        solvers = getattr(newton, "solvers", None)
        if solvers is None or not hasattr(solvers, "SolverMuJoCo"):
            missing.append("newton.solvers.SolverMuJoCo")
        if missing:
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                "The installed newton package is missing required simulator "
                f"APIs: {', '.join(missing)}. Reinstall the physics_agent "
                "Newton extra to pick up newton[sim,importers]."
            )
        return newton

    def warmup(self) -> None:
        """Eagerly probe that Newton+Warp+CUDA are importable and functional.

        Called by :class:`NewtonBackend` before any LLM call so a missing
        ``[newton]`` extra or missing CUDA fails fast with an actionable
        message rather than after a paid scenario-author LLM call has been
        spent.
        """
        newton = self._import_newton()
        try:
            import warp as wp
        except ImportError as exc:
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                f"(underlying import error: {exc})"
            ) from exc
        # Build a 1-body Model+Solver to flush kernel compile + device init
        # without simulating anything meaningful.
        builder = newton.ModelBuilder()
        if not callable(getattr(builder, "add_usd", None)):
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                "The installed newton package is missing ModelBuilder.add_usd. "
                "Reinstall the physics_agent Newton extra to pick up "
                "newton[sim,importers]."
            )
        builder.add_body(xform=wp.transform_identity())
        builder.add_shape_box(body=0, hx=0.01, hy=0.01, hz=0.01)
        builder.add_shape_plane()
        try:
            model = builder.finalize(device=self._device)
            _ = newton.solvers.SolverMuJoCo(
                model,
                njmax=_DEFAULT_MUJOCO_NJMAX,
                use_mujoco_cpu=self._use_mujoco_cpu(),
            )
        except Exception as exc:
            raise NewtonUnavailableError(
                "Newton warmup failed. Set PA_NEWTON_DEVICE=cpu if no GPU is "
                f"available, or check the Newton install. Underlying error: {exc}"
            ) from exc

    def shutdown(self) -> None:
        """No-op; Newton keeps Warp kernels cached for the process lifetime."""

    def evaluate(
        self,
        *,
        scene_usd: Path,
        body_pattern: str,
        duration_s: float,
        dt: float = 1.0 / 240.0,
        sample_fps: int = 30,
        initial_linear_velocity: Sequence[float] | None = None,
        initial_angular_velocity: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Run one tune trial. See :class:`Simulator` for return contract."""
        self._validate_timing(duration_s=duration_s, dt=dt, sample_fps=sample_fps)

        newton = self._import_newton()

        # 1. Load scene USD into a ModelBuilder.
        # ``ModelBuilder.add_usd`` parses UsdPhysics schemas (Scene, RigidBody,
        # Collision, Mass, MaterialAPI) and returns prim-path → body/shape maps.
        builder = newton.ModelBuilder()
        add_usd = getattr(builder, "add_usd", None)
        if not callable(add_usd):
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                "The installed newton package is missing ModelBuilder.add_usd. "
                "Reinstall the physics_agent Newton extra to pick up "
                "newton[sim,importers]."
            )
        try:
            results = add_usd(
                str(scene_usd),
                collapse_fixed_joints=True,
                apply_up_axis_from_stage=True,
                # Import collision shapes only. Visual meshes can share prim
                # paths with colliders and shadow collision material settings
                # in Newton's path_shape_map on SimReady-style assets.
                load_visual_shapes=False,
            )
        except TypeError as exc:
            message = str(exc)
            if not (
                (
                    "apply_up_axis_from_stage" in message
                    or "load_visual_shapes" in message
                )
                and "unexpected keyword" in message
            ):
                raise
            raise NewtonUnavailableError(
                f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n"
                "NewtonSimulator requires newton.ModelBuilder.add_usd support "
                "for apply_up_axis_from_stage and load_visual_shapes. "
                "Reinstall the physics_agent Newton extra to pick up the "
                "newton[sim,importers] install."
            ) from exc
        if results is None:
            raise RuntimeError(
                f"NewtonSimulator: ModelBuilder.add_usd returned None for "
                f"{scene_usd!s}."
            )
        path_body_map: dict[str, int] = dict(results.get("path_body_map", {}))
        if not path_body_map:
            raise RuntimeError(
                f"NewtonSimulator: no rigid bodies in {scene_usd!s}. "
                "ModelBuilder.add_usd returned an empty path_body_map."
            )

        # The drop_settle / freeform scene builders author the ground as a
        # planar ``UsdGeom.Mesh`` (``/SceneRoot/GroundPlane``) because that's
        # what OvPhysX expects. Newton's collision pipeline does not
        # reliably generate contacts against a planar mesh — bodies pass
        # through. Inject a primitive ground plane on the world body so
        # contacts are deterministic. ``apply_up_axis_from_stage=True`` above
        # sets ``builder.up_axis`` from the USD stage, and ``add_ground_plane``
        # derives the plane normal from ``builder.up_vector``. When
        # possible, clone the authored USD ground contact fields onto the
        # primitive plane. Newton MuJoCo bounce is driven by the contact
        # stiffness/damping arrays rather than UsdPhysics restitution.
        ground_cfg = self._ground_plane_shape_config(builder, results)
        self._add_ground_plane(builder, cfg=ground_cfg)

        # 2. Resolve body_pattern to a body index.
        body_idx = self._resolve_body_index(body_pattern, path_body_map)

        # 3. Finalize model + build solver. The scene builder authors a
        # planar mesh ground (``GroundPlane``) that the MuJoCo contact
        # generator rejects, but Newton's native ``CollisionPipeline``
        # handles it. Drive collisions through Newton's pipeline (the same
        # pattern the upstream examples use when meshes are involved):
        # build the solver with ``use_mujoco_contacts=False`` and call
        # ``model.collide(state, contacts)`` each step before ``solver.step``.
        model = builder.finalize(device=self._device)
        solver = newton.solvers.SolverMuJoCo(
            model,
            njmax=_DEFAULT_MUJOCO_NJMAX,
            use_mujoco_contacts=False,
            use_mujoco_cpu=self._use_mujoco_cpu(),
        )
        contacts = model.contacts()

        state_0 = model.state()
        state_1 = model.state()
        control = model.control()

        # 4. Inject initial velocities (freeform path).
        self._inject_initial_velocity(
            newton,
            model,
            state_0,
            body_idx,
            initial_linear_velocity=initial_linear_velocity,
            initial_angular_velocity=initial_angular_velocity,
        )

        # 5. Sim loop. n_steps total; sample every sample_period steps.
        n_steps = max(1, int(round(duration_s / dt)))
        sample_period = max(1, int(round(1.0 / (dt * sample_fps))))

        trajectory: list[tuple[float, list[float], list[float]]] = []
        body_q = state_0.body_q.numpy()[body_idx]
        body_qd = state_0.body_qd.numpy()[body_idx]
        trajectory.append((0.0, list(map(float, body_q)), list(map(float, body_qd))))

        for step in range(1, n_steps + 1):
            state_0.clear_forces()
            # Re-generate contacts from the current state via Newton's
            # native pipeline; required when we opted out of MuJoCo's
            # built-in contact generation above.
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0, state_1 = state_1, state_0
            if step % sample_period == 0 or step == n_steps:
                t = step * dt
                body_q = state_0.body_q.numpy()[body_idx]
                body_qd = state_0.body_qd.numpy()[body_idx]
                trajectory.append(
                    (
                        float(t),
                        list(map(float, body_q)),
                        list(map(float, body_qd)),
                    )
                )

        final_pose = list(map(float, state_0.body_q.numpy()[body_idx]))
        final_velocity = list(map(float, state_0.body_qd.numpy()[body_idx]))

        return {
            "trajectory": trajectory,
            "final_pose": final_pose,
            "final_velocity": final_velocity,
            "n_bodies": int(model.body_count),
            "duration_s": float(n_steps * dt),
            "n_steps": int(n_steps),
        }

    @staticmethod
    def _validate_timing(*, duration_s: float, dt: float, sample_fps: int) -> None:
        """Reject non-finite or non-positive simulation timing parameters."""
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError(
                f"NewtonSimulator: duration_s must be finite and > 0; got {duration_s}."
            )
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"NewtonSimulator: dt must be finite and > 0; got {dt}.")
        if not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError(
                f"NewtonSimulator: sample_fps must be finite and > 0; got {sample_fps}."
            )

    @staticmethod
    def _inject_initial_velocity(
        newton: Any,
        model: Any,
        state: Any,
        body_idx: int,
        *,
        initial_linear_velocity: Sequence[float] | None,
        initial_angular_velocity: Sequence[float] | None,
    ) -> None:
        """Seed Newton body and joint velocities for the freeform scenario."""
        if initial_linear_velocity is None and initial_angular_velocity is None:
            return
        qd = state.body_qd.numpy()
        lin = (
            NewtonSimulator._velocity_triplet(
                initial_linear_velocity,
                name="initial_linear_velocity",
            )
            if initial_linear_velocity is not None
            else [0.0, 0.0, 0.0]
        )
        ang = (
            NewtonSimulator._velocity_triplet(
                initial_angular_velocity,
                name="initial_angular_velocity",
            )
            if initial_angular_velocity is not None
            else [0.0, 0.0, 0.0]
        )
        # SolverMuJoCo syncs MuJoCo qvel from ``state.joint_qd`` before each
        # step. Convert the world-frame body velocity we just seeded into the
        # corresponding generalized joint velocity so the first step does not
        # silently start from rest.
        joint_idx, dof_count = NewtonSimulator._joint_info_for_body(model, body_idx)
        if dof_count <= 0:
            raise RuntimeError(
                "NewtonSimulator: cannot seed initial velocity for body index "
                f"{body_idx}; no movable Newton joint DOFs are attached to that "
                "body, and SolverMuJoCo consumes joint_qd rather than body_qd."
            )
        if dof_count < 6:
            raise RuntimeError(
                "NewtonSimulator: cannot seed arbitrary world-frame initial "
                f"velocity for body index {body_idx}; attached Newton joint "
                f"has only {dof_count} movable DOFs. Use a free body with 6 "
                "DOFs or omit freeform initial velocity for constrained joints."
            )
        if not hasattr(state, "joint_q") or not hasattr(state, "joint_qd"):
            raise RuntimeError(
                "NewtonSimulator: cannot seed initial velocity; Newton state "
                "does not expose joint_q and joint_qd for SolverMuJoCo sync."
            )
        qd[body_idx] = [*lin, *ang]
        state.body_qd.assign(qd)
        indices = NewtonSimulator._eval_ik_indices_for_joint(model, joint_idx)
        if indices is None:
            newton.eval_ik(model, state, state.joint_q, state.joint_qd)
        else:
            newton.eval_ik(model, state, state.joint_q, state.joint_qd, indices=indices)

    @staticmethod
    def _velocity_triplet(value: Sequence[float], *, name: str) -> list[float]:
        values = [float(component) for component in value]
        if len(values) != 3:
            raise ValueError(
                f"NewtonSimulator: {name} must contain exactly 3 values; "
                f"got {len(values)}."
            )
        return values

    @staticmethod
    def _array_to_list(value: Any, *, name: str) -> list[Any]:
        if value is None:
            return []
        if hasattr(value, "numpy"):
            value = value.numpy()
        try:
            return list(value)
        except TypeError as exc:
            raise RuntimeError(
                f"NewtonSimulator: expected {name} to be sequence-like, got "
                f"{type(value).__name__}."
            ) from exc

    @staticmethod
    def _joint_dof_count_for_body(model: Any, body_idx: int) -> int:
        return NewtonSimulator._joint_info_for_body(model, body_idx)[1]

    @staticmethod
    def _joint_info_for_body(model: Any, body_idx: int) -> tuple[int, int]:
        joint_child = NewtonSimulator._array_to_list(
            getattr(model, "joint_child", None),
            name="model.joint_child",
        )
        joint_qd_start = NewtonSimulator._array_to_list(
            getattr(model, "joint_qd_start", None),
            name="model.joint_qd_start",
        )
        for joint_idx, child_body_idx in enumerate(joint_child):
            if int(child_body_idx) != int(body_idx):
                continue
            if len(joint_qd_start) <= joint_idx + 1:
                return joint_idx, 0
            return (
                joint_idx,
                int(joint_qd_start[joint_idx + 1]) - int(joint_qd_start[joint_idx]),
            )
        return -1, 0

    @staticmethod
    def _eval_ik_indices_for_joint(model: Any, joint_idx: int) -> Any | None:
        joint_articulation = NewtonSimulator._array_to_list(
            getattr(model, "joint_articulation", None),
            name="model.joint_articulation",
        )
        if joint_idx < 0 or len(joint_articulation) <= joint_idx:
            return None
        articulation_idx = int(joint_articulation[joint_idx])
        try:
            import warp as wp
        except ImportError:  # pragma: no cover - Newton depends on Warp
            return None

        device = getattr(model, "device", None)
        try:
            return wp.array([articulation_idx], dtype=wp.int32, device=device)
        except TypeError:
            return wp.array([articulation_idx], dtype=wp.int32)

    @staticmethod
    def _ground_plane_shape_config(
        builder: Any, add_usd_results: dict[str, Any]
    ) -> Any | None:
        """Clone authored ground contact material or return None for defaults."""
        shape_idx = NewtonSimulator._ground_shape_index(add_usd_results)
        if shape_idx is None:
            return None
        shape_config_type = getattr(builder, "ShapeConfig", None)
        if shape_config_type is None:
            logger.debug(
                "Newton builder has no ShapeConfig; injected ground plane will "
                "use Newton's default contact material."
            )
            return None
        fields = {
            "shape_material_ke": "ke",
            "shape_material_kd": "kd",
            "shape_material_kf": "kf",
            "shape_material_ka": "ka",
            "shape_material_mu": "mu",
            "shape_material_restitution": "restitution",
            "shape_material_mu_torsional": "mu_torsional",
            "shape_material_mu_rolling": "mu_rolling",
        }
        values: dict[str, float] = {}
        for builder_attr, cfg_attr in fields.items():
            raw_values = getattr(builder, builder_attr, None)
            if raw_values is None:
                continue
            try:
                if len(raw_values) <= shape_idx:
                    continue
                values[cfg_attr] = float(raw_values[shape_idx])
            except (IndexError, TypeError, ValueError) as exc:
                logger.debug(
                    "Could not read Newton ground material field %s from %s: %s",
                    cfg_attr,
                    builder_attr,
                    exc,
                )
                continue
        if not values:
            logger.debug(
                "Newton builder ground shape has no accessible contact "
                "material arrays; injected ground plane will use Newton's "
                "default contact material."
            )
            return None
        try:
            return shape_config_type(**values)
        except TypeError as exc:
            logger.debug(
                "Could not clone authored ground contact material into "
                "Newton ShapeConfig: %s",
                exc,
            )
            return None

    @staticmethod
    def _ground_shape_index(add_usd_results: dict[str, Any]) -> int | None:
        path_shape_map = dict(add_usd_results.get("path_shape_map") or {})
        if not path_shape_map:
            return None
        candidates = [
            (str(path), int(shape_idx))
            for path, shape_idx in path_shape_map.items()
            if NewtonSimulator._is_ground_shape_path(str(path))
        ]
        for suffix in ("/GroundPlane", "/Ground_Plane", "/Ground"):
            for path, shape_idx in candidates:
                if path.lower().endswith(suffix.lower()):
                    return shape_idx
        return candidates[0][1] if candidates else None

    @staticmethod
    def _is_ground_shape_path(path: str) -> bool:
        last_component = path.rsplit("/", maxsplit=1)[-1].lower()
        return last_component in {"ground", "groundplane", "ground_plane"}

    @staticmethod
    def _add_ground_plane(builder: Any, *, cfg: Any | None = None) -> None:
        """Author a primitive plane; cfg=None uses Newton's default material."""
        if hasattr(builder, "add_ground_plane"):
            if cfg is None:
                builder.add_ground_plane()
            else:
                try:
                    builder.add_ground_plane(cfg=cfg)
                except TypeError as exc:
                    if not NewtonSimulator._looks_like_unknown_cfg_kwarg(exc):
                        raise
                    logger.debug(
                        "Newton builder rejected add_ground_plane(cfg=...); "
                        "falling back to default ground contact material: %s",
                        exc,
                    )
                    builder.add_ground_plane()
            return
        up_vector = getattr(builder, "up_vector", (0.0, 0.0, 1.0))
        kwargs: dict[str, Any] = {
            "plane": (*up_vector, 0.0),
            "width": 0.0,
            "length": 0.0,
        }
        if cfg is not None:
            kwargs["cfg"] = cfg
        try:
            builder.add_shape_plane(**kwargs)
        except TypeError as exc:
            if cfg is None or not NewtonSimulator._looks_like_unknown_cfg_kwarg(exc):
                raise
            logger.debug(
                "Newton builder rejected add_shape_plane(cfg=...); falling "
                "back to default ground contact material: %s",
                exc,
            )
            kwargs.pop("cfg", None)
            builder.add_shape_plane(**kwargs)

    @staticmethod
    def _looks_like_unknown_cfg_kwarg(exc: TypeError) -> bool:
        message = str(exc).lower()
        return "cfg" in message and "unexpected keyword" in message

    @staticmethod
    def _resolve_body_index(body_pattern: str, path_body_map: dict[str, int]) -> int:
        # Scenarios author scene USDs where ``body_pattern`` is the exact
        # USD prim path (``_scene_builder.py:810`` and ``:953``). Exact match
        # is the common case; we keep an fnmatch fallback for robustness if
        # the path representation drifts between USD round-trips.
        import fnmatch

        if body_pattern in path_body_map:
            return path_body_map[body_pattern]

        matches = [
            (path, idx)
            for path, idx in path_body_map.items()
            if fnmatch.fnmatchcase(path, body_pattern)
        ]
        if len(matches) == 1:
            return matches[0][1]
        if not matches:
            raise RuntimeError(
                f"NewtonSimulator: body_pattern {body_pattern!r} matched no "
                f"prim in scene. Known: {sorted(path_body_map)}"
            )
        raise RuntimeError(
            f"NewtonSimulator: body_pattern {body_pattern!r} matched multiple "
            f"prims; refusing to guess. Matches: {[p for p, _ in matches]}"
        )
