# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""USD time-sample helpers for physics-trajectory recording.

The recording.usda authored by ``physics_agent.recording`` carries the
**raw simulator output** — body pose AND velocity — as time samples on
standard schema attributes:

* ``xformOp:translate`` (Vec3d) — body position over time
* ``xformOp:orient`` (Quatf) — body orientation over time
* ``physics:velocity`` (Vec3f, ``UsdPhysics.RigidBodyAPI``) — linear vel
* ``physics:angularVelocity`` (Vec3f, ``UsdPhysics.RigidBodyAPI``) — ang vel

Splitting the transform into separate translate + orient ops (instead
of a single ``xformOp:transform`` matrix) makes the recording directly
inspectable in usdview's property panel without decomposing a 4×4. The
velocity attributes are part of ``UsdPhysics.RigidBodyAPI`` so they
stay schema-compliant.

Derived metrics (max_linear_speed, settle_time, fell_over, etc.) are
computed from these raw values on demand by
``world_understanding.functions.physics.trajectory`` — they are NOT
pre-baked into the recording.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from pxr import Usd

# Pose layout: 7 floats = [px, py, pz, qx, qy, qz, qw]. Matches ovphysx
# ``TensorType.RIGID_BODY_POSE`` shape (N, 7). Quat convention: real==qw last.
_POSE7_LEN = 7

# Velocity layout: 6 floats = [vx, vy, vz, wx, wy, wz]. Matches ovphysx
# ``TensorType.RIGID_BODY_VELOCITY`` shape (N, 6).
_VEL6_LEN = 6


def add_pose_velocity_trajectory(
    prim: Usd.Prim,
    trajectory: Sequence[tuple[float, Sequence[float], Sequence[float]]],
) -> None:
    """Author a full trajectory at once on ``prim``.

    For each ``(time, pose7, vel6)`` tuple writes:

    * ``xformOp:translate`` time sample at ``time`` from ``pose7[0:3]``.
    * ``xformOp:orient`` time sample from ``pose7[3:7]`` (USD ``Quatf``
      convention is ``Quatf(real=qw, imaginary=Vec3f(qx,qy,qz))``).
    * ``physics:velocity`` time sample from ``vel6[0:3]``.
    * ``physics:angularVelocity`` time sample from ``vel6[3:6]``.

    ``time`` is interpreted as **wall-clock seconds**. The function
    multiplies by the stage's ``timeCodesPerSecond`` so the round-trip
    via :func:`read_pose_velocity_trajectory` returns seconds at any
    playback rate. ``GetTimeCodesPerSecond`` always returns a value (USD
    falls back to 24 when not authored), so the multiplication is
    well-defined; the ``> 0.0`` clamp here is purely a guard against a
    stage that explicitly authored 0 or a negative rate.

    The prim's xformOpOrder is replaced with
    ``["xformOp:translate", "xformOp:orient"]`` so renderers compose
    T·R uniformly. Any pre-existing translate/orient ops are reused;
    other ops (transform, scale, rotateXYZ, ...) are dropped from the
    order.

    The velocity attributes use the ``UsdPhysics.RigidBodyAPI``
    schema's ``physics:velocity`` and ``physics:angularVelocity`` —
    standard, schema-compliant, surfaced by usdview without custom
    schema registration.

    Raises:
        TypeError: ``prim`` is not ``UsdGeom.Xformable``.
        ValueError: a tuple in ``trajectory`` has the wrong arity.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise TypeError(f"prim {prim.GetPath()} is not UsdGeom.Xformable")
    stage = prim.GetStage()
    tcps_raw = float(stage.GetTimeCodesPerSecond() or 0.0) if stage is not None else 0.0
    tcps = tcps_raw if tcps_raw > 0.0 else 1.0

    # Compose xformOpOrder = [translate, orient, ...preserved_ops]. The
    # simulator's pose is in world space, so any pre-existing pose ops
    # (matrix transform, rotate-*, duplicate translate/orient) MUST be
    # dropped — they would otherwise stack on top of the time-sampled
    # pose. But ops like ``xformOp:scale`` carry intrinsic body geometry
    # (e.g. a unit-cube scaled to 0.5 m) that the simulator doesn't
    # know about, so we keep those tail ops. USD applies xformOpOrder
    # right-to-left: ``[T, R, S]`` ⇒ M = T·R·S, scale-then-rotate-then-translate
    # — the standard SRT composition.
    pose_op_types = {
        UsdGeom.XformOp.TypeTranslate,
        # Axis-translate ops (e.g. ``xformOp:translate:x``) would stack
        # on top of the time-sampled translate and create a double offset.
        UsdGeom.XformOp.TypeTranslateX,
        UsdGeom.XformOp.TypeTranslateY,
        UsdGeom.XformOp.TypeTranslateZ,
        UsdGeom.XformOp.TypeOrient,
        UsdGeom.XformOp.TypeTransform,
        UsdGeom.XformOp.TypeRotateX,
        UsdGeom.XformOp.TypeRotateY,
        UsdGeom.XformOp.TypeRotateZ,
        UsdGeom.XformOp.TypeRotateXYZ,
        UsdGeom.XformOp.TypeRotateXZY,
        UsdGeom.XformOp.TypeRotateYXZ,
        UsdGeom.XformOp.TypeRotateYZX,
        UsdGeom.XformOp.TypeRotateZXY,
        UsdGeom.XformOp.TypeRotateZYX,
    }
    existing = list(xformable.GetOrderedXformOps())
    translate_op = None
    orient_op = None
    preserved_ops: list[Any] = []
    for op in existing:
        t = op.GetOpType()
        if t == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif t == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op
        elif t not in pose_op_types:
            preserved_ops.append(op)
    if translate_op is None:
        translate_op = xformable.AddTranslateOp()
    if orient_op is None:
        # AddOrientOp authors a Quatf op by default — matches our
        # pose7 quaternion precision.
        orient_op = xformable.AddOrientOp()
    xformable.SetXformOpOrder([translate_op, orient_op, *preserved_ops])

    # Velocity / angular velocity attrs from the standard schema. We
    # don't require RigidBodyAPI to already be applied (the recorder's
    # input USD has it applied; defensive Apply is cheap and idempotent).
    rb_api = UsdPhysics.RigidBodyAPI.Apply(prim)
    velocity_attr = rb_api.GetVelocityAttr() or rb_api.CreateVelocityAttr(
        Gf.Vec3f(0, 0, 0)
    )
    angvel_attr = rb_api.GetAngularVelocityAttr() or rb_api.CreateAngularVelocityAttr(
        Gf.Vec3f(0, 0, 0)
    )

    # Wipe any pre-existing time samples on the four attrs we are about
    # to author. If the source USD already has trajectory animation on
    # the body (e.g. someone re-records over an existing recording, or
    # the source scene baked physics:velocity for setup), prior samples
    # would interleave with the new ones and corrupt playback. Clearing
    # is cheap (one Sdf-level edit per attr) and produces a recording
    # whose only animation comes from this call.
    for attr in (
        translate_op.GetAttr(),
        orient_op.GetAttr(),
        velocity_attr,
        angvel_attr,
    ):
        if attr.GetTimeSamples():
            attr.Clear()

    for entry in trajectory:
        if len(entry) != 3:
            raise ValueError(
                "trajectory entries must be (time, pose7, vel6) 3-tuples; "
                f"got an entry of length {len(entry)}"
            )
        time, pose7, vel6 = entry
        if len(pose7) != _POSE7_LEN:
            raise ValueError(f"pose7 must be length {_POSE7_LEN}, got {len(pose7)}")
        if len(vel6) != _VEL6_LEN:
            raise ValueError(f"vel6 must be length {_VEL6_LEN}, got {len(vel6)}")
        px, py, pz, qx, qy, qz, qw = (float(v) for v in pose7)
        vx, vy, vz, wx, wy, wz = (float(v) for v in vel6)
        # Convert seconds → USD timecode using the stage's playback rate.
        tc = Usd.TimeCode(float(time) * tcps)

        translate_op.Set(Gf.Vec3d(px, py, pz), time=tc)
        orient_op.Set(Gf.Quatf(qw, Gf.Vec3f(qx, qy, qz)), time=tc)
        velocity_attr.Set(Gf.Vec3f(vx, vy, vz), time=tc)
        angvel_attr.Set(Gf.Vec3f(wx, wy, wz), time=tc)

    # Avoid 'unused import' lint when Sdf isn't otherwise referenced.
    _ = Sdf


def read_pose_velocity_trajectory(
    stage: Usd.Stage,
    prim_path: str,
) -> list[tuple[float, list[float], list[float]]]:
    """Inverse of :func:`add_pose_velocity_trajectory`.

    Reads ``xformOp:translate``, ``xformOp:orient``, ``physics:velocity``
    and ``physics:angularVelocity`` time samples on ``prim_path`` and
    returns ``[(t, pose7, vel6), ...]`` — same shape the daemon emits.

    The returned ``t`` values are wall-clock SECONDS, the inverse of
    :func:`add_pose_velocity_trajectory` which accepts seconds and
    converts to USD timecodes via the stage's ``timeCodesPerSecond``. A
    stage with ``timeCodesPerSecond <= 0`` (or unset) is treated as 1.0
    (timecodes already in seconds).

    Lets a downstream consumer validate the simulation by opening the
    recording: ``traj = read_pose_velocity_trajectory(stage,
    "/World/Body"); max_speed = max_linear_speed(traj)``.

    Returns an empty list when the prim has no time samples on any of
    the four attributes. When sample sets disagree (which shouldn't
    happen for our recorder output), uses the union of timecodes and
    fills missing values from the static (non-time-sampled) attribute
    value, falling back to zeros.

    Raises:
        ValueError: ``prim_path`` is missing or not Xformable.
    """
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # type: ignore[import-untyped]

    prim = stage.GetPrimAtPath(Sdf.Path(prim_path))
    if not prim:
        raise ValueError(f"prim_path {prim_path!r} not found in stage")
    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise ValueError(f"prim {prim_path!r} is not Xformable")

    translate_attr = None
    orient_attr = None
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_attr is None:
            translate_attr = op.GetAttr()
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_attr is None:
            orient_attr = op.GetAttr()

    rb_api = UsdPhysics.RigidBodyAPI(prim)
    velocity_attr = rb_api.GetVelocityAttr() if rb_api else None
    angvel_attr = rb_api.GetAngularVelocityAttr() if rb_api else None

    timecodes: set[float] = set()
    for attr in (translate_attr, orient_attr, velocity_attr, angvel_attr):
        if attr is not None:
            for tc in attr.GetTimeSamples():
                timecodes.add(float(tc))

    if not timecodes:
        return []

    # Recorder authors at frame indices (t_seconds * fps) and pins
    # timeCodesPerSecond to fps; convert back to seconds on read.
    tcps = float(stage.GetTimeCodesPerSecond() or 0.0)
    seconds_per_timecode = 1.0 / tcps if tcps > 0.0 else 1.0

    def _vec3_to_list(v: object) -> list[float]:
        if v is None:
            return [0.0, 0.0, 0.0]
        return [float(v[0]), float(v[1]), float(v[2])]

    def _quat_to_xyzw(q: object) -> list[float]:
        # USD Quatf is (real=w, imaginary=(x,y,z)); pose7 is (qx,qy,qz,qw).
        if q is None:
            return [0.0, 0.0, 0.0, 1.0]
        imag = q.GetImaginary()
        return [float(imag[0]), float(imag[1]), float(imag[2]), float(q.GetReal())]

    out: list[tuple[float, list[float], list[float]]] = []
    for tc in sorted(timecodes):
        timecode = Usd.TimeCode(tc)
        translation = (
            translate_attr.Get(timecode) if translate_attr is not None else None
        )
        orientation = orient_attr.Get(timecode) if orient_attr is not None else None
        linear = velocity_attr.Get(timecode) if velocity_attr is not None else None
        angular = angvel_attr.Get(timecode) if angvel_attr is not None else None
        pose7 = _vec3_to_list(translation) + _quat_to_xyzw(orientation)
        vel6 = _vec3_to_list(linear) + _vec3_to_list(angular)
        out.append((tc * seconds_per_timecode, pose7, vel6))
    _ = Gf
    return out


# ---------------------------------------------------------------------------
# Deprecated single-pose / matrix-transform helpers
# ---------------------------------------------------------------------------
# Kept for callers that want a single static pose written as a matrix.
# Trajectory authoring should use ``add_pose_velocity_trajectory`` above.


def _matrix_from_pose7(pose7: list[float]) -> object:
    """Build a ``Gf.Matrix4d`` from ``[px,py,pz,qx,qy,qz,qw]``."""
    from pxr import Gf  # type: ignore[import-untyped]

    if len(pose7) != _POSE7_LEN:
        raise ValueError(
            f"pose7 must be length {_POSE7_LEN}, got {len(pose7)}: {pose7!r}"
        )
    px, py, pz, qx, qy, qz, qw = (float(v) for v in pose7)
    matrix = Gf.Matrix4d()
    matrix.SetIdentity()
    matrix.SetRotateOnly(Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz)))
    matrix.SetTranslateOnly(Gf.Vec3d(px, py, pz))
    return matrix


def add_xform_time_sample(
    prim: Usd.Prim,
    time: float,
    pose7: list[float],
) -> None:
    """Author one ``xformOp:transform`` time sample on ``prim``.

    Deprecated for trajectory authoring; use
    :func:`add_pose_velocity_trajectory` for the time-sampled
    physics-recording path. Kept for static / single-pose use cases
    where a 4×4 matrix is the natural representation.
    """
    from pxr import Usd, UsdGeom  # type: ignore[import-untyped]

    xformable = UsdGeom.Xformable(prim)
    if not xformable:
        raise TypeError(f"prim {prim.GetPath()} is not UsdGeom.Xformable")
    matrix = _matrix_from_pose7(pose7)

    existing_ops = xformable.GetOrderedXformOps()
    transform_op = None
    for op in existing_ops:
        if op.GetOpType() == UsdGeom.XformOp.TypeTransform:
            transform_op = op
            break
    if transform_op is None:
        transform_op = xformable.AddTransformOp()
        xformable.SetXformOpOrder([transform_op])
    transform_op.Set(matrix, time=Usd.TimeCode(time))


def iter_time_samples(
    stage: Usd.Stage,
) -> Iterator[tuple[float, dict[str, list[float]]]]:
    """Iterate ``(timecode, {prim_path: pose7})`` over every transform-op
    time sample in the stage.

    Deprecated for the recording-trajectory path; use
    :func:`read_pose_velocity_trajectory` to read the new
    translate/orient/physics:velocity layout. Kept for matrix-based
    iteration in test fixtures.
    """
    from pxr import UsdGeom  # noqa: F401

    by_time: dict[float, dict[str, list[float]]] = {}
    for prim in stage.Traverse():
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            continue
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() != UsdGeom.XformOp.TypeTransform:
                continue
            attr = op.GetAttr()
            for tc in attr.GetTimeSamples():
                from pxr import Usd

                value = attr.Get(Usd.TimeCode(tc))
                if value is None:
                    continue
                pose7 = _matrix_to_pose7(value)
                by_time.setdefault(float(tc), {})[str(prim.GetPath())] = pose7

    for tc in sorted(by_time):
        yield tc, by_time[tc]


def _matrix_to_pose7(matrix: object) -> list[float]:
    """Inverse of :func:`_matrix_from_pose7`."""
    from pxr import Gf  # type: ignore[import-untyped]

    if not isinstance(matrix, Gf.Matrix4d):
        raise TypeError(f"expected Gf.Matrix4d, got {type(matrix).__name__}")
    translation = matrix.ExtractTranslation()
    quat = matrix.ExtractRotationQuat()
    real = quat.GetReal()
    imag = quat.GetImaginary()
    return [
        float(translation[0]),
        float(translation[1]),
        float(translation[2]),
        float(imag[0]),
        float(imag[1]),
        float(imag[2]),
        float(real),
    ]


__all__ = [
    "add_pose_velocity_trajectory",
    "read_pose_velocity_trajectory",
    "add_xform_time_sample",
    "iter_time_samples",
]
