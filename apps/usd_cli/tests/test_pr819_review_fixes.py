# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the five review findings from world-understanding PR #819.

The findings were reported against the usd-cli copy vendored into that PR and are
fixed here upstream (branch port/pr819-review-fixes) so the fix re-syncs cleanly:

1. Runtime simulation applied a hard-coded excursion verdict instead of returning
   raw metrics for the workflow to evaluate.
2. `duration_s / dt` step totals were unbounded on the client path (`--duration
   60 --dt 1e-6` = 60,000,000 steps); the service model and the runtime now share
   one MAX_PHYSICS_STEPS limit.
3. `author_trajectory_usda` cleared the body's whole xformOpOrder, so an
   intrinsic `xformOp:scale` was lost and recordings played back unscaled.
4. OVRTX auto-provisioning ran bare unpinned `pip install`; it now installs only
   the checked-in hash-pinned pylock via uv (mirroring the ovphysx path).

Everything here runs without a GPU/solver: pxr-only authoring plus monkeypatched
executors, the same pattern as tests/test_remote_physics.py.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shlex as _shlex
import shutil as _shutil
import sys
import tomllib
from pathlib import Path

import pytest
from conftest import ovrtx_lock_this_interpreter_supports

pytest.importorskip("pxr")

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_DIR = REPO_ROOT / "apps" / "ovrtx_rendering_api"

#: Only these hosts may serve artifacts named by the OVRTX runtime lock. The
#: component gate's secrets scan excludes the lock (its sha256 digests read as
#: high-entropy hex), so the lock gets this stricter shape check instead — a
#: credential smuggled into an artifact URL (https://user:token@host/...)
#: would fail the startswith test.
OVRTX_LOCK_ARTIFACT_HOSTS = (
    "https://files.pythonhosted.org/",
    "https://pypi.nvidia.com/",
)


def _cm_stage(tmp_path, *, cube_size: float = 100.0, name: str = "src.usda",
              meters_per_unit: float = 0.01):
    """A rigid-body cube on a non-metre stage (default: centimetres)."""
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateNew(str(tmp_path / name))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, meters_per_unit)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    cube = UsdGeom.Cube.Define(stage, "/World/Body")
    cube.CreateSizeAttr(float(cube_size))
    h = cube_size / 2.0
    cube.CreateExtentAttr([(-h, -h, -h), (h, h, h)])
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    UsdPhysics.Scene.Define(stage, "/PhysicsScenario")
    stage.GetRootLayer().Save()
    return stage


def _still_trajectory(pos):
    """[t, pose7, vel6] samples settling at `pos` (already still — settle detector
    sees lin speed < 0.05 sustained from t=0)."""
    x, y, z = (float(v) for v in pos)
    still = [0.0] * 6
    return [[0.1 * i, [x, y, z, 0.0, 0.0, 0.0, 1.0], list(still)] for i in range(6)]


# ── 1. simulation reports facts instead of workflow verdicts ────────────────────────


# ── 2. bounded-range acceptance compares metres, not stage units ─────────────────


def _run_simulation_with_canned_remote(tmp_path, monkeypatch, stage, rest_pos):
    from usd_core import physics_runtime

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    monkeypatch.setattr(
        physics_runtime,
        "evaluate_remote",
        lambda scene_usd, **kw: {
            "trajectory": _still_trajectory(rest_pos),
            "n_bodies": 1,
            "n_steps": 240,
        },
    )
    return physics_runtime.simulate_scene(
        stage.GetRootLayer().realPath,
        str(tmp_path / "out"),
        body_path="/World/Body",
        rest_position=list(rest_pos),
        world_up=[0.0, 0.0, 1.0],
        remote={"base_url": "http://gpu:8000"},
    )


def _satisfiable_lock_text(ovrtx_version: str) -> str:
    """A minimal pylock the provisioner accepts: any interpreter, and the
    ovrtx the current pin names. Provisioning refuses a lock that would not
    produce OVRTX_PIN -- a stale override could otherwise delete the runtime,
    install its own ovrtx and be stamped current -- so both halves matter."""
    from usd_core.render import ovrtx

    profile = {
        "ovstage": ovrtx._QUALIFIED_RUNTIME_PROFILE["ovstage"],
        "warp-lang": ovrtx._QUALIFIED_RUNTIME_PROFILE["warp-lang"],
        "ovrtx": ovrtx_version,
    }
    packages = "".join(
        '[[packages]]\nname = "%s"\nversion = "%s"\n' % item
        for item in profile.items()
    )
    return 'requires-python = ">=3.0"\n' + packages


def _qualified_probe_stdout(ovrtx) -> str:
    return json.dumps(ovrtx._QUALIFIED_RUNTIME_PROFILE, sort_keys=True) + "\n"


def _world_understanding_graphics() -> Path | None:
    """world_understanding/functions/graphics, or None when it is absent.

    A fixed parent depth is wrong in two directions: usd-cli is vendored
    upstream as a standalone repo with no world_understanding tree, and in
    this repo it has already moved once (root -> apps/usd_cli). Walk up.
    """
    for base in (REPO_ROOT, *REPO_ROOT.parents):
        candidate = base / "world_understanding" / "functions" / "graphics"
        if candidate.is_dir():
            return candidate
    return None


def _assert_ready_marker(marker: Path, ovrtx, runtime_lock: Path) -> None:
    """Assert both the semantic and byte-for-byte readiness contract."""
    body = marker.read_text(encoding="utf-8")
    assert body == ovrtx._readiness_marker_body(runtime_lock)
    assert json.loads(body) == {
        "packages": ovrtx._QUALIFIED_RUNTIME_PROFILE,
        ovrtx._LOCK_DIGEST_KEY: ovrtx._runtime_lock_digest(runtime_lock),
        "schema_version": ovrtx._READY_MARKER_SCHEMA_VERSION,
    }


def test_runtime_reports_metric_excursion_without_applying_a_bound(tmp_path, monkeypatch):
    """The CLI reports raw trajectory metrics; workflow policy owns excursion bounds."""
    report = _run_simulation_with_canned_remote(
        tmp_path, monkeypatch, _cm_stage(tmp_path), rest_pos=(150.0, 0.0, 5.0))
    assert report["metrics"]["max_abs_position"] == pytest.approx(150.0)
    assert report["simulation_facts"]["trajectory_finite"] is True
    assert "ok" not in report


def test_runtime_does_not_apply_a_hard_coded_excursion_threshold(tmp_path, monkeypatch):
    """A large excursion remains an observable metric, not a CLI failure verdict."""
    report = _run_simulation_with_canned_remote(
        tmp_path, monkeypatch,
        _cm_stage(tmp_path, cube_size=2.0, meters_per_unit=1.0),
        rest_pos=(150.0, 0.0, 1.0))
    assert report["metrics"]["max_abs_position"] == pytest.approx(150.0)
    assert report["simulation_facts"]["trajectory_finite"] is True
    assert "ok" not in report


# ── 3. MAX_PHYSICS_STEPS bounds duration_s / dt on both sides ────────────────────


def test_step_total_guard_rejects_runaway_duration_dt(tmp_path, monkeypatch):
    """--duration 60 --dt 1e-6 (60,000,000 steps) is rejected up front — before
    the scene is authored, a daemon is provisioned, or anything is uploaded."""
    from usd_core import physics_runtime

    monkeypatch.setattr(physics_runtime, "ovphysx_platform_supported", lambda: False)
    monkeypatch.setattr(
        physics_runtime, "evaluate_remote",
        lambda *a, **kw: pytest.fail("solver invoked despite runaway step total"))
    with pytest.raises(ValueError, match=r"60000000 simulation steps.*limit of 120000"):
        stage = _cm_stage(tmp_path)
        physics_runtime.simulate_scene(
            stage.GetRootLayer().realPath,
            str(tmp_path / "out"),
            body_path="/World/Body",
            rest_position=[0.0, 0.0, 0.0],
            world_up=[0.0, 0.0, 1.0],
            duration_s=60.0,
            dt=1e-6,
            remote={"base_url": "http://gpu:8000"},
        )
    assert not (tmp_path / "out").exists()  # rejected before any output authoring


def test_step_total_guard_admits_documented_maximum():
    """The service model's documented headroom case — a full 60 s at the default
    1/240 s timestep — stays admissible, and the limit itself is inclusive."""
    from usd_core.physics_runtime import MAX_PHYSICS_STEPS, _checked_total_steps

    assert _checked_total_steps(60.0, 1.0 / 240.0) == 14_400
    assert _checked_total_steps(float(MAX_PHYSICS_STEPS), 1.0) == MAX_PHYSICS_STEPS
    with pytest.raises(ValueError, match="raise dt or shorten duration_s"):
        _checked_total_steps(float(MAX_PHYSICS_STEPS + 1), 1.0)


def test_service_model_enforces_the_same_step_limit():
    """The service-side PhysicsUploadParams rejects the same runaway request the
    client does, with the same default ceiling."""
    pytest.importorskip("pydantic")
    from pydantic import ValidationError

    sys.path.insert(0, str(SERVICE_DIR))
    from service import models as service_models
    from usd_core.physics_runtime import MAX_PHYSICS_STEPS as client_limit

    assert service_models.MAX_PHYSICS_STEPS == client_limit
    with pytest.raises(ValidationError, match="simulation steps"):
        service_models.PhysicsUploadParams(body_pattern="/World/Body",
                                           duration_s=60.0, dt=1e-6)
    ok = service_models.PhysicsUploadParams(body_pattern="/World/Body",
                                            duration_s=60.0, dt=1.0 / 240.0)
    assert ok.duration_s == 60.0


# ── 4. recordings preserve non-pose xform ops (intrinsic scale) ──────────────────


def test_recording_preserves_intrinsic_scale(tmp_path):
    """A body sized by xformOp:scale keeps that scale in the recording; only the
    pose ops (translate/rotate/orient/matrix) are replaced by the simulated pose."""
    from pxr import Gf, Usd, UsdGeom
    from usd_core.physics_runtime import author_trajectory_usda

    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    cube = UsdGeom.Cube.Define(stage, "/World/Body")
    xf = UsdGeom.Xformable(cube)
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 5.0))
    xf.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 45.0))
    xf.AddScaleOp().Set(Gf.Vec3f(2.0, 3.0, 4.0))
    stage.GetRootLayer().Save()

    traj = [(0.0, [0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
            (0.1, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6)]
    out = author_trajectory_usda(str(scene), traj, "/World/Body",
                                 str(tmp_path / "recording.usda"))

    rec = Usd.Stage.Open(out)
    ops = UsdGeom.Xformable(rec.GetPrimAtPath("/World/Body")).GetOrderedXformOps()
    names = [op.GetOpName() for op in ops]
    # pose ops first (T·R), the intrinsic scale preserved as the tail op
    assert names == ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
    assert Gf.Vec3f(ops[2].Get()) == Gf.Vec3f(2.0, 3.0, 4.0)
    assert ops[0].GetAttr().GetTimeSamples() == [0.0, 1.0]
    assert ops[1].GetAttr().GetTimeSamples() == [0.0, 1.0]
    assert "xformOp:rotateXYZ" not in names  # replaced by the simulated orient


def test_rerecording_clears_stale_pose_samples(tmp_path):
    """Re-recording over a recording must not interleave old and new samples."""
    from pxr import Usd, UsdGeom
    from usd_core.physics_runtime import author_trajectory_usda

    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Cube.Define(stage, "/World/Body")
    stage.GetRootLayer().Save()

    long_traj = [(0.1 * i, [0.0, 0.0, float(i), 0.0, 0.0, 0.0, 1.0], [0.0] * 6)
                 for i in range(5)]
    first = author_trajectory_usda(str(scene), long_traj, "/World/Body",
                                   str(tmp_path / "rec1.usda"))
    short_traj = long_traj[:2]
    second = author_trajectory_usda(first, short_traj, "/World/Body",
                                    str(tmp_path / "rec2.usda"))
    rec = Usd.Stage.Open(second)
    t_attr = rec.GetPrimAtPath("/World/Body").GetAttribute("xformOp:translate")
    assert t_attr.GetTimeSamples() == [0.0, 1.0]  # the 5 stale samples are gone


def test_recording_rebases_relative_asset_paths_to_its_output_directory(tmp_path):
    """A recording in a side directory keeps source-relative assets resolvable."""
    from pxr import Sdf, Usd, UsdGeom, UsdShade, UsdUtils
    from usd_core.physics_runtime import author_trajectory_usda

    texture = tmp_path / "scene_assets" / "diffuse.png"
    texture.parent.mkdir()
    texture.write_bytes(b"texture fixture")
    scene = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(scene))
    UsdGeom.Xform.Define(stage, "/World/Body")
    shader = UsdShade.Shader.Define(stage, "/World/Shader")
    shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
        Sdf.AssetPath("scene_assets/diffuse.png")
    )
    stage.GetRootLayer().Save()
    trajectory = [
        (0.0, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], [0.0] * 6),
    ]
    output = tmp_path / "simulation" / "recording.usda"

    author_trajectory_usda(
        str(scene), trajectory, "/World/Body", str(output)
    )

    recording = Usd.Stage.Open(str(output))
    assert recording is not None
    authored = recording.GetPrimAtPath("/World/Shader").GetAttribute("inputs:file")
    assert authored.Get().path == "../scene_assets/diffuse.png"
    _layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(output))
    assert not unresolved
    assert str(texture.resolve()) in {str(asset) for asset in assets}


# ── 5. OVRTX provisioning installs only the checked-in hash-pinned lock ──────────


def test_every_ovrtx_runtime_pin_agrees_with_the_module_profile() -> None:
    """Every packaged copy of the qualified profile must remain exact.

    `OVRTX_PIN` and the compiled lock are the runtime pair, but the service
    Dockerfile provisions its own venv and `ovrtx_runtime_profile.in` is what a
    lock regeneration reads. A bump that misses either delivers or restores the
    previous version silently.
    """
    from usd_core.render import ovrtx

    lock = tomllib.loads(ovrtx.OVRTX_RUNTIME_LOCK.read_text())
    packages = {p["name"]: p for p in lock["packages"]}
    assert {
        name: packages[name]["version"]
        for name in ovrtx._QUALIFIED_RUNTIME_PROFILE
    } == ovrtx._QUALIFIED_RUNTIME_PROFILE

    profile = REPO_ROOT / "src" / "usd_core" / "render" / "ovrtx_runtime_profile.in"
    profile_pins = {
        line.partition("==")[0]: line.partition("==")[2]
        for line in profile.read_text().splitlines()
        if "==" in line and line.partition("==")[0] in ovrtx._QUALIFIED_RUNTIME_PROFILE
    }
    assert profile_pins == ovrtx._QUALIFIED_RUNTIME_PROFILE, (
        f"{profile.name} pins {profile_pins}, expected "
        f"{ovrtx._QUALIFIED_RUNTIME_PROFILE}; "
        "regenerating the lock from this file would revert the runtime"
    )

    # The fifth copy of the pin, and the one `uv sync --extra ovrtx` installs --
    # which is the first remedy this module prints. A bump that misses it leaves
    # the documented pre-install path installing a version the ambient probe
    # then rejects, with the suite still green.
    extra = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    declared = extra["project"]["optional-dependencies"]["ovrtx"]
    for package_name, version in ovrtx._QUALIFIED_RUNTIME_PROFILE.items():
        pin = f"{package_name}=={version}"
        assert any(pin in item for item in declared), (
            f"pyproject.toml declares {declared} for the ovrtx extra but omits "
            f"{pin}; uv sync --extra ovrtx would install an incomplete profile"
        )

    dockerfile = SERVICE_DIR / "Dockerfile"
    docker_text = dockerfile.read_text()
    for version in ovrtx._QUALIFIED_RUNTIME_PROFILE.values():
        assert version in docker_text, (
            f"{dockerfile.name} omits qualified runtime version {version}"
        )


def test_an_ambient_ovrtx_of_the_wrong_version_is_not_used(monkeypatch) -> None:
    """The ambient interpreter must agree with the pin, not merely import.

    The probe used to accept any interpreter where ``import ovrtx`` worked, so
    a host with the previous release installed in the project environment kept
    serving every render from it after a bump -- the same staleness the venv
    marker check closes, arriving by the other path.
    """
    import subprocess

    from usd_core.render import ovrtx

    class _Result:
        def __init__(self, stdout):
            self.returncode = 0
            self.stdout = stdout

    wrong = {**ovrtx._QUALIFIED_RUNTIME_PROFILE, "ovrtx": "0.0.0.000000"}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Result(json.dumps(wrong) + "\n")
    )
    assert ovrtx._env_ovrtx_python() is None, (
        "an ambient ovrtx older than the pin must not serve renders"
    )

    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: _Result(_qualified_probe_stdout(ovrtx))
    )
    assert ovrtx._env_ovrtx_python() == sys.executable


def test_a_runtime_this_interpreter_cannot_rebuild_is_not_deleted(tmp_path, monkeypatch) -> None:
    """Never destroy a runtime the current interpreter could not reinstall.

    Replacement is staged before activation, but an unsupported lock cannot
    produce a verified sibling. Refuse it before a multi-gigabyte doomed
    install and leave the current renderer untouched.
    """
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")
    survivor = venv_dir / "lib" / "keep.bin"
    survivor.parent.mkdir(parents=True)
    survivor.write_text("the only working runtime on this host")

    lock = tmp_path / "lock.toml"
    lock.write_text('requires-python = ">=3.99"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())

    with _pytest.raises(RuntimeError, match="refusing to replace"):
        ovrtx._provision_venv(venv_dir)

    assert survivor.read_text() == "the only working runtime on this host"
    assert (venv_dir / "bin" / "python").exists()

    # A spec this check cannot evaluate must also refuse: WU_OVRTX_RUNTIME_LOCK
    # is an operator override, so an unfamiliar requires-python can arrive here,
    # and guessing 'satisfiable' would route straight to delete-then-fail.
    # (~= and >=X.Y.Z are evaluated now -- a pre-release bound still is not.)
    lock.write_text('requires-python = ">=3.12b1"' + chr(10), encoding="utf-8")
    with _pytest.raises(RuntimeError, match="cannot be evaluated"):
        ovrtx._provision_venv(venv_dir)
    assert survivor.read_text() == "the only working runtime on this host"

    # A two-component bound normalises to X.Y.0 under PEP 440, so a patch
    # release does not satisfy ==X.Y or <=X.Y. Comparing truncated tuples
    # called these satisfied and routed straight to the delete.
    for unsatisfied in ("==%d.%d" % sys.version_info[:2],
                        "<=%d.%d" % sys.version_info[:2]):
        if sys.version_info[2] == 0:
            continue  # on an exact X.Y.0 these are genuinely satisfied
        lock.write_text(
            'requires-python = "%s"' % unsatisfied + chr(10), encoding="utf-8"
        )
        with _pytest.raises(RuntimeError, match="refusing to replace"):
            ovrtx._provision_venv(venv_dir)
        assert survivor.read_text() == "the only working runtime on this host"

def test_a_marked_tree_without_an_interpreter_is_still_replaced(tmp_path, monkeypatch) -> None:
    """Ownership comes from the markers, not from bin/python existing.

    A tree can carry our marker and be missing its interpreter -- a delete that
    failed partway, a hand-removed python, a Windows layout keeping it under
    Scripts/. Deciding on the interpreter sent all of those down the in-place
    upgrade this replacement exists to avoid.
    """
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")
    leftover = venv_dir / "lib" / "ovrtx" / "bin" / "cache" / "shader.bin"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("cache from the previous wheel")
    # no bin/python: every earlier guard keyed on it would skip the replace

    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])

    def _fake_run(cmd, *, what, **kwargs):
        if "venv" in cmd:
            staged = Path(cmd[-1])
            assert staged != venv_dir
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").write_text("")

    monkeypatch.setattr(ovrtx, "_run_logged", _fake_run)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    ovrtx._provision_venv(venv_dir)

    assert not leftover.exists(), (
        "a marked tree missing its interpreter was upgraded in place"
    )
    _assert_ready_marker(
        venv_dir / ovrtx._READY_MARKER_NAME, ovrtx, runtime_lock
    )


def test_a_populated_non_venv_directory_is_never_claimed(tmp_path, monkeypatch) -> None:
    """A typo'd WU_OVRTX_VENV_DIR must not put someone's files on death row.

    Every ownership check is gated on bin/python existing, so a directory that
    is populated but is not a venv slips past all of them. Claiming it would
    stamp the provisioning marker, and the next retry would then treat the
    whole directory as ours and rmtree it.
    """
    import pytest as _pytest

    from usd_core.render import ovrtx

    target = tmp_path / "home"
    target.mkdir()
    keepsake = target / "thesis.txt"
    keepsake.write_text("years of work")

    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())

    with _pytest.raises(RuntimeError, match="not a runtime usd-cli created"):
        ovrtx._provision_venv(target)

    assert keepsake.read_text() == "years of work"
    assert not (target / ovrtx._PROVISIONING_MARKER_NAME).exists(), (
        "claiming the directory is what makes the next retry delete it"
    )


def test_a_matching_wu_runtime_is_reused_not_refused(tmp_path, monkeypatch) -> None:
    """The documented shared configuration must keep working, read-only.

    Three checked-in docs tell operators to export WU_OVRTX_VENV_DIR at
    world_understanding's managed runtime. Refusing it outright protected the
    tree but broke that configuration; when the interpreter already has the
    pinned ovrtx there is nothing to provision, so it is simply used.
    """
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    py = venv_dir / "bin" / "python"
    py.write_text("")
    (venv_dir / ovrtx._WU_MANAGED_MARKER_NAME).write_text("owned by wu")

    class _Result:
        returncode = 0
        stdout = _qualified_probe_stdout(ovrtx)

    monkeypatch.setattr(ovrtx.subprocess, "run", lambda *a, **k: _Result())

    def _never(*a, **k):
        raise AssertionError("must not provision into a runtime we do not own")

    monkeypatch.setattr(ovrtx, "_provision_venv", _never)
    assert ovrtx._ovrtx_python(venv_dir, auto_install=True) == str(py)
    assert not (venv_dir / ovrtx._READY_MARKER_NAME).exists(), (
        "reusing a shared runtime must not stamp it as ours"
    )
    assert not (venv_dir / ovrtx._PROVISIONING_MARKER_NAME).exists()


def test_a_co_managed_runtime_is_never_replaced(tmp_path, monkeypatch) -> None:
    """Our marker is not proof of ownership when wu's marker is there too.

    The previous provisioner installed into whatever tree already existed and
    stamped it, so a directory shared with world_understanding carries both
    markers. Treating ours as exclusive would rmtree wu's managed runtime --
    the destruction this guard exists to prevent.
    """
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    (venv_dir / ovrtx._WU_MANAGED_MARKER_NAME).write_text(
        "owned by world_understanding")
    # the stale usd-cli marker a previous in-place install left behind
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())

    with _pytest.raises(RuntimeError, match="world-understanding managed") as caught:
        ovrtx._provision_venv(venv_dir)

    message = str(caught.value)
    assert all(
        pin in message for pin in (ovrtx.OVRTX_PIN, ovrtx.OVSTAGE_PIN, ovrtx.WARP_PIN)
    )
    assert (venv_dir / ovrtx._WU_MANAGED_MARKER_NAME).exists(), (
        "wu's managed runtime must survive untouched"
    )
    assert (venv_dir / "bin" / "python").exists()


def test_an_interrupted_install_leaves_the_tree_replaceable(tmp_path, monkeypatch) -> None:
    """A failed sibling install is cleaned up and a later retry can succeed."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])

    def _make_venv(path: Path) -> None:
        (path / "bin").mkdir(parents=True, exist_ok=True)
        (path / "bin" / "python").write_text("")

    def _die_during_install(cmd, *, what, **kwargs):
        if "venv" in cmd:
            _make_venv(Path(cmd[-1]))
            return
        raise RuntimeError("network dropped at 80%")

    monkeypatch.setattr(ovrtx, "_run_logged", _die_during_install)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    with _pytest.raises(RuntimeError, match="network dropped"):
        ovrtx._provision_venv(venv_dir)

    assert not venv_dir.exists()
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))

    def _succeed(cmd, *, what, **kwargs):
        if "venv" in cmd:
            _make_venv(Path(cmd[-1]))

    monkeypatch.setattr(ovrtx, "_run_logged", _succeed)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    ovrtx._provision_venv(venv_dir)

    _assert_ready_marker(
        venv_dir / ovrtx._READY_MARKER_NAME, ovrtx, runtime_lock
    )
    assert not (venv_dir / ovrtx._PROVISIONING_MARKER_NAME).exists()


def test_a_venv_from_an_older_pin_is_replaced_not_upgraded(tmp_path, monkeypatch) -> None:
    """The stale tree must be gone, not installed over.

    uv removes only what the old distribution's RECORD lists, so anything the
    previous wheel produced after install -- ovrtx keeps an in-package shader
    cache -- would survive an in-place upgrade and then be marked ready.
    """
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")
    leftover = venv_dir / "lib" / "ovrtx" / "bin" / "cache" / "shader.bin"
    leftover.parent.mkdir(parents=True)
    leftover.write_text("cache written by the 0.3 wheel after install")

    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])

    def _fake_run(cmd, *, what, **kwargs):
        if "venv" in cmd:
            staged = Path(cmd[-1])
            assert staged != venv_dir
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").write_text("")

    monkeypatch.setattr(ovrtx, "_run_logged", _fake_run)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    ovrtx._provision_venv(venv_dir)

    assert not leftover.exists(), (
        "the previous wheel's post-install files survived into the new tree"
    )
    _assert_ready_marker(
        venv_dir / ovrtx._READY_MARKER_NAME, ovrtx, runtime_lock
    )
    assert not (venv_dir / ovrtx._PROVISIONING_MARKER_NAME).exists(), (
        "the provisioning sentinel must be cleared once the venv is ready"
    )


def test_a_venv_usd_cli_did_not_provision_is_never_deleted(tmp_path, monkeypatch) -> None:
    """Replacing a stale venv must not reach a runtime we do not own.

    WU_OVRTX_VENV_DIR is also world_understanding's managed runtime path, and
    the CAD docs tell operators to point both at it. Deleting a tree neither
    provisioner marked would destroy whatever is there, and if it was wu's the
    two would then rebuild a ~2.5 GB venv over each other.
    """
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    # deliberately unmarked by either provisioner: a tree someone built by
    # hand. The co-managed case is covered separately.

    runtime_lock = tmp_path / "lock.toml"
    # the rebuildability guard reads this, so it has to exist and be
    # satisfiable; an unreadable lock is refused by design
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])

    with _pytest.raises(RuntimeError, match="not a runtime usd-cli created"):
        ovrtx._provision_venv(venv_dir)

    assert (venv_dir / "bin" / "python").exists(), (
        "refusing to replace an unowned runtime must leave it intact"
    )
    assert not (venv_dir / ovrtx._READY_MARKER_NAME).exists()


def test_a_venv_provisioned_for_an_older_pin_is_not_reused(tmp_path) -> None:
    """A cached runtime must be re-provisioned when the pin moves.

    Readiness used to be `marker.exists()`, so a machine that already had a
    0.3 venv kept rendering on 0.3 forever after a bump: the marker records
    the pin it was written for, but nothing compared it. Every render then
    silently used the superseded runtime.
    """
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    marker = venv_dir / ".usd-cli-ovrtx-ready"

    marker.write_text("ovrtx==0.0.0.000000")
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "a venv provisioned for the previous pin must not satisfy this one"
    )

    marker.write_text(ovrtx.OVRTX_PIN)
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "the legacy ovrtx-only marker must not attest ovstage and Warp"
    )

    marker.write_text(
        json.dumps(
            {
                "packages": ovrtx._QUALIFIED_RUNTIME_PROFILE,
                "schema_version": "usd-cli.ovrtx-runtime-ready.v2",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "the profile-only v2 marker must not attest the selected lock"
    )

    marker.write_text(
        ovrtx._readiness_marker_body(ovrtx._ovrtx_runtime_lock()))
    assert ovrtx._venv_matches_pin(venv_dir)

    marker.unlink()
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "a half-provisioned venv must still be repaired, not trusted"
    )


def test_dockerfile_pins_the_locked_ovrtx_artifact_and_companions() -> None:
    """The image must not resolve any runtime package by version alone.

    The marker records the whole lock identity, so all three packages must come
    from exact locked artifacts for every supported Python/architecture pair.
    """
    from usd_core.render import ovrtx

    lock = tomllib.loads(ovrtx.OVRTX_RUNTIME_LOCK.read_text())
    packages = {p["name"]: p for p in lock["packages"]}
    dockerfile = (SERVICE_DIR / "Dockerfile").read_text()

    assert "TARGETARCH" in dockerfile, (
        "the ovrtx wheel is no longer selected per architecture"
    )
    assert "PYTHON_MINOR" in dockerfile, (
        "numpy and pillow are no longer selected per supported Python minor"
    )

    expected_wheels = []
    for name in ("ovrtx", "ovstage", "warp-lang", "numpy", "pillow"):
        wheels = packages[name].get("wheels") or []
        if name in {"ovrtx", "ovstage", "warp-lang"}:
            selected = [
                wheel
                for wheel in wheels
                if "manylinux" in wheel["url"]
                and ("x86_64" in wheel["url"] or "aarch64" in wheel["url"])
            ]
        else:
            selected = [
                wheel
                for wheel in wheels
                if ("-cp311-cp311-" in wheel["url"] or "-cp312-cp312-" in wheel["url"])
                and "manylinux" in wheel["url"]
                and ("x86_64" in wheel["url"] or "aarch64" in wheel["url"])
            ]
        expected_count = 2 if name in {"ovrtx", "ovstage", "warp-lang"} else 4
        assert len(selected) == expected_count, (name, [wheel["url"] for wheel in selected])
        expected_wheels.extend((name, wheel) for wheel in selected)

    for name, wheel in expected_wheels:
        pinned = f"{wheel['url']}#sha256={wheel['hashes']['sha256']}"
        assert pinned in dockerfile, (
            f"Dockerfile does not install the locked {name} artifact {pinned}"
        )

    assert '"numpy==' not in dockerfile
    assert '"pillow==' not in dockerfile

    assert "--extra-index-url https://pypi.nvidia.com ovrtx==" not in dockerfile, (
        "the unpinned index install is back; it resolves ovrtx by version only"
    )

def test_dockerfile_marks_the_baked_venv_with_the_full_profile() -> None:
    """The baked venv is used only when its marker records the full profile.

    The Dockerfile imports the marker value from usd_core instead of carrying a
    second hand-maintained copy of the three versions.
    """
    dockerfile = (SERVICE_DIR / "Dockerfile").read_text()
    assert (
        "from usd_core.render.ovrtx import OVRTX_RUNTIME_LOCK, "
        "_readiness_marker_body" in dockerfile
    )
    assert "_readiness_marker_body(OVRTX_RUNTIME_LOCK)" in dockerfile
    assert "> /opt/ovrtx_venv/.usd-cli-ovrtx-ready" in dockerfile, (
        "the canonical readiness record is not written to the baked venv"
    )
    assert "_READY_MARKER_VALUE" not in dockerfile
    assert "sha256sum" not in dockerfile


def test_dockerfile_selected_wheels_belong_to_the_runtime_locks() -> None:
    """Every wheel baked into either daemon venv is admitted by its lock."""
    from usd_core import physics_runtime
    from usd_core.render import ovrtx

    dockerfile = (SERVICE_DIR / "Dockerfile").read_text(encoding="utf-8")
    selected = set(
        re.findall(
            r"https?://[^\"\s;]+#sha256=[0-9a-f]{64}",
            dockerfile,
        )
    )
    lock_paths = {
        ovrtx.OVRTX_RUNTIME_LOCK,
        *physics_runtime._PACKAGED_RUNTIME_LOCKS.values(),
    }
    admitted: set[str] = set()
    for lock_path in lock_paths:
        lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
        for package in lock["packages"]:
            artifacts = [package.get("archive"), *package.get("wheels", [])]
            admitted.update(
                f"{artifact['url']}#sha256={artifact['hashes']['sha256']}"
                for artifact in artifacts
                if artifact is not None
            )

    assert selected
    assert selected <= admitted
    assert "OVRTX wheel references absent from runtime lock" in dockerfile


def test_ovrtx_runtime_lock_is_checked_in_and_fully_hashed():
    import re

    from usd_core.render import ovrtx

    lock_path = ovrtx.OVRTX_RUNTIME_LOCK
    assert lock_path.is_file(), f"missing checked-in lock: {lock_path}"
    lock = tomllib.loads(lock_path.read_text())
    assert lock["requires-python"] == ">=3.11"
    packages = {p["name"]: p for p in lock["packages"]}
    # the whole daemon runtime, nothing floating
    assert set(packages) == {"ovrtx", "ovstage", "warp-lang", "numpy", "pillow"}
    assert {
        name: packages[name]["version"]
        for name in ovrtx._QUALIFIED_RUNTIME_PROFILE
    } == ovrtx._QUALIFIED_RUNTIME_PROFILE
    for name, pkg in packages.items():
        artifacts = list(pkg.get("wheels", []))
        if "sdist" in pkg:
            artifacts.append(pkg["sdist"])
        if "archive" in pkg:
            artifacts.append(pkg["archive"])
        assert artifacts, f"{name}: no artifacts in lock"
        for art in artifacts:
            # This file is excluded from the release gate's secrets scan, so
            # enforce its shape here: real sha256 digests, artifacts only from
            # the two known indexes, nothing credential-shaped in the URLs.
            sha = art.get("hashes", {}).get("sha256", "")
            assert re.fullmatch(r"[0-9a-f]{64}", sha), f"{name}: bad sha256 {sha!r}"
            url = art.get("url", "")
            assert url.startswith(OVRTX_LOCK_ARTIFACT_HOSTS), f"{name}: {url}"
            assert "@" not in url, f"{name}: credential-shaped URL {url}"


def test_provisioning_installs_from_the_lock_with_hashes_required(tmp_path, monkeypatch):
    from usd_core.render import ovrtx

    commands: list[list[str]] = []
    installed_lock_bytes: list[bytes] = []

    def record_command(cmd, *, what, **kw):
        commands.append(list(cmd))
        if what == "ovrtx install (venv)":
            staged = Path(cmd[-1])
            assert staged != venv_dir
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
        if what == "ovrtx install (locked runtime)":
            installed_lock_bytes.append(Path(cmd[cmd.index("-r") + 1]).read_bytes())

    monkeypatch.setattr(ovrtx, "_run_logged", record_command)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv-stub"])
    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    ovrtx._provision_venv(venv_dir)

    assert commands[0][:3] == [sys.executable, "-m", "venv"]
    install = commands[1]
    assert install[:3] == ["uv-stub", "pip", "install"]
    for flag in ("--require-hashes", "--no-deps", "--no-config", "--no-sources"):
        assert flag in install, f"{flag} missing from {install}"
    installed_from = Path(install[install.index("-r") + 1])
    assert installed_from != lock
    assert installed_lock_bytes == [lock.read_bytes()]
    assert not installed_from.exists()
    # no loose requirement names — everything comes from the reviewed lock
    assert "numpy" not in install and "pillow" not in install
    assert ovrtx.OVRTX_PIN not in install
    _assert_ready_marker(
        venv_dir / ".usd-cli-ovrtx-ready", ovrtx, lock
    )


def test_provisioning_fails_closed_without_uv_or_lock(tmp_path, monkeypatch):
    import importlib.util
    import shutil

    from usd_core.render import ovrtx

    monkeypatch.delenv("USD_CLI_UV_EXECUTABLE", raising=False)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uv is required"):
        ovrtx._uv_command()

    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(tmp_path / "nope.toml"))
    with pytest.raises(RuntimeError, match="WU_OVRTX_RUNTIME_LOCK"):
        ovrtx._ovrtx_runtime_lock()

    override = tmp_path / "custom.toml"
    override.write_text(ovrtx.OVRTX_RUNTIME_LOCK.read_text())
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(override))
    assert ovrtx._ovrtx_runtime_lock() == override


def test_no_unpinned_pip_install_remains_in_provisioning():
    """Source-level guard (mirrors wu's install-guidance tests): the provisioning
    path must never regress to a bare `pip install <names>`."""
    from usd_core.render import ovrtx

    source = Path(ovrtx.__file__).read_text()
    assert '"numpy", "pillow"' not in source
    assert '"--require-hashes"' in source


def _release_gate_secrets_exclusion() -> str:
    """Return the exact exclusion used by the release scan entry point."""
    script = REPO_ROOT / "scripts" / "scan_component_secrets.py"
    spec = importlib.util.spec_from_file_location("usd_cli_release_secrets", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DETECT_SECRETS_EXCLUSION


LOCK_REL = "src/usd_core/render/pylock.ovrtx-runtime.toml"


def _staged_scan_layout(tmp_path):
    """The gate's scan layout: extracted-sdist root + the real lock + a planted
    secret in another shipped file (the control proving the scan still bites)."""
    import shutil

    from usd_core.render import ovrtx

    staged = tmp_path / "artifact-sources"
    lock_dst = staged / "usd-cli" / LOCK_REL
    lock_dst.parent.mkdir(parents=True)
    shutil.copyfile(ovrtx.OVRTX_RUNTIME_LOCK, lock_dst)
    planted = staged / "usd-cli" / "src" / "leaky_config.py"
    planted.write_text(
        'aws_secret = "9f86d081884c7d659a2feaa0c55ad015'
        'a3bf4f1b2b0b822cd15d6c15b0f00a08"\n')
    return staged


def _detect_secrets_scan(staged, *extra_args):
    import json
    import subprocess

    scan = subprocess.run(
        [sys.executable, "-m", "detect_secrets", "scan", "--all-files", *extra_args],
        cwd=staged, capture_output=True, text=True, check=True, timeout=120)
    return json.loads(scan.stdout)["results"]


def test_release_gate_secrets_exclusion_skips_only_the_lock(tmp_path):
    """Mirror the release gate's own detect-secrets invocation (with the
    exclusion flag) over the extracted-sdist layout: the pylock's sha256
    digests must not fail the scan, while a secret planted in any other
    shipped file must still be caught."""
    import re

    pytest.importorskip("detect_secrets")
    exclusion = _release_gate_secrets_exclusion()
    # detect-secrets 1.5.0 applies exclusions with re.search
    # (detect_secrets/filters/regex.py::should_exclude_file), but the regex
    # must ALSO hold under re.match anchoring so a scanner-version change
    # cannot silently re-open this: assert the stricter re.match for both the
    # repo-rooted path and the vendored `usd-cli/` prefix.
    for path in (LOCK_REL, f"usd-cli/{LOCK_REL}"):
        assert re.match(exclusion, path), (exclusion, path)
        assert re.search(exclusion, path), (exclusion, path)
    # ... and it names exactly that one file
    assert not re.search(exclusion, "usd-cli/src/usd_core/render/other.toml")
    assert not re.search(exclusion, f"usd-cli/{LOCK_REL}.orig")

    staged = _staged_scan_layout(tmp_path)
    results = _detect_secrets_scan(staged, "--exclude-files", exclusion)
    assert not any("pylock" in path for path in results), results
    assert any("leaky_config" in path for path in results), \
        "the planted secret was not detected — the scan lost its teeth"


def test_lock_passes_secrets_scan_without_any_exclusion(tmp_path):
    """The invocation that actually failed in world-understanding CI: its
    `usd-cli source gate` mirror workflow scans the vendored tree with a plain
    `detect-secrets scan --all-files` and NO exclusion flags, so the lock must
    defend itself — the in-file `# pragma: allowlist secret` markers on the
    digest lines (re-applied by scripts/gen-ovrtx-lock.sh on every regen).
    The planted secret proves the flagless scan still bites."""
    pytest.importorskip("detect_secrets")
    staged = _staged_scan_layout(tmp_path)
    results = _detect_secrets_scan(staged)
    assert not any("pylock" in path for path in results), results
    assert any("leaky_config" in path for path in results), \
        "the planted secret was not detected — the scan lost its teeth"


# ── P1 (review round 4): render input dropped session-layer edits ────────────────
#
# `appearance clear` deliberately leaves the edit target on the SESSION layer, so
# every accepted material edit after it lives there. For a staged non-USDZ input
# with a writable parent, prepare_render_input exported only the ROOT layer —
# every supported backend (remote or local OVRTX) then reopened the
# pre-clear asset: final renders showed the OLD materials while save --flatten
# wrote the new ones. Wrong final evidence, silently.


def _material_scene(tmp_path, name="scene.usda"):
    """An on-disk asset with a mesh bound to material Old (writable parent)."""
    from pxr import Usd, UsdGeom, UsdShade

    path = tmp_path / name
    stage = Usd.Stage.CreateNew(str(path))
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    cube = UsdGeom.Cube.Define(stage, "/World/Body")
    cube.CreateDisplayColorAttr([(1.0, 0.0, 0.0)])
    old = UsdShade.Material.Define(stage, "/World/Looks/Old")
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(old)
    stage.GetRootLayer().Save()
    return path


def _clear_and_rebind_in_session(scene_path):
    """The real post-`appearance clear` state: clear (session layer, edit target
    left on session) + one accepted material bind authored through that target."""
    from pxr import Usd, UsdShade
    from usd_core.appearance import clear_appearance

    stage = Usd.Stage.Open(str(scene_path))
    clear_appearance(stage)
    assert stage.GetEditTarget().GetLayer() == stage.GetSessionLayer()
    new = UsdShade.Material.Define(stage, "/World/Looks/New")
    api = UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath("/World/Body"))
    assert api.Bind(new)
    return stage


def _bound_material_path(usd_path):
    from pxr import Usd, UsdShade

    stage = Usd.Stage.Open(str(usd_path))
    mesh = stage.GetPrimAtPath("/World/Body")
    material = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
    return material.GetPath().pathString if material else None


def test_render_input_carries_session_layer_appearance_edits(tmp_path):
    """The P1 itself: after a real `appearance clear` + session-layer rebind, the
    prepared render input must compose the NEW binding, not the pre-clear root."""
    from usd_core.render.base import prepare_render_input

    stage = _clear_and_rebind_in_session(_material_scene(tmp_path))
    prepared, is_temp = prepare_render_input(stage, tmp_path / "out")
    assert is_temp and prepared.parent == tmp_path  # the staged writable-parent path
    assert _bound_material_path(prepared) == "/World/Looks/New", \
        "render input reopened the pre-clear asset — wrong final evidence"


def test_render_input_session_merge_preserves_relative_references(tmp_path):
    """The root-layer export exists to keep relative references resolving; the
    session-aware merge must preserve that — arcs stay arcs (nothing inlined) and
    a sibling-file reference still composes from the merged file's location."""
    from pxr import Sdf, Usd, UsdGeom
    from usd_core.render.base import prepare_render_input

    part = Usd.Stage.CreateNew(str(tmp_path / "part.usda"))
    part_root = UsdGeom.Xform.Define(part, "/Part")
    UsdGeom.Sphere.Define(part, "/Part/Ball")
    part.SetDefaultPrim(part_root.GetPrim())
    part.GetRootLayer().Save()

    scene_path = _material_scene(tmp_path)
    scene = Usd.Stage.Open(str(scene_path))
    ref_prim = scene.DefinePrim("/World/Referenced")
    ref_prim.GetReferences().AddReference("./part.usda")  # relative, sibling file
    scene.GetRootLayer().Save()

    stage = _clear_and_rebind_in_session(scene_path)
    prepared, _ = prepare_render_input(stage, tmp_path / "out")

    merged = Sdf.Layer.FindOrOpen(str(prepared))
    spec = merged.GetPrimAtPath("/World/Referenced")
    assert spec is not None and spec.hasReferences, "reference arc was inlined/lost"
    composed = Usd.Stage.Open(str(prepared))
    assert composed.GetPrimAtPath("/World/Referenced/Ball").IsValid(), \
        "relative reference no longer resolves from the prepared file"
    assert _bound_material_path(prepared) == "/World/Looks/New"


def test_render_input_without_session_edits_is_the_plain_root_export(tmp_path):
    """No session opinions → exactly the pre-existing path: root layer exported
    next to the original, root-layer in-memory edits (the managed camera) carried."""
    from pxr import Usd, UsdGeom
    from usd_core.render.base import prepare_render_input

    stage = Usd.Stage.Open(str(_material_scene(tmp_path)))
    UsdGeom.Camera.Define(stage, "/World/render_cam")  # root-layer in-memory edit
    prepared, is_temp = prepare_render_input(stage, tmp_path / "out")
    assert is_temp and prepared.parent == tmp_path
    reopened = Usd.Stage.Open(str(prepared))
    assert reopened.GetPrimAtPath("/World/render_cam").IsValid()
    assert _bound_material_path(prepared) == "/World/Looks/Old"


def test_render_input_usdz_composes_session_edits(tmp_path):
    """USDZ path: untouched packages still go straight through (immutable root,
    nothing to carry); with session opinions the package cannot express them, so
    the input falls back to the flattened compose — which must show the session
    binding."""
    from pxr import Usd, UsdShade, UsdUtils
    from usd_core.render.base import prepare_render_input

    scene_path = _material_scene(tmp_path)
    usdz = tmp_path / "scene.usdz"
    assert UsdUtils.CreateNewUsdzPackage(str(scene_path), str(usdz))

    untouched = Usd.Stage.Open(str(usdz))
    prepared, is_temp = prepare_render_input(untouched, tmp_path / "out")
    assert prepared == usdz and not is_temp  # unchanged fast path

    stage = Usd.Stage.Open(str(usdz))
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    new = UsdShade.Material.Define(stage, "/World/Looks/New")
    assert UsdShade.MaterialBindingAPI.Apply(
        stage.GetPrimAtPath("/World/Body")).Bind(new)
    prepared, _ = prepare_render_input(stage, tmp_path / "out")
    assert prepared != usdz
    assert _bound_material_path(prepared) == "/World/Looks/New"


def test_render_input_readonly_parent_flatten_composes_session_edits(tmp_path, monkeypatch):
    """The read-only-parent fallback flattens the full stage, session included —
    unchanged, and still correct for session edits."""
    from usd_core.render import base as render_base
    from usd_core.render.base import prepare_render_input

    stage = _clear_and_rebind_in_session(_material_scene(tmp_path))
    # base.py's writability probe is plain os.access; forcing it False routes the
    # writable tmp dir through the read-only-parent fallback under test
    monkeypatch.setattr(render_base.os, "access", lambda path, mode: False)
    prepared, is_temp = prepare_render_input(stage, tmp_path / "out")
    assert prepared.parent == tmp_path / "out" and not is_temp
    assert _bound_material_path(prepared) == "/World/Looks/New"


def test_lock_digest_lines_carry_scanner_pragmas():
    """Every sha256 digest line in the lock carries the allowlist pragma (a
    bare `uv pip compile` regen drops them — scripts/gen-ovrtx-lock.sh is the
    supported regen path), and nothing else in the lock is pragma'd, so the
    allowlist stays as narrow as the digests themselves."""
    import re

    from usd_core.render import ovrtx

    for line in ovrtx.OVRTX_RUNTIME_LOCK.read_text().splitlines():
        has_digest = re.search(r'sha256 = "[0-9a-f]{64}"', line)
        has_pragma = line.rstrip().endswith("# pragma: allowlist secret")
        assert bool(has_digest) == bool(has_pragma), line
    regen = (REPO_ROOT / "scripts" / "gen-ovrtx-lock.sh").read_text()
    assert "pragma: allowlist secret" in regen
    assert "uv pip compile" in regen
    normalized_regen = regen.replace("\\\n", " ")
    assert "--python-version 3.11 --universal" in normalized_regen


def test_ordinary_requires_python_specs_do_not_block_provisioning(tmp_path):
    """The guard exists to stop a destructive replace, not to stop provisioning.

    An operator lock (WU_OVRTX_RUNTIME_LOCK) carrying a perfectly ordinary spec
    -- >=3.12.0, >=3.9,<4, ~=3.12, >=3 -- must be evaluated, not rejected as
    unreadable. Failing closed on those blocks the install outright.
    """
    from usd_core.render import ovrtx

    major, minor, micro = sys.version_info[:3]

    def verdict(spec):
        lock = tmp_path / "pylock.toml"
        lock.write_text('requires-python = "%s"' % spec, encoding="utf-8")
        reason = ovrtx._interpreter_satisfies_lock(lock)
        if reason is None:
            return "allow"
        return "refuse" if "requires Python" in reason else "uneval"

    # satisfied by this interpreter, whatever it is
    for spec in (
        ">=%d.%d.%d" % (major, minor, micro),
        ">=%d.%d,<%d" % (major, minor, major + 1),
        "~=%d.%d" % (major, minor),
        ">=%d" % major,
        "==%d.%d.*" % (major, minor),
        "<=%d.%d.%d" % (major, minor, micro),
    ):
        assert verdict(spec) == "allow", spec

    # genuinely unsatisfiable: refused, and reported as a version mismatch
    for spec in (
        ">=%d.%d" % (major, minor + 50),
        "~=%d.%d" % (major, minor + 50),
        "==%d.%d.*" % (major, minor + 50),
    ):
        assert verdict(spec) == "refuse", spec

    # still fails closed on what it cannot evaluate
    for spec in ("~=%d" % major, ">=%d.%d.*" % (major, minor), "gibberish"):
        assert verdict(spec) == "uneval", spec


def test_the_refusal_never_tells_an_operator_to_clear_wus_runtime(tmp_path, monkeypatch):
    """WU_OVRTX_VENV_DIR may point at world_understanding's managed runtime --
    the CAD docs tell operators to do exactly that. The "auto-provision is not
    allowed" path used to diagnose "no provisioned venv" and print a recipe
    starting `python -m venv --clear`, which erases ~2.5 GB and the marker that
    records who owns it."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "shared_ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ".wu-managed-ovrtx-venv").write_text("ovrtx==0.3.0.312915")
    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    message = str(caught.value)
    # the destructive recipe, not the words: the message says "Do not --clear it"
    assert "-m venv --clear" not in message, message
    assert ".wu-managed-ovrtx-venv" in message
    assert "render_ovrtx --provision-only" in message
    assert "no provisioned venv" not in message
    assert all(
        pin in message for pin in (ovrtx.OVRTX_PIN, ovrtx.OVSTAGE_PIN, ovrtx.WARP_PIN)
    )


def test_partial_backup_cleanup_is_reclaimed_on_the_next_retry(tmp_path, monkeypatch):
    """A partial rmtree may erase in-tree markers before it fails.

    The external ownership sidecar must keep that old runtime recognizable so
    the next locked provisioning run can finish reclaiming it.
    """
    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "lib").mkdir(parents=True)
    (venv_dir / "lib" / "stale.so").write_text("from the previous pin")
    (venv_dir / ".usd-cli-ovrtx-ready").write_text("ovrtx==0.3.0.312915")

    real_rmtree = _shutil.rmtree

    def rmtree_that_cannot_unlink_the_root(target, *args, **kwargs):
        for child in Path(target).iterdir():
            if child.is_dir():
                real_rmtree(child)
            else:
                child.unlink()
        raise OSError(16, "Device or resource busy")

    monkeypatch.setattr(ovrtx.shutil, "rmtree", rmtree_that_cannot_unlink_the_root)

    def fake_run(cmd, *, what="", **kw):
        if not what.endswith("(venv)"):
            return
        staged = Path(cmd[-1])
        py = staged / "bin" / "python"
        py.parent.mkdir(parents=True, exist_ok=True)
        py.touch()

    monkeypatch.setattr(ovrtx, "_run_logged", fake_run)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv-stub"])

    ovrtx._provision_venv(venv_dir)

    _assert_ready_marker(
        venv_dir / ".usd-cli-ovrtx-ready", ovrtx, lock
    )
    assert not (venv_dir / "lib" / "stale.so").exists()
    backups = [
        path for path in tmp_path.glob(".ovrtx_venv.previous-*")
        if path.is_dir()
    ]
    assert len(backups) == 1
    assert not (backups[0] / ovrtx._READY_MARKER_NAME).exists()
    assert ovrtx._sibling_owner_marker(backups[0]).exists()

    monkeypatch.setattr(ovrtx.shutil, "rmtree", real_rmtree)
    ovrtx._recover_interrupted_venv_swap(venv_dir)

    assert not list(tmp_path.glob(".ovrtx_venv.previous-*"))
    assert not list(
        tmp_path.glob(
            f".ovrtx_venv.previous-*{ovrtx._SIBLING_OWNER_SUFFIX}"
        )
    )


def test_no_clear_recipe_is_offered_for_a_directory_we_do_not_own(tmp_path, monkeypatch):
    """A typo'd WU_OVRTX_VENV_DIR can resolve to $HOME. The remediation used to
    embed `python -m venv --clear <that directory>`, so copying it erased
    unrelated data. Offer the recipe only for a tree usd-cli owns or an empty
    one; anything else gets pointed at a fresh directory."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    # uv is not in usd-cli's dev extra, and the recipe is only
    # offered when _uv_command() resolves. Without this the CI
    # gate sees 'no recipe is offered: uv is required...'.
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    # The recipe is withheld when THIS interpreter cannot install the selected
    # lock; use a satisfiable fixture here so these cover ownership and quoting.
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)

    home_by_typo = tmp_path / "home"
    (home_by_typo / "Documents").mkdir(parents=True)
    (home_by_typo / "taxes.pdf").write_text("not ours to erase")
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(home_by_typo, auto_install=False)
    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "fresh directory" in message

    # A stale tree we own points at the failure-safe automated replacement,
    # never a copy-paste command that clears it first.
    ours = tmp_path / "ovrtx_venv"
    ours.mkdir()
    (ours / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(ours, auto_install=False)
    owned_message = str(caught.value)
    assert "venv --clear" not in owned_message
    assert "WU_OVRTX_AUTO_PROVISION=1" in owned_message
    assert "sibling venv" in owned_message

    # An absent directory still gets a non-destructive bootstrap recipe.
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(tmp_path / "not_created_yet", auto_install=False)
    fresh_message = str(caught.value)
    assert "venv --clear" not in fresh_message
    assert " -m venv " in fresh_message


def test_a_lock_that_would_not_produce_the_pin_never_authorises_a_delete(tmp_path, monkeypatch):
    """WU_OVRTX_RUNTIME_LOCK is an operator override, so the selected lock is
    not necessarily the one usd-cli ships. Checking only requires-python let a
    stale lock delete the runtime, install the ovrtx *it* names, and get stamped
    with the current pin -- the staleness the readiness marker exists to stop."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    survivor = venv_dir / "bin" / "python"
    survivor.write_text("the working runtime")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text(
        "ovrtx==0.0.0.000000")  # stale, so a replace is wanted

    stale = tmp_path / "stale-pylock.toml"
    stale.write_text(
        'requires-python = ">=3.0"' + chr(10)
        + "[[packages]]" + chr(10)
        + 'name = "ovrtx"' + chr(10)
        + 'version = "0.3.0.312915"' + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: stale)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())

    with _pytest.raises(RuntimeError, match="pins ovrtx"):
        ovrtx._provision_venv(venv_dir)

    assert survivor.read_text() == "the working runtime"
    assert (venv_dir / ovrtx._READY_MARKER_NAME).read_text() == (
        "ovrtx==0.0.0.000000"
    )

    # a lock with no ovrtx at all is just as unusable
    empty = tmp_path / "no-ovrtx.toml"
    empty.write_text('requires-python = ">=3.0"' + chr(10), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: empty)
    with _pytest.raises(RuntimeError, match="lists no packages"):
        ovrtx._provision_venv(venv_dir)
    assert survivor.read_text() == "the working runtime"


def test_a_superseded_wu_runtime_is_not_reused(tmp_path, monkeypatch):
    """The version-mismatch branch of _shared_runtime_matching_pin had no
    guardrail: the positive test stubbed the probe to the pinned version, and
    the other wu test built the tree without bin/python, so the comparison was
    never reached. A bug returning the python regardless of version would pass
    the suite while reintroducing exactly the staleness this PR closes."""
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "wu_ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    (venv_dir / ovrtx._WU_MANAGED_MARKER_NAME).write_text("owned by wu")

    reported = {
        **ovrtx._QUALIFIED_RUNTIME_PROFILE,
        "ovrtx": "0.3.0.312915",
    }

    class _Probe:
        returncode = 0

        @property
        def stdout(self):
            return json.dumps(reported, sort_keys=True) + "\n"

    monkeypatch.setattr(ovrtx.subprocess, "run", lambda *a, **k: _Probe())

    assert ovrtx._shared_runtime_matching_pin(venv_dir) is None

    reported["ovrtx"] = ovrtx._pinned_ovrtx_version()
    assert ovrtx._shared_runtime_matching_pin(venv_dir) == str(
        venv_dir / "bin" / "python")


def test_the_printed_recipe_names_the_lock_provisioning_would_use(tmp_path, monkeypatch):
    """WU_OVRTX_RUNTIME_LOCK overrides the shipped lock. A recipe that names the
    shipped one anyway installs a different set of artifacts than the automated
    path would -- and then stamps OVRTX_PIN on the result."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    # uv is not in usd-cli's dev extra, and the recipe is only
    # offered when _uv_command() resolves. Without this the CI
    # gate sees 'no recipe is offered: uv is required...'.
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    override = tmp_path / "operator-pylock.toml"
    override.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(override))

    venv_dir = tmp_path / "fresh_ovrtx_venv"

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    recipe = [l for l in str(caught.value).splitlines() if " -r " in l]
    assert recipe, str(caught.value)
    assert str(override) in recipe[0], recipe[0]
    assert str(ovrtx.OVRTX_RUNTIME_LOCK) not in recipe[0], recipe[0]


def test_a_lock_for_another_architecture_never_authorises_a_delete(tmp_path, monkeypatch):
    """The shipped lock is compiled x86_64-manylinux and carries one x86_64
    wheel with no sdist. On aarch64 both halves of the guard used to pass, the
    rmtree ran, and only then did uv discover there was no installable ovrtx --
    exactly the "leave the host with no renderer" outcome it exists to stop."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    survivor = venv_dir / "bin" / "python"
    survivor.write_text("the only runtime on this aarch64 host")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    version = ovrtx._pinned_ovrtx_version()
    foreign = tmp_path / "x86-only-pylock.toml"
    foreign.write_text(
        _satisfiable_lock_text(version)
        + 'wheels = [{ url = "https://pypi.nvidia.com/ovrtx/ovrtx-%s-py3-none-manylinux_2_35_x86_64.whl" }]'
        % version + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: foreign)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())
    # pin both axes: the check is OS as well as architecture now, so the
    # assertions have to describe a host rather than inherit this one
    monkeypatch.setattr(ovrtx.sys, "platform", "linux")
    monkeypatch.setattr(ovrtx.platform, "machine", lambda: "aarch64")

    with _pytest.raises(RuntimeError, match="no ovrtx wheel for this machine"):
        ovrtx._provision_venv(venv_dir)

    assert survivor.read_text() == "the only runtime on this aarch64 host"

    # a wheel this machine can run is still accepted
    monkeypatch.setattr(ovrtx.platform, "machine", lambda: "x86_64")
    assert ovrtx._lock_provides_pin(foreign) is None

    # same architecture, wrong OS: win_amd64 shares the amd64 token with
    # manylinux_x86_64, so architecture alone let it through and the rmtree
    # ran before uv rejected the wheel
    windows_only = tmp_path / "win-only-pylock.toml"
    windows_only.write_text(
        _satisfiable_lock_text(version)
        + 'wheels = [{ url = "https://pypi.nvidia.com/ovrtx/ovrtx-%s-py3-none-win_amd64.whl" }]'
        % version + chr(10),
        encoding="utf-8",
    )
    assert ovrtx._lock_provides_pin(windows_only) is not None


def test_world_understanding_lock_satisfies_usd_cli_on_supported_linux_arches(
    monkeypatch,
):
    """WU's documented shared runtime must pass usd-cli's exact-profile gate."""
    from usd_core.render import ovrtx

    graphics = _world_understanding_graphics()
    if graphics is None:
        pytest.skip("world_understanding tree not present (standalone usd-cli)")
    shared_lock = graphics / "pylock.ovrtx-runtime.toml"
    assert shared_lock.is_file()
    monkeypatch.setattr(ovrtx.sys, "platform", "linux")

    for machine in ("x86_64", "aarch64"):
        monkeypatch.setattr(
            ovrtx.platform,
            "machine",
            lambda machine=machine: machine,
        )
        assert ovrtx._lock_provides_pin(shared_lock) is None


def test_every_path_in_the_recipe_is_shell_quoted(tmp_path, monkeypatch):
    """Every path in the non-destructive bootstrap recipe stays one argument."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    # uv is not in usd-cli's dev extra, and the recipe is only
    # offered when _uv_command() resolves. Without this the CI
    # gate sees 'no recipe is offered: uv is required...'.
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    # The recipe is withheld when THIS interpreter cannot install the selected
    # lock; use a satisfiable fixture here so these cover path quoting.
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "my ovrtx venv"

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    create = [
        line for line in str(caught.value).splitlines() if " -m venv " in line
    ][0]
    # the directory must survive shlex.split as ONE argument
    assert str(venv_dir) in _shlex.split(create.rstrip(chr(92)).strip()), create


def test_wus_interrupted_provision_is_not_treated_as_an_unowned_tree(tmp_path, monkeypatch):
    """world_understanding stamps .wu-managed-ovrtx-venv.provisioning before it
    installs and removes it only on success. Recognising just the completed
    marker meant an interrupted wu provision looked unowned: the operator was
    invited to stamp our marker onto wu's half-built tree, and where usd-cli had
    provisioned the same directory before, the rmtree destroyed it."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "shared_ovrtx_venv"
    (venv_dir / "lib").mkdir(parents=True)
    (venv_dir / "lib" / "half_installed.so").write_text("wu was mid-install")
    (venv_dir / ovrtx._WU_PROVISIONING_MARKER_NAME).write_text("wu, in flight")
    # usd-cli provisioned here previously, so `ours` is true and the old code
    # would have gone straight to the rmtree
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    runtime_lock = tmp_path / "lock.toml"
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)
    # the installer is stubbed, so nothing real is installed;
    # these cover provisioning mechanics, not the post-install
    # probe that has its own tests
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version())

    with _pytest.raises(RuntimeError, match="world-understanding managed"):
        ovrtx._provision_venv(venv_dir)

    assert (venv_dir / "lib" / "half_installed.so").exists(), (
        "wu's in-flight tree must survive"
    )


def test_the_probe_does_not_inherit_pythonpath(tmp_path, monkeypatch):
    """The daemon drops PYTHONPATH before launching. A probe that keeps it can
    resolve an injected ovrtx and accept a venv the daemon then cannot import
    from."""
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "wu_ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("")
    (venv_dir / ovrtx._WU_MANAGED_MARKER_NAME).write_text("owned by wu")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "injected"))

    seen = {}

    class _Probe:
        returncode = 0
        stdout = ""

    def record(cmd, **kwargs):
        seen.update(kwargs.get("env") or {})
        seen["_had_env"] = kwargs.get("env") is not None
        return _Probe()

    monkeypatch.setattr(ovrtx.subprocess, "run", record)
    ovrtx._shared_runtime_matching_pin(venv_dir)

    assert seen.get("_had_env"), "the probe must pass an explicit environment"
    assert "PYTHONPATH" not in seen


def test_the_image_fails_closed_on_an_unknown_targetarch():
    """TARGETARCH is a BuildKit automatic ARG; under the legacy builder it
    expands empty, the arm64 branch never fires, and the image bakes the x86_64
    wheel while still writing the readiness marker."""
    dockerfile = (
        REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile"
    ).read_text(encoding="utf-8")
    assert 'case "${TARGETARCH}" in amd64|arm64' in dockerfile
    guard = dockerfile.index('case "${TARGETARCH}"')
    selection = dockerfile.index('if [ "${TARGETARCH}" = "arm64" ]')
    assert guard < selection, "the guard must run before the wheel is chosen"


def test_no_recipe_is_offered_that_would_stamp_a_pin_it_does_not_install(tmp_path, monkeypatch):
    """The recipe ends by writing OVRTX_PIN into the readiness marker, and
    _venv_matches_pin trusts that marker without probing the install. With
    WU_OVRTX_RUNTIME_LOCK still on a complete 0.3 lock, following the printed
    recipe installed 0.3 and recorded 0.4, so every later render silently used
    the superseded renderer -- the fail-closed renderer contract, defeated
    through the error message rather than the code."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    # uv is not in usd-cli's dev extra, and the recipe is only
    # offered when _uv_command() resolves. Without this the CI
    # gate sees 'no recipe is offered: uv is required...'.
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    superseded = tmp_path / "old-0.3-pylock.toml"
    superseded.write_text(_satisfiable_lock_text("0.3.0.312915"), encoding="utf-8")
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(superseded))

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "pins ovrtx '0.3.0.312915'" in message
    assert ovrtx.OVRTX_PIN in message

    # With a lock that does name the pin, the existing owned runtime points at
    # automated side-by-side replacement rather than a destructive recipe.
    good = tmp_path / "good-pylock.toml"
    good.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(good))
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)
    assert "venv --clear" not in str(caught.value)
    assert "WU_OVRTX_AUTO_PROVISION=1" in str(caught.value)

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(tmp_path / "fresh_venv", auto_install=False)
    assert " -m venv " in str(caught.value)


def test_the_ambient_probe_does_not_inherit_pythonpath(tmp_path, monkeypatch):
    """_shared_runtime_matching_pin drops PYTHONPATH because the daemon does;
    the ambient probe one function earlier did not, so an ovrtx reachable only
    through PYTHONPATH passed the pin check and the daemon then could not
    import it."""
    from usd_core.render import ovrtx

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "injected"))
    seen = {}

    class _Probe:
        returncode = 1
        stdout = ""
        stderr = ""

    def record(cmd, **kwargs):
        seen["env"] = kwargs.get("env")
        return _Probe()

    monkeypatch.setattr(ovrtx.subprocess, "run", record)
    ovrtx._env_ovrtx_python()

    assert seen["env"] is not None, "the probe must pass an explicit environment"
    assert "PYTHONPATH" not in seen["env"]


def test_the_marker_records_what_was_installed_not_what_the_lock_claimed(tmp_path, monkeypatch):
    """_venv_matches_pin trusts .usd-cli-ovrtx-ready without probing, so the
    marker is the whole readiness contract. _lock_provides_pin reads the lock's
    declared version, but a lock can name one ovrtx and carry a wheel URL for
    another -- the hashes pin the artifact, not the agreement between the two.
    Stamping an unverified tree is how a superseded renderer goes unnoticed."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"

    def fake_install(cmd, *, what="", **kw):
        if what.endswith("(venv)"):
            py = Path(cmd[-1]) / "bin" / "python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.touch()

    monkeypatch.setattr(ovrtx, "_run_logged", fake_install)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv-stub"])
    # the lock said 0.4.1; the artifact behind it was 0.3
    monkeypatch.setattr(ovrtx, "_installed_ovrtx_version", lambda py: "0.3.0.312915")

    with _pytest.raises(RuntimeError, match="did not provide the qualified"):
        ovrtx._provision_venv(venv_dir)

    assert not (venv_dir / ovrtx._READY_MARKER_NAME).exists(), (
        "an unverified tree must not be recorded as ready"
    )

    # an install missing any worker dependency is refused the same way
    monkeypatch.setattr(ovrtx, "_installed_ovrtx_version", lambda py: None)
    with _pytest.raises(RuntimeError, match="did not provide the qualified"):
        ovrtx._provision_venv(venv_dir)
    assert not (venv_dir / ovrtx._READY_MARKER_NAME).exists()

    # and the matching install is stamped
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version", lambda py: ovrtx._pinned_ovrtx_version())
    ovrtx._provision_venv(venv_dir)
    _assert_ready_marker(
        venv_dir / ovrtx._READY_MARKER_NAME, ovrtx, lock
    )


def test_no_recipe_is_offered_this_interpreter_could_not_run(tmp_path, monkeypatch):
    """_provision_venv refuses to delete a runtime this interpreter cannot
    rebuild. The printed recipe has to be symmetric: it must not offer a venv
    creation/install sequence that the selected lock will reject."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    unsupported = tmp_path / "future-pylock.toml"
    unsupported.write_text(
        'requires-python = ">=%d.%d"' % (sys.version_info[0], sys.version_info[1] + 50)
        + chr(10) + "[[packages]]" + chr(10) + 'name = "ovrtx"' + chr(10)
        + 'version = "%s"' % ovrtx._pinned_ovrtx_version() + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(unsupported))

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "requires Python" in message


def test_usd_cli_and_wu_agree_on_the_ownership_markers() -> None:
    """The whole "never delete world_understanding's runtime" protection keys on
    two string literals duplicated across the trees. If wu renames a marker,
    _shared_runtime_matching_pin stops matching and _provision_venv reaches the
    `ours` branch and rmtree's wu's managed runtime -- with the suite green.
    This mirrors the guard the pin already has.
    """
    from usd_core.render import ovrtx

    wu = (
        (_world_understanding_graphics() or REPO_ROOT)
        / "render_ovrtx.py"
    )
    if not wu.is_file():
        _pytest_mod = __import__("pytest")
        _pytest_mod.skip("world_understanding tree not present (standalone usd-cli)")

    source = wu.read_text(encoding="utf-8")
    theirs = dict(
        re.findall(r'^(_OVRTX_(?:MANAGED|PROVISIONING)_MARKER)\s*=\s*"([^"]+)"',
                   source, re.MULTILINE)
    )
    assert theirs.get("_OVRTX_MANAGED_MARKER") == ovrtx._WU_MANAGED_MARKER_NAME, (
        "world_understanding renamed its managed-runtime marker; usd-cli would "
        "stop recognising it and delete that runtime"
    )
    assert theirs.get("_OVRTX_PROVISIONING_MARKER") == (
        ovrtx._WU_PROVISIONING_MARKER_NAME
    ), (
        "world_understanding renamed its in-flight marker; usd-cli would treat "
        "an interrupted wu provision as an unowned tree"
    )


def test_the_install_probe_imports_and_ignores_a_stray_pythonpath(tmp_path, monkeypatch):
    """_installed_ovrtx_version is the only barrier between an install and the
    readiness stamp, and every other test monkeypatches it. Exercise it: it must
    import rather than read dist-info (an interrupted install leaves metadata
    intact and the payload broken), drop PYTHONPATH the way the daemon does, and
    report None when the interpreter cannot run."""
    from usd_core.render import ovrtx

    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "injected"))
    seen = {}

    class _Probe:
        returncode = 0
        stdout = _qualified_probe_stdout(ovrtx)

    def record(cmd, **kwargs):
        seen["cmd"] = cmd
        seen["env"] = kwargs.get("env")
        return _Probe()

    monkeypatch.setattr(ovrtx.subprocess, "run", record)
    assert ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python") == (
        ovrtx._pinned_ovrtx_version())
    assert seen["env"] is not None and "PYTHONPATH" not in seen["env"]
    assert "import numpy, ovrtx, ovstage, warp" in seen["cmd"][-1], (
        "must import, not just read dist-info: %s" % seen["cmd"][-1])

    class _Failed:
        returncode = 1
        stdout = ""

    monkeypatch.setattr(ovrtx.subprocess, "run", lambda *a, **k: _Failed())
    assert ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python") is None

    def explode(*a, **k):
        raise OSError("not executable")

    monkeypatch.setattr(ovrtx.subprocess, "run", explode)
    assert ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python") is None


def test_the_printed_recipe_verifies_and_claims_before_it_installs(tmp_path, monkeypatch):
    """The recipe is the last writer of the readiness marker. It must prove the
    tree imports before stamping -- the automated path and the image both do --
    and it must claim the directory before the fallible install, or a failed
    manual run leaves a populated unmarked tree that auto-provision refuses to
    repair."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    # uv is not in usd-cli's dev extra, and the recipe is only
    # offered when _uv_command() resolves. Without this the CI
    # gate sees 'no recipe is offered: uv is required...'.
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)
    recipe = str(caught.value)

    lines = recipe.splitlines()
    start = next(
        n for n, line in enumerate(lines) if "isolated venv" in line
    )
    body = chr(10).join(lines[start:])
    assert "import numpy, ovrtx, ovstage, warp" in body, body
    assert "from PIL import Image" in body, body
    # -I: environment, user site, and current-directory imports cannot vouch
    # for the staged tree.
    assert "-I -c" in body, body
    assert ovrtx._pinned_ovrtx_version() in body

    # order matters: claim, install, verify, stamp
    def at(needle):
        return body.index(needle)

    assert "venv --clear" not in body
    assert at(" -m venv ") < at(ovrtx._PROVISIONING_MARKER_NAME)
    assert at(ovrtx._PROVISIONING_MARKER_NAME) < at("pip install --python")
    assert at("pip install --python") < at("import numpy, ovrtx, ovstage, warp")
    assert at("import numpy, ovrtx, ovstage, warp") < at(
        ovrtx._READY_MARKER_NAME
    )


def test_the_service_image_never_replaces_its_baked_runtime() -> None:
    """Readiness is the marker's contents now, so a persisted volume from an
    older image or a remapped WU_OVRTX_VENV_DIR reaches the replace path, which
    rmtrees ~2.5 GB and then needs pypi.nvidia.com. The world-understanding
    image guards this with the same variable."""
    dockerfile = (
        REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile"
    ).read_text(encoding="utf-8")
    # an active ENV line, not one left behind in a comment: this block
    # interleaves comments with backslash continuations, so a bare
    # substring check stays green with the guard commented out
    assert re.search(
        r'^\s*WU_OVRTX_AUTO_PROVISION=0', dockerfile, re.MULTILINE
    ), dockerfile


def test_no_recipe_is_offered_when_uv_cannot_be_resolved(tmp_path, monkeypatch):
    """A bootstrap recipe is withheld unless its installer can be resolved."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"

    def no_uv():
        raise RuntimeError("uv is required to provision the pinned ovrtx runtime venv")

    monkeypatch.setattr(ovrtx, "_uv_command", no_uv)
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)
    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "uv is required" in message

    # and when uv resolves, the recipe names what usd-cli would actually run
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["/opt/tools/uv"])
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)
    install = [l for l in str(caught.value).splitlines() if "pip install --python" in l]
    assert install and "/opt/tools/uv" in install[0], install


def test_the_adoption_recipe_is_quoted_and_ignores_pythonpath(tmp_path, monkeypatch):
    """The refusal for a populated unmarked tree ends in a marker stamp, so it is
    a writer like the others. Unquoted, `printf ... > /srv/render cache/...`
    truncates /srv/render and never writes the marker; without -I, an ovrtx
    reachable only through PYTHONPATH vouches for a venv the daemon then cannot
    import from."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "srv render" / "ovrtx_venv"
    (venv_dir / "lib").mkdir(parents=True)
    (venv_dir / "lib" / "someone_elses.so").write_text("not ours")

    runtime_lock = tmp_path / "lock.toml"
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._provision_venv(venv_dir)
    message = str(caught.value)

    probe = [line for line in message.splitlines() if "-I -c" in line]
    # the daemon imports all three; a hand-built venv from a one-package
    # lock passes an ovrtx-only probe and then dies at startup
    assert probe and "import numpy, ovrtx, ovstage, warp" in probe[0], message
    assert "from PIL import Image" in probe[0], message
    probe_tokens = _shlex.split(probe[0].strip().removesuffix("\\").strip())
    assert str(venv_dir / "bin" / "python") in probe_tokens

    stamp = [l for l in message.splitlines() if "printf" in l][0]
    # the redirect target must survive the shell as one argument
    assert str(venv_dir / ovrtx._READY_MARKER_NAME) in _shlex.split(
        stamp.replace(">", " ").strip())


def test_the_running_environment_is_never_replaced(tmp_path, monkeypatch):
    """WU_OVRTX_VENV_DIR aimed at the venv usd-cli is executing from is reachable
    in practice: the provisioner this PR replaced installed in place, so that
    directory still carries our readiness marker, and the version check routes a
    0.3 marker straight to the replacement path. Deleting it removes
    sys.executable and the project's dependencies mid-render."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    running = tmp_path / "project_venv"
    (running / "bin").mkdir(parents=True)
    interpreter = running / "bin" / "python"
    interpreter.write_text("the interpreter this process is using")
    (running / "lib").mkdir()
    (running / "lib" / "everything_we_depend_on.py").write_text("...")
    # a stale marker from the in-place provisioner this PR replaced
    (running / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")

    monkeypatch.setattr(ovrtx.sys, "executable", str(interpreter))
    monkeypatch.setattr(ovrtx.sys, "prefix", str(running))

    runtime_lock = tmp_path / "lock.toml"
    runtime_lock.write_text(
        _satisfiable_lock_text(ovrtx._pinned_ovrtx_version()), encoding="utf-8")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: runtime_lock)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)

    with _pytest.raises(RuntimeError, match="running from"):
        ovrtx._provision_venv(running)

    assert interpreter.exists(), "usd-cli deleted the interpreter it is using"
    assert (running / "lib" / "everything_we_depend_on.py").exists()

    # a directory of its own is still provisioned normally
    elsewhere = tmp_path / "ovrtx_venv"

    def create_staged_venv(cmd, *, what, **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            assert staged != elsewhere
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()

    monkeypatch.setattr(ovrtx, "_run_logged", create_staged_venv)
    monkeypatch.setattr(
        ovrtx, "_installed_ovrtx_version", lambda py: ovrtx._pinned_ovrtx_version())
    ovrtx._provision_venv(elsewhere)
    _assert_ready_marker(
        elsewhere / ovrtx._READY_MARKER_NAME, ovrtx, runtime_lock
    )


def test_readiness_requires_every_import_the_daemon_makes(tmp_path, monkeypatch):
    """The install is --no-deps, so an override lock can carry the pinned ovrtx
    alone and satisfy the version check while numpy or Pillow are missing. The
    daemon then dies on import and the marker keeps vouching for the tree."""
    from usd_core.render import ovrtx

    seen = {}

    class _Probe:
        returncode = 0
        stdout = _qualified_probe_stdout(ovrtx)

    def record(cmd, **kwargs):
        seen["cmd"] = cmd[-1]
        return _Probe()

    monkeypatch.setattr(ovrtx.subprocess, "run", record)
    ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python")

    for module in ("ovrtx", "ovstage", "warp", "numpy", "PIL"):
        assert module in seen["cmd"], (
            "%s is imported by the daemon but not by the readiness probe: %s"
            % (module, seen["cmd"])
        )


@pytest.mark.parametrize("missing", ["ovstage", "warp-lang"])
def test_readiness_rejects_a_missing_worker_runtime(
    missing, tmp_path, monkeypatch
) -> None:
    """An ovrtx-only or ovrtx+ovstage environment must never launch the worker."""
    from usd_core.render import ovrtx

    incomplete = dict(ovrtx._QUALIFIED_RUNTIME_PROFILE)
    incomplete.pop(missing)

    class _Probe:
        returncode = 0
        stdout = json.dumps(incomplete, sort_keys=True) + "\n"

    monkeypatch.setattr(ovrtx.subprocess, "run", lambda *a, **k: _Probe())
    assert ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python") is None


def test_the_recipe_never_tells_you_to_clear_the_environment_you_are_in(tmp_path, monkeypatch):
    """_provision_venv refuses to replace the venv usd-cli runs from, but the
    refusal *message* is a separate path and printed the command anyway. Since
    auto-provisioning is off by default -- and the service image now forces it
    off -- that message is the common path, not an edge case. Following it
    deletes sys.executable, usd_core and everything installed beside them."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)

    running = tmp_path / "project_venv"
    (running / "bin").mkdir(parents=True)
    interpreter = running / "bin" / "python"
    interpreter.write_text("the interpreter this process is using")
    # the marker the in-place provisioner this PR replaced would have left
    (running / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")
    monkeypatch.setattr(ovrtx.sys, "executable", str(interpreter))
    monkeypatch.setattr(ovrtx.sys, "prefix", str(running))

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(running, auto_install=False)

    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "running from" in message


def test_a_lock_that_names_wheels_by_path_is_not_refused(tmp_path, monkeypatch):
    """PEP 751 entries may carry `path` instead of `url` -- how a lock
    references locally mirrored wheels, the documented setup where
    pypi.nvidia.com is blocked. Reading a missing url as an empty string
    refused every such install; skipping path entries instead let a
    wrong-architecture mirror through to the rmtree. The filename names the
    platform either way, so it is checked either way."""
    from usd_core.render import ovrtx

    version = ovrtx._pinned_ovrtx_version()
    mirrored = tmp_path / "mirrored-pylock.toml"
    mirrored.write_text(
        _satisfiable_lock_text(version)
        + 'wheels = [{ path = "/srv/wheels/ovrtx-%s-py3-none-manylinux_2_35_x86_64.whl" }]'
        % version + chr(10),
        encoding="utf-8",
    )
    monkeypatch.setattr(ovrtx.sys, "platform", "linux")

    # the mirrored wheel is x86_64; on a matching host it installs
    monkeypatch.setattr(ovrtx.platform, "machine", lambda: "x86_64")
    assert ovrtx._lock_provides_pin(mirrored) is None, (
        "a mirrored wheel this machine can run must not be refused"
    )

    # and on a host it cannot run, it is refused BEFORE the rmtree
    monkeypatch.setattr(ovrtx.platform, "machine", lambda: "aarch64")
    assert ovrtx._lock_provides_pin(mirrored) is not None, (
        "a path entry still names the platform; it must be checked"
    )

    # a url-bearing entry for another machine is still refused
    url_based = tmp_path / "x86-pylock.toml"
    url_based.write_text(
        _satisfiable_lock_text(version)
        + 'wheels = [{ url = "https://pypi.nvidia.com/ovrtx/ovrtx-%s-py3-none-manylinux_2_35_x86_64.whl" }]'
        % version + chr(10),
        encoding="utf-8",
    )
    assert ovrtx._lock_provides_pin(url_based) is not None


def test_the_readiness_probe_ignores_a_stray_pythonhome(tmp_path, monkeypatch):
    """PYTHONHOME relocates the standard library and can resolve packages from
    outside the venv being validated. The printed recipes use -E, which ignores
    it; the automated probe has to drop it explicitly."""
    from usd_core.render import ovrtx

    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "elsewhere"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "injected"))
    seen = {}

    class _Probe:
        returncode = 0
        stdout = _qualified_probe_stdout(ovrtx)

    monkeypatch.setattr(
        ovrtx.subprocess, "run",
        lambda cmd, **kw: (seen.update(env=kw.get("env")), _Probe())[1])
    ovrtx._installed_ovrtx_version(tmp_path / "bin" / "python")

    for leaked in ("PYTHONHOME", "PYTHONPATH"):
        assert leaked not in (seen["env"] or {}), leaked


def test_the_image_proves_the_baked_venv_imports_before_stamping_it() -> None:
    """The image is the fourth writer of the readiness marker. It must prove the
    venv imports what the daemon imports before recording it ready, or it ships
    a runtime the service accepts and the daemon dies on."""
    dockerfile = (
        REPO_ROOT / "apps" / "ovrtx_rendering_api" / "Dockerfile"
    ).read_text(encoding="utf-8")
    proof = dockerfile.index("import ovrtx, ovstage, warp, numpy")
    stamp = dockerfile.index(".usd-cli-ovrtx-ready")
    assert proof < stamp, "the import proof must run before the marker is written"


def test_a_refusal_reports_every_reason_the_lock_is_unusable(tmp_path, monkeypatch):
    """Both writers gate on the same two checks. `or` reported whichever failed
    first, so an operator whose override lock needs a different interpreter AND
    names the wrong ovrtx fixed one axis and met the identical refusal with the
    other reason still hidden."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    major, minor = sys.version_info[:2]
    both_wrong = tmp_path / "doubly-wrong-pylock.toml"
    both_wrong.write_text(
        'requires-python = ">=%d.%d"' % (major, minor + 50) + chr(10)
        + "[[packages]]" + chr(10)
        + 'name = "ovrtx"' + chr(10)
        + 'version = "0.3.0.312915"' + chr(10),
        encoding="utf-8",
    )

    # the automated path
    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("the only runtime here")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: both_wrong)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", lambda *a, **k: None)

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._provision_venv(venv_dir)
    message = str(caught.value)
    assert "requires Python" in message, message
    assert "pins ovrtx" in message, message

    # and the copy-pasteable path
    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(both_wrong))
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(tmp_path / "other_venv", auto_install=False)
    recipe_message = str(caught.value)
    assert "requires Python" in recipe_message, recipe_message
    assert "pins ovrtx" in recipe_message, recipe_message


def test_a_broken_lock_override_withholds_the_recipe(tmp_path, monkeypatch):
    """WU_OVRTX_RUNTIME_LOCK pointing at a missing or unreadable file used to be
    swallowed, and the recipe printed against the shipped lock instead. Following
    it installs artifacts the operator did not choose and stamps the pin on the
    result, so the bad override is never noticed."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(tmp_path / "does-not-exist.toml"))
    inspected_locks = []

    def record_lock_inspection(runtime_lock):
        inspected_locks.append(runtime_lock)
        return None

    monkeypatch.setattr(ovrtx, "_lock_provides_pin", record_lock_inspection)
    monkeypatch.setattr(
        ovrtx, "_interpreter_satisfies_lock", record_lock_inspection
    )

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)

    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert inspected_locks == [], "a rejected override must not inspect a fallback lock"
    assert "does-not-exist.toml" in message, (
        "the refusal must name the override that failed, not the "
        "shipped lock it would otherwise fall back to"
    )


def test_readiness_tracks_the_whole_lock_not_only_the_ovrtx_version(tmp_path, monkeypatch):
    """_provision_venv installs ovrtx, numpy and pillow from the lock, but
    readiness compared the ovrtx pin alone. A lock that moves numpy or pillow
    *without* moving ovrtx therefore left every existing venv reporting ready
    while serving superseded companions -- the staleness class the pin check
    closed for ovrtx, arriving through the other half of the lock. This mirrors
    world_understanding, which already records runtime_lock_sha256."""
    from usd_core.render import ovrtx

    lock = tmp_path / "pylock.toml"

    def write_lock(numpy_version):
        lock.write_text(
            'requires-python = ">=3.0"' + chr(10)
            + "[[packages]]" + chr(10)
            + 'name = "ovrtx"' + chr(10)
            + 'version = "%s"' % ovrtx._pinned_ovrtx_version() + chr(10)
            + "[[packages]]" + chr(10)
            + 'name = "numpy"' + chr(10)
            + 'version = "%s"' % numpy_version + chr(10),
            encoding="utf-8",
        )

    write_lock("2.4.6")
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: lock)

    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").touch()
    marker = venv_dir / ovrtx._READY_MARKER_NAME
    body = ovrtx._readiness_marker_body(lock)
    payload = {
        "packages": ovrtx._QUALIFIED_RUNTIME_PROFILE,
        ovrtx._LOCK_DIGEST_KEY: ovrtx._runtime_lock_digest(lock),
        "schema_version": ovrtx._READY_MARKER_SCHEMA_VERSION,
    }
    assert body == json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n"
    marker.write_text(body, encoding="utf-8")

    assert ovrtx._venv_matches_pin(venv_dir), "an unchanged lock must stay ready"

    marker.write_text(" " + body, encoding="utf-8")
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "only the canonical marker bytes may attest readiness"
    )
    marker.write_text(body, encoding="utf-8")

    # numpy moves; ovrtx does not
    write_lock("2.4.7")
    assert not ovrtx._venv_matches_pin(venv_dir), (
        "a lock that moves numpy without moving ovrtx must invalidate readiness"
    )
    write_lock("2.4.6")

    # a marker written before this contract existed records no digest, so it
    # cannot prove its companions and is reprovisioned rather than trusted
    marker.write_text(ovrtx.OVRTX_PIN, encoding="utf-8")
    assert not ovrtx._venv_matches_pin(venv_dir)

    old_v2 = {
        "packages": ovrtx._QUALIFIED_RUNTIME_PROFILE,
        "schema_version": "usd-cli.ovrtx-runtime-ready.v2",
    }
    marker.write_text(
        json.dumps(old_v2, separators=(",", ":"), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert not ovrtx._venv_matches_pin(venv_dir)

    for stale_payload in (
        "not-json\n",
        json.dumps({**payload, "schema_version": "wrong"}) + "\n",
        json.dumps({**payload, "packages": {"ovrtx": "wrong"}}) + "\n",
        json.dumps({**payload, ovrtx._LOCK_DIGEST_KEY: "0" * 64}) + "\n",
    ):
        marker.write_text(stale_payload, encoding="utf-8")
        assert not ovrtx._venv_matches_pin(venv_dir)

    marker.write_bytes(b"\xff")
    assert not ovrtx._venv_matches_pin(venv_dir)

    with pytest.raises(RuntimeError, match="invalid SHA-256 identity"):
        ovrtx._readiness_marker_body(lock, runtime_lock_digest="not-a-digest")

    # and an unreadable lock cannot prove anything either
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: tmp_path / "gone.toml")
    marker.write_text(ovrtx._readiness_marker_body(lock), encoding="utf-8")
    assert not ovrtx._venv_matches_pin(venv_dir)


def test_no_recipe_is_offered_when_the_lock_identity_cannot_be_read(tmp_path, monkeypatch):
    """The recipe's last step writes the lock digest into the marker. If the
    digest cannot be computed the command would embed the literal "None", and
    an operator following it stamps a marker that can never match -- so the
    venv would be reprovisioned on every render. Withhold the recipe instead,
    as for every other reason the lock is unusable."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.0.0.000000")

    monkeypatch.setattr(ovrtx, "_runtime_lock_digest", lambda lock: None)
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(venv_dir, auto_install=False)
    message = str(caught.value)
    assert "venv --clear" not in message, message
    assert "could not be read to record its identity" in message


def test_recipe_records_a_readable_lock_digest(tmp_path, monkeypatch):
    """A usable lock produces a concrete digest in the manual recipe."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    monkeypatch.delenv("WU_OVRTX_AUTO_PROVISION", raising=False)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    fresh_venv = tmp_path / "fresh_ovrtx_venv"
    with _pytest.raises(RuntimeError) as caught:
        ovrtx._ovrtx_python(fresh_venv, auto_install=False)
    stamp = [
        line for line in str(caught.value).splitlines()
        if ovrtx._LOCK_DIGEST_KEY in line
    ]
    assert stamp, str(caught.value)
    assert ovrtx._runtime_lock_digest(lock) in stamp[0], stamp[0]
    assert "=None" not in stamp[0], stamp[0]


def test_lock_change_during_install_never_produces_a_ready_marker(tmp_path, monkeypatch):
    """The marker must identify the bytes uv actually installed.

    Hashing only after installation can record a replacement lock and accept a
    runtime installed from the previous bytes as current.
    """
    import pytest as _pytest

    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    before = ovrtx._runtime_lock_digest(lock)
    venv_dir = tmp_path / "ovrtx_venv"

    def fake_run(cmd, *, what="", **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
            return
        snapshot = Path(cmd[cmd.index("-r") + 1])
        assert snapshot != lock
        assert snapshot.name.startswith("pylock.")
        assert ovrtx._runtime_lock_digest(snapshot) == before
        lock.write_text(lock.read_text(encoding="utf-8") + "# replaced\n")

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", fake_run)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )

    with _pytest.raises(RuntimeError, match="changed while.*provisioned"):
        ovrtx._provision_venv(venv_dir)

    assert before != ovrtx._runtime_lock_digest(lock)
    assert not (venv_dir / ovrtx._READY_MARKER_NAME).exists()


def test_directly_mounted_runtime_is_refused_before_staging(tmp_path, monkeypatch):
    """A mount root cannot participate in the atomic directory swap."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "mounted_ovrtx_venv"
    venv_dir.mkdir()
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(
        ovrtx.os.path,
        "ismount",
        lambda path: Path(path) == venv_dir,
    )
    monkeypatch.setattr(
        ovrtx,
        "_run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a mount point must be rejected before staging")
        ),
    )

    with _pytest.raises(RuntimeError, match="mount point") as caught:
        ovrtx._provision_venv(venv_dir)

    assert "Mount its parent" in str(caught.value)
    assert not list(tmp_path.glob(".mounted_ovrtx_venv.staging-*"))


def test_group_or_world_writable_runtime_parent_is_refused(tmp_path, monkeypatch):
    """An operator override may not place fixed child writes in a shared parent."""
    import os
    import stat

    import pytest as _pytest

    from usd_core.render import ovrtx

    if not hasattr(os, "geteuid"):
        _pytest.skip("POSIX ownership contract")
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    shared_parent = tmp_path / "shared"
    shared_parent.mkdir()
    shared_parent.chmod(0o777)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(
        ovrtx,
        "_run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("an untrusted parent must be rejected before staging")
        ),
    )

    with _pytest.raises(RuntimeError, match="group- or world-writable"):
        ovrtx._provision_venv(shared_parent / "ovrtx_venv")

    assert stat.S_IMODE(shared_parent.stat().st_mode) == 0o777
    assert list(shared_parent.iterdir()) == []


def test_runtime_parent_replacement_during_install_stops_path_writes(
    tmp_path, monkeypatch
):
    """A renamed parent is detected before any later staged-child operation."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    if not hasattr(ovrtx.os, "geteuid"):
        _pytest.skip("POSIX identity contract")
    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    runtime_parent = tmp_path / "runtime-parent"
    runtime_parent.mkdir(mode=0o700)
    moved_parent = tmp_path / "moved-runtime-parent"
    venv_dir = runtime_parent / "ovrtx_venv"

    def replace_parent(cmd, *, what="", **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
            return
        runtime_parent.rename(moved_parent)
        runtime_parent.mkdir(mode=0o700)

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", replace_parent)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )

    with _pytest.raises(RuntimeError, match="runtime parent changed"):
        ovrtx._provision_venv(venv_dir)

    assert list(runtime_parent.iterdir()) == []
    assert list(moved_parent.glob(".ovrtx_venv.staging-*"))


def test_runtime_sibling_claim_does_not_follow_a_preexisting_symlink(tmp_path):
    """The external recovery marker must be a newly created regular file."""
    import os

    import pytest as _pytest

    from usd_core.render import ovrtx

    if not hasattr(os, "O_NOFOLLOW"):
        _pytest.skip("no-follow flags unavailable")
    candidate = tmp_path / ".ovrtx_venv.staging-candidate"
    candidate.mkdir()
    survivor = tmp_path / "operator-data"
    survivor.write_text("keep", encoding="utf-8")
    marker = ovrtx._sibling_owner_marker(candidate)
    marker.symlink_to(survivor)

    with _pytest.raises(FileExistsError):
        ovrtx._claim_runtime_sibling(candidate)

    assert survivor.read_text(encoding="utf-8") == "keep"
    assert marker.is_symlink()


def test_symlinked_runtime_is_refused_before_staging(tmp_path, monkeypatch):
    """Replacing a leaf symlink would silently discard its storage routing."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    real_venv = tmp_path / "real_ovrtx_venv"
    real_venv.mkdir()
    survivor = real_venv / "keep.bin"
    survivor.write_text("operator data")
    (real_venv / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")
    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.symlink_to(real_venv, target_is_directory=True)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(
        ovrtx,
        "_run_logged",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("a symlink must be rejected before staging")
        ),
    )

    with _pytest.raises(RuntimeError, match="target is a symlink"):
        ovrtx._provision_venv(venv_dir)

    assert venv_dir.is_symlink()
    assert survivor.read_text() == "operator data"


def test_target_created_during_staging_is_not_replaced(tmp_path, monkeypatch):
    """A second provisioner may create the target during the long install."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"

    def fake_run(cmd, *, what="", **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
            return
        venv_dir.mkdir()
        (venv_dir / "other-provisioner.bin").write_text("keep me")

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", fake_run)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )

    with _pytest.raises(RuntimeError, match="target changed"):
        ovrtx._provision_venv(venv_dir)

    assert (venv_dir / "other-provisioner.bin").read_text() == "keep me"
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))


def test_empty_target_claimed_by_wu_during_staging_is_not_replaced(
    tmp_path, monkeypatch
):
    """Repeat the world-understanding ownership veto immediately before swap."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()

    def fake_run(cmd, *, what="", **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
            return
        (venv_dir / ovrtx._WU_PROVISIONING_MARKER_NAME).write_text("claimed")

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_run_logged", fake_run)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )

    with _pytest.raises(RuntimeError, match="became world-understanding managed"):
        ovrtx._provision_venv(venv_dir)

    assert (venv_dir / ovrtx._WU_PROVISIONING_MARKER_NAME).read_text() == "claimed"
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))


def test_adoption_recipe_installs_the_exact_lock_before_stamping(tmp_path, monkeypatch):
    """Importing companions does not prove they came from the recorded lock."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["/trusted/uv"])
    venv_dir = tmp_path / "unmarked_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").touch()

    with _pytest.raises(RuntimeError) as caught:
        ovrtx._provision_venv(venv_dir)

    message = str(caught.value)
    install = message.index("pip install --python")
    stamp = message.index(ovrtx._readiness_marker_body(lock).strip())
    assert install < stamp
    assert "--require-hashes --no-deps --no-config --no-sources" in message
    assert f"-r {lock}" in message
    assert "importlib.metadata" in message
    assert "metadata.version" in message
    assert "import numpy, ovrtx, ovstage, warp" in message
    assert "from PIL import Image" in message
    assert ovrtx._pinned_ovrtx_version() in message
    assert "=None" not in message


def test_failed_staged_replacement_preserves_the_active_runtime(tmp_path, monkeypatch):
    """A failed download must not damage the runtime it was meant to replace."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("working interpreter")
    old_marker = "ovrtx==0.3.0.312915"
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text(old_marker)
    survivor = venv_dir / "lib" / "only-working-runtime.bin"
    survivor.parent.mkdir()
    survivor.write_text("keep me")

    def fail_install(cmd, *, what, **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()
            return
        assert what.endswith("(locked runtime)")
        assert survivor.exists(), "the active runtime moved before validation"
        raise RuntimeError("download failed")

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: lock)
    monkeypatch.setattr(ovrtx, "_run_logged", fail_install)

    with _pytest.raises(RuntimeError, match="download failed"):
        ovrtx._provision_venv(venv_dir)

    assert survivor.read_text() == "keep me"
    assert (venv_dir / ovrtx._READY_MARKER_NAME).read_text() == old_marker
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))
    assert not list(tmp_path.glob(".ovrtx_venv.previous-*"))


def test_staged_replacement_rolls_back_when_activation_fails(tmp_path, monkeypatch):
    """The old runtime is restored if the verified sibling cannot be activated."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("working interpreter")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")
    survivor = venv_dir / "lib" / "only-working-runtime.bin"
    survivor.parent.mkdir()
    survivor.write_text("keep me")

    def successful_install(cmd, *, what, **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").touch()

    real_replace = ovrtx.os.replace
    replacements = 0

    def fail_activation(source, destination):
        nonlocal replacements
        replacements += 1
        if replacements == 2:
            raise OSError("activation failed")
        return real_replace(source, destination)

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: lock)
    monkeypatch.setattr(ovrtx, "_run_logged", successful_install)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )
    monkeypatch.setattr(ovrtx.os, "replace", fail_activation)

    with _pytest.raises(OSError, match="activation failed"):
        ovrtx._provision_venv(venv_dir)

    assert survivor.read_text() == "keep me"
    assert replacements == 3, "move old, fail activation, restore old"
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))
    assert not list(tmp_path.glob(".ovrtx_venv.previous-*"))


def test_successful_staged_replacement_swaps_only_after_validation(tmp_path, monkeypatch):
    """A verified sibling replaces the old tree and leaves no swap debris."""
    from usd_core.render import ovrtx

    lock = ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path)
    venv_dir = tmp_path / "ovrtx_venv"
    (venv_dir / "bin").mkdir(parents=True)
    (venv_dir / "bin" / "python").write_text("old interpreter")
    (venv_dir / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")
    survivor = venv_dir / "lib" / "old-shader-cache.bin"
    survivor.parent.mkdir()
    survivor.write_text("old payload")

    def successful_install(cmd, *, what, **kwargs):
        if what.endswith("(venv)"):
            staged = Path(cmd[-1])
            assert staged != venv_dir
            assert survivor.exists(), "the active runtime moved before validation"
            (staged / "bin").mkdir(parents=True, exist_ok=True)
            (staged / "bin" / "python").write_text("new interpreter")
        elif what.endswith("(locked runtime)"):
            assert survivor.exists(), "the active runtime moved during install"

    monkeypatch.setattr(ovrtx, "_uv_command", lambda: ["uv"])
    monkeypatch.setattr(ovrtx, "_ovrtx_runtime_lock", lambda: lock)
    monkeypatch.setattr(ovrtx, "_run_logged", successful_install)
    monkeypatch.setattr(
        ovrtx,
        "_installed_ovrtx_version",
        lambda py: ovrtx._pinned_ovrtx_version(),
    )

    ovrtx._provision_venv(venv_dir)

    assert not survivor.exists()
    assert (venv_dir / "bin" / "python").read_text() == "new interpreter"
    _assert_ready_marker(
        venv_dir / ovrtx._READY_MARKER_NAME, ovrtx, lock
    )
    assert not list(tmp_path.glob(".ovrtx_venv.staging-*"))
    assert not list(tmp_path.glob(".ovrtx_venv.previous-*"))


def test_interrupted_swap_restores_the_owned_backup_before_retry(tmp_path):
    """A hard exit between renames must not strand the active path missing."""
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    backup = tmp_path / ".ovrtx_venv.previous-deadbeef"
    (backup / "bin").mkdir(parents=True)
    (backup / "bin" / "python").write_text("previous interpreter")
    (backup / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")

    abandoned = tmp_path / ".ovrtx_venv.staging-cafebabe"
    abandoned.mkdir()
    (abandoned / ovrtx._PROVISIONING_MARKER_NAME).write_text(ovrtx.OVRTX_PIN)

    ovrtx._recover_interrupted_venv_swap(venv_dir)

    assert (venv_dir / "bin" / "python").read_text() == "previous interpreter"
    assert not backup.exists()
    assert not abandoned.exists()


def test_recovery_escapes_runtime_name_glob_metacharacters(tmp_path):
    """Recovery must not claim a different runtime's wildcard-matching sibling."""
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx?"
    backup = tmp_path / ".ovrtx?.previous-deadbeef"
    (backup / "bin").mkdir(parents=True)
    (backup / "bin" / "python").write_text("expected interpreter")
    (backup / ovrtx._READY_MARKER_NAME).write_text("ovrtx==0.3.0.312915")

    other = tmp_path / ".ovrtxX.previous-cafebabe"
    other.mkdir()
    (other / ovrtx._READY_MARKER_NAME).write_text("other runtime")

    ovrtx._recover_interrupted_venv_swap(venv_dir)

    assert (venv_dir / "bin" / "python").read_text() == "expected interpreter"
    assert other.exists()


def test_recovery_never_activates_a_sidecar_only_partially_cleaned_backup(
    tmp_path,
):
    """A sidecar proves cleanup ownership, not readiness for reactivation."""
    import pytest as _pytest

    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    backup = tmp_path / ".ovrtx_venv.previous-deadbeef"
    (backup / "bin").mkdir(parents=True)
    (backup / "bin" / "python").write_text("previous interpreter")
    ovrtx._claim_runtime_sibling(backup)

    with _pytest.raises(RuntimeError, match="payload may be partial"):
        ovrtx._recover_interrupted_venv_swap(venv_dir)

    assert not venv_dir.exists()
    assert (backup / "bin" / "python").read_text() == "previous interpreter"
    assert ovrtx._sibling_owner_marker(backup).exists()


def test_recovery_keeps_a_working_backup_until_the_target_is_verified(tmp_path):
    """A recreated partial target is not evidence that activation succeeded."""
    from usd_core.render import ovrtx

    venv_dir = tmp_path / "ovrtx_venv"
    venv_dir.mkdir()
    (venv_dir / ovrtx._PROVISIONING_MARKER_NAME).write_text(ovrtx.OVRTX_PIN)
    backup = tmp_path / ".ovrtx_venv.previous-deadbeef"
    (backup / "bin").mkdir(parents=True)
    (backup / "bin" / "python").write_text("only working interpreter")
    ovrtx._claim_runtime_sibling(backup)

    ovrtx._recover_interrupted_venv_swap(venv_dir)

    assert backup.exists()
    assert ovrtx._sibling_owner_marker(backup).exists()


def test_daemon_keeps_native_stdout_off_the_json_protocol() -> None:
    """A native warning without a newline must not consume the next JSON reply."""
    import json as _json
    import os as _os
    import subprocess as _subprocess

    from usd_core.render import ovrtx

    prelude = r'''
import os, sys, types
numpy = types.ModuleType("numpy")
numpy.isfinite = lambda value: True
sys.modules["numpy"] = numpy

warp = types.ModuleType("warp")
warp.float16 = object()
warp.float32 = object()
warp.float64 = object()
warp.int32 = object()
warp.uint8 = object()
warp.uint32 = object()
warp.array = lambda *args, **kwargs: object()
warp.kernel = lambda function: function
warp.init = lambda: None
sys.modules["warp"] = warp

ovstage = types.ModuleType("ovstage")
class _Wait:
    def wait(self):
        pass
class Stage:
    def __init__(self, name):
        pass
    def advance_write_floor(self, ordinal, scope):
        return _Wait()
    def destroy(self):
        pass
class Population:
    @staticmethod
    def open_usd(stage, path, *, ordinal):
        os.write(1, b"native-render-warning-without-newline")
ovstage.Stage = Stage
ovstage.Scope = types.SimpleNamespace(ALL=object())
ovstage.population = Population()
sys.modules["ovstage"] = ovstage

ovrtx_stub = types.ModuleType("ovrtx")
class Renderer:
    def __init__(self):
        os.write(1, b"native-startup-warning-without-newline")
    def attach_ovstage(self, stage):
        pass
    def detach_ovstage(self):
        pass
    def destroy(self):
        pass
ovrtx_stub.Renderer = Renderer
sys.modules["ovrtx"] = ovrtx_stub
pil = types.ModuleType("PIL")
pil.Image = types.ModuleType("PIL.Image")
sys.modules["PIL"] = pil
sys.modules["PIL.Image"] = pil.Image
'''
    request = {
        "command": "render",
        "usd_path": "scene.usda",
        "product_paths": ["/Render/Product"],
        "out_paths": ["out.png"],
        "cameras": ["/World/Camera"],
        "num_sensor_updates": 0,
    }
    env = _os.environ.copy()
    env["USD_CLI_OVRTX_PARENT_PID"] = str(_os.getpid())
    result = _subprocess.run(
        [sys.executable, "-c", prelude + ovrtx._DAEMON_SCRIPT],
        input=_json.dumps(request) + "\n" + _json.dumps({"command": "shutdown"}) + "\n",
        capture_output=True,
        text=True,
        check=False,
        env=env,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    replies = [_json.loads(line) for line in result.stdout.splitlines()]
    assert replies[0]["status"] == "ready"
    assert replies[0]["worker_protocol_version"] == 3
    assert replies[1]["status"] == "error"
    assert "num_sensor_updates" in replies[1]["error"]
    assert "native-" not in result.stdout
    assert "native-startup-warning-without-newline" in result.stderr
    assert "native-render-warning-without-newline" in result.stderr
