# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Black-box test harness for the usd-cli CLI.

These tests drive the *installed command surface* — they shell out to `usd-cli` exactly
like an agent would, parse the structured `--json` envelope, and assert on it. Nothing
imports `usd_core`/`usd_server` directly, so the daemon transport, argument parsing, and
the engine are all exercised end-to-end.

Design goals (so the suite stays cheap to iterate on):

* **One daemon per project, reused.** A session-scoped `project` fixture spins up a
  single per-project daemon in a temp dir and tears it down at the end. Render-backend
  tests get their own parametrized projects (the renderer is fixed at daemon startup).
* **Isolated.** Each project is a throwaway temp dir with its own ``HOME`` and ``.usd-cli/``,
  so user/global config never bleeds in and renders/logs don't touch the repo.
* **Parametrized over assets and OVRTX backends.** Add a row to ``ASSETS`` or ``BACKENDS``
  and every relevant test picks it up. Unavailable OVRTX backends are skipped, not failed.

Knobs (all optional env vars):
    USD_CLI_SAMPLE_ASSETS   dir holding the simready/* assets
                             (default: <repo>/sample_assets/simready, shipped in-tree)
    USD_CLI_TEST_OVRTX=1    enable the ovrtx backend (needs Linux + RTX + Vulkan)
    USD_CLI_TEST_RENDER_ARTIFACTS=1  also write the overlay PNGs a few tests can draw;
                            off by default so a plain run leaves no images behind
    USD_CLI_TEST_REMOTE_URL  enable the remote backend, pointed at a render service
"""

from __future__ import annotations

import json as _json
import os
import re
import struct
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"

# Pool health persists across processes by design (daemon restarts must not
# amnesty a degraded backend) — but across TESTS it leaks one test's recorded
# failures into the next test's scheduling expectations. Disable it suite-wide;
# the persistence itself is unit-tested with explicit persist paths.
os.environ.setdefault("USD_CLI_POOL_STATE_DISABLE", "1")

#: Where the sample USD assets live. Defaults to the assets shipped in this repo, so the
#: suite runs out-of-the-box on a clone or in CI; override with USD_CLI_SAMPLE_ASSETS.
SAMPLE_ASSETS = Path(
    os.environ.get(
        "USD_CLI_SAMPLE_ASSETS",
        str(REPO_ROOT / "sample_assets" / "simready"),
    )
)


# ── assets under test ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Asset:
    """A sample scene plus the facts a black-box test can assert about it."""

    id: str
    relpath: str
    prims: int  # `open`/`resolve` ref count (the indexed, non-boring prims)
    materials: int  # Material prims indexed as @m refs (counted in open/snapshot totals)
    up_axis: str
    root_name: str
    prim_names: tuple[str, ...]  # a subset that must appear in the snapshot tree

    @property
    def path(self) -> Path:
        return SAMPLE_ASSETS / self.relpath


CUBE = Asset(
    id="cube",
    relpath="Cube/cube.usda",
    prims=4,
    materials=6,
    up_axis="Z",
    root_name="cube",
    prim_names=("cube", "Materials", "Geometry", "color_cube"),
)
SPRAY = Asset(
    id="spray_bottle",
    relpath="Spray_Bottle/spray_bottle.usda",
    prims=7,
    materials=1,
    up_axis="Z",
    root_name="spray_bottle",
    prim_names=("spray_bottle", "Geometry", "bottle_body", "spray_nozzle"),
)
ASSETS = [CUBE, SPRAY]


_REMOTE_URL = os.environ.get("USD_CLI_TEST_REMOTE_URL")

BACKENDS = [
    pytest.param(
        ("ovrtx", None),
        id="ovrtx",
        marks=pytest.mark.skipif(
            not (sys.platform.startswith("linux") and os.environ.get("USD_CLI_TEST_OVRTX")),
            reason="ovrtx needs Linux+RTX; set USD_CLI_TEST_OVRTX=1 to enable",
        ),
    ),
    pytest.param(
        ("remote", _REMOTE_URL),
        id="remote",
        marks=pytest.mark.skipif(
            not _REMOTE_URL, reason="set USD_CLI_TEST_REMOTE_URL to enable",
        ),
    ),
]


# ── CLI runner ────────────────────────────────────────────────────────────────────
@dataclass
class CliResult:
    code: int
    out: str
    err: str
    argv: list[str]

    @property
    def ok(self) -> bool:
        return self.code == 0

    def json(self) -> dict:
        return _json.loads(self.out)


class Project:
    """A throwaway usd-cli project dir + its auto-started daemon."""

    def __init__(self, root: Path, env: dict[str, str]):
        self.root = root
        self.env = env

    def cli(
        self,
        *args,
        json: bool = False,
        expect_ok: bool | None = None,
        timeout: float = 180.0,
        extra_env: dict[str, str] | None = None,
    ) -> CliResult:
        argv = [sys.executable, "-m", "usd_cli"]
        if json:
            argv.append("--json")
        argv += [str(a) for a in args]
        env = dict(self.env)
        if extra_env:
            env.update(extra_env)
        proc = subprocess.run(
            argv, cwd=self.root, env=env, capture_output=True, text=True, timeout=timeout
        )
        res = CliResult(proc.returncode, proc.stdout, proc.stderr, argv)
        if expect_ok is True:
            assert res.ok, f"expected success from {args}\n  exit={res.code}\n  out={res.out}\n  err={res.err}"
        elif expect_ok is False:
            assert not res.ok, f"expected failure from {args}\n  out={res.out}"
        return res

    def open(self, asset: Asset) -> CliResult:
        # --force-reload: tests share one daemon and re-open assets to RESET
        # state, deliberately discarding the previous test's unsaved edits —
        # exactly the case the same-file reopen guard exists to intercept.
        return self.cli("open", asset.path, "--force-reload", json=True, expect_ok=True)

    def stop(self) -> None:
        self.cli("server", "stop", timeout=30)


def _make_project(root: Path, renderer: str | None = None, remote_url: str | None = None) -> Project:
    (root / ".usd-cli").mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    # Isolate from the user's global config (~/.config/usd-cli) and keep the daemon's
    # state/renders inside the temp project.
    env["HOME"] = str(root)
    # Make the package importable for both the CLI and the daemon it spawns, even
    # without an editable install.
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(SRC), os.environ.get("PYTHONPATH", "")) if p
    )
    # Backstop: a leaked daemon (teardown failed) self-destructs instead of lingering.
    # Tests always re-open their scene, so an idle restart between commands is harmless.
    env["USD_CLI_SERVER_IDLE_TIMEOUT"] = "300s"
    # server.allowed_roots defaults to the project dir alone. The suite keeps its
    # project in a throwaway temp dir but reads scenes from the checked-in sample
    # assets and writes fixtures under pytest's temp root, so both must be
    # declared or every request is refused as outside the sandbox.
    # SAMPLE_ASSETS.parent: not every sample scene lives under simready/ — the
    # instanced Siemens PCB sits at the sample_assets/ root, and at 51 MB it is read
    # in place rather than copied into each project.
    env["USD_CLI_SERVER_ALLOWED_ROOTS"] = os.pathsep.join(
        dict.fromkeys(
            str(Path(p).resolve())
            for p in (root, root.parent, SAMPLE_ASSETS, SAMPLE_ASSETS.parent,
                      tempfile.gettempdir())
        )
    )
    if renderer:
        env["USD_CLI_RENDER_RENDERER"] = renderer
    if remote_url:
        env["USD_CLI_RENDER_REMOTE_URL"] = remote_url
    return Project(root, env)


# ── fixtures ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="session", autouse=True)
def _assets_present():
    if not SAMPLE_ASSETS.is_dir():
        pytest.skip(
            f"sample assets not found at {SAMPLE_ASSETS} "
            "(set USD_CLI_SAMPLE_ASSETS to override)"
        )
    missing = [a.relpath for a in ASSETS if not a.path.exists()]
    if missing:
        pytest.skip(f"missing sample assets: {missing}")


@pytest.fixture(scope="session")
def qualified_camera_analysis_runtime():
    """Skip only tests that execute the optional Newton/Warp backend.

    The development extra deliberately leaves camera analysis optional.  The
    isolated component artifact gate installs ``dev,camera-analysis``, so these tests
    still run there; a base development environment keeps the import-light and
    validation contracts runnable instead of failing during backend startup.
    """
    from usd_core.camera_analysis.newton_backend import backend_versions

    versions = backend_versions()
    if versions.get("qualified") is not True:
        pytest.skip(
            "requires optional qualified camera-analysis runtime "
            "(Newton 1.5.x + Warp 1.16.x); found "
            f"newton={versions.get('newton', 'missing')}, "
            f"warp={versions.get('warp', 'missing')}"
        )
    return versions


@pytest.fixture(scope="session")
def qualified_warp_runtime():
    """Skip Warp-only execution tests when the scoring runtime is absent."""
    from usd_core.camera_analysis.newton_backend import (
        QUALIFIED_WARP_VERSION_PREFIX,
        backend_versions,
    )

    version = str(backend_versions().get("warp", "missing"))
    if not version.startswith(QUALIFIED_WARP_VERSION_PREFIX):
        pytest.skip(
            "requires optional qualified Warp scoring runtime "
            f"({QUALIFIED_WARP_VERSION_PREFIX}x); found warp={version}"
        )
    return version


@pytest.fixture
def qualified_camera_analysis_metadata(monkeypatch):
    """Qualify metadata for tests that prove work stops before runtime import."""
    from usd_core.camera_analysis import newton_backend

    versions = {"newton": "1.5.0", "warp": "1.16.0", "qualified": True}
    monkeypatch.setattr(
        newton_backend,
        "require_qualified_backend_versions",
        lambda: versions,
    )
    return versions


@pytest.fixture(scope="session")
def project(tmp_path_factory) -> Project:
    """A single shared daemon for analytic perception and camera tests."""
    proj = _make_project(tmp_path_factory.mktemp("ov_proj"))
    yield proj
    proj.stop()


@pytest.fixture(params=BACKENDS, scope="module")
def render_project(request, tmp_path_factory) -> Project:
    """A daemon whose renderer is fixed at startup — one project per backend param."""
    renderer, remote_url = request.param
    proj = _make_project(
        tmp_path_factory.mktemp(f"ov_render_{renderer}"),
        renderer=renderer,
        remote_url=remote_url,
    )
    yield proj
    proj.stop()


# ── helpers shared across test modules ──────────────────────────────────────────
def png_dimensions(path: str | Path) -> tuple[int, int]:
    """Read a PNG's (width, height) from its IHDR header — no image library needed."""
    with open(path, "rb") as fh:
        head = fh.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    width, height = struct.unpack(">II", head[16:24])
    return width, height


def ovrtx_lock_this_interpreter_supports(monkeypatch, tmp_path):
    """Point ovrtx provisioning at the shipped lock, widened to the running
    interpreter.

    usd-cli supports >=3.11,<3.13 but its ovrtx lock declares >=3.12 and pins
    cp312 numpy/pillow wheels, so on 3.11 the compatibility guard refuses
    before any of the provisioning mechanics run. Widening the lock itself is
    tracked separately (#1263); tests that exercise the install command shape
    or the auto-provision gating are not about lock breadth, so they get a
    lock this interpreter satisfies. Returns the path the run should use.
    """
    from usd_core.render import ovrtx

    text = ovrtx.OVRTX_RUNTIME_LOCK.read_text(encoding="utf-8")
    widened = re.sub(
        r'requires-python\s*=\s*"[^"]*"',
        'requires-python = ">=%d.%d"' % sys.version_info[:2],
        text,
        count=1,
    )
    # The shipped lock carries x86_64 manylinux wheels for each qualified
    # worker package, and _wheel_runs_here checks OS as well as architecture.
    # These tests cover provisioning mechanics on whatever host runs them --
    # including Windows developer machines and the aarch64 image build -- so
    # give those entries platform-agnostic wheels rather than teaching every
    # test about the host's tags. Lock breadth itself is issue #1263.
    widened = re.sub(
        r'((?:ovrtx|ovstage|warp_lang)-[^/]*?)-py3-none-[a-z0-9_.]+\.whl',
        r'\1-py3-none-any.whl',
        widened,
    )
    lock = tmp_path / "pylock.ovrtx-runtime.toml"
    lock.write_text(widened, encoding="utf-8")
    monkeypatch.setenv("WU_OVRTX_RUNTIME_LOCK", str(lock))
    return lock
