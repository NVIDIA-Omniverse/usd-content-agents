#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Provision the pinned IsaacLab BYOR example and emit its runtime config."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

ISAACLAB_URL = "https://github.com/isaac-sim/IsaacLab.git"
ISAACLAB_COMMIT = "100ef39f128207cb536f87c77a7481fa416f6a7d"
ISAACSIM_VERSION = "6.0.1.0"
NVIDIA_PYPI_URL = "https://pypi.nvidia.com"


def _run(
    argv: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    subprocess.run(
        argv,
        cwd=str(cwd) if cwd else None,
        env=environment,
        check=True,
        shell=False,
    )


def _output(argv: list[str], *, cwd: Path) -> str:
    return subprocess.check_output(argv, cwd=str(cwd), text=True, shell=False).strip()


def _clean_setup_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("VIRTUAL_ENV", None)
    environment.pop("CONDA_PREFIX", None)
    return environment


def _require_eula_consent() -> None:
    accepted = os.environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().lower()
    if accepted not in {"yes", "y", "1"}:
        raise RuntimeError(
            "set OMNI_KIT_ACCEPT_EULA=YES after reviewing the NVIDIA "
            "Omniverse License Agreement"
        )


def _prepare_checkout(checkout: Path) -> None:
    fresh_clone = not checkout.exists()
    if checkout.exists() and not (checkout / ".git").exists():
        raise RuntimeError(
            f"checkout exists but is not an IsaacLab Git checkout: {checkout}"
        )
    if fresh_clone:
        checkout.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                ISAACLAB_URL,
                str(checkout),
            ]
        )
    if not fresh_clone and _output(["git", "status", "--porcelain"], cwd=checkout):
        raise RuntimeError(f"refusing to change a dirty IsaacLab checkout: {checkout}")
    current = _output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if current != ISAACLAB_COMMIT:
        _run(["git", "fetch", "origin", ISAACLAB_COMMIT], cwd=checkout)
    if fresh_clone or current != ISAACLAB_COMMIT:
        _run(["git", "checkout", "--detach", ISAACLAB_COMMIT], cwd=checkout)
    resolved = _output(["git", "rev-parse", "HEAD"], cwd=checkout)
    if resolved != ISAACLAB_COMMIT:
        raise RuntimeError(
            f"IsaacLab checkout resolved to {resolved}, expected {ISAACLAB_COMMIT}"
        )


def _install(checkout: Path) -> None:
    environment = _clean_setup_environment()
    launcher = str(checkout / "isaaclab.sh")
    environment_path = checkout / "env_isaaclab"
    _run(
        [launcher, "--uv", "env_isaaclab"],
        cwd=checkout,
        environment=environment,
    )
    environment["VIRTUAL_ENV"] = str(environment_path)
    executable_dir = environment_path / ("Scripts" if os.name == "nt" else "bin")
    environment["PATH"] = os.pathsep.join(
        [str(executable_dir), environment.get("PATH", "")]
    )
    _run(
        [
            "uv",
            "pip",
            "install",
            f"isaacsim[all,extscache]=={ISAACSIM_VERSION}",
            "--extra-index-url",
            NVIDIA_PYPI_URL,
            "--index-strategy",
            "unsafe-best-match",
        ],
        cwd=checkout,
        environment=environment,
    )
    _run(
        [launcher, "--install", "none"],
        cwd=checkout,
        environment=environment,
    )


def _smoke(checkout: Path) -> None:
    _run(
        [
            str(checkout / "isaaclab.sh"),
            "-p",
            "-c",
            "from isaaclab.app import AppLauncher; launcher = AppLauncher(); launcher.app.close(); print('IsaacLab launch OK')",
        ],
        cwd=checkout,
        environment=_clean_setup_environment(),
    )


def _config(checkout: Path, adapter: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task": "isaaclab_cube_bounce",
        "runtime": {
            "python": str(checkout / "isaaclab.sh"),
            "python_args": ["-p"],
            "script": str(adapter),
            "cwd": str(checkout),
            "timeout_s": 900,
            "extra_args": [],
            "pass_env": ["OMNI_KIT_ACCEPT_EULA", "PRIVACY_CONSENT"],
            "fingerprint_paths": [
                str(checkout / "source" / "isaaclab"),
                str(checkout / "source" / "isaaclab_physx"),
                str(adapter),
            ],
            "trial": {
                "drop_height_m": 1.0,
                "target_bounce_height_m": 0.55,
                "steps": 360,
                "dt_s": 0.008333333333333333,
            },
        },
        "parameters": {
            "restitution": {"min": 0.0, "max": 1.0},
        },
        "objective": {
            "name": "bounce_height_error",
            "unit": "m",
            "direction": "minimize",
            "failure_penalty": 1000000000000.0,
        },
        "optimizer": {
            "name": "botorch",
            "max_trials": 30,
            "seed": 42,
            "replicas": 1,
            "replica_seed": 1000,
        },
        "qualification": {
            "nominal_params": {"restitution": 0.5},
            "seed": 1000,
        },
        "publish_artifacts": ["trajectory"],
        "evidence": {
            "artifact_name": "frames",
            "renderer": "isaac_sim_kit_rtx",
            "media_type": "application/json",
            "width": 960,
            "height": 720,
            "fps": 30,
            "min_frames": 16,
            "require_motion": True,
            "camera": {
                "position": [2.4, -2.4, 1.6],
                "target": [0.0, 0.0, 0.45],
            },
            "recording_artifact_name": "recording_usd",
            "playback_renderer": "ovrtx",
            "max_duration_seconds": 4.0,
            "num_sensor_updates": 32,
            "render_mode": "rt2",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkout-dir",
        type=Path,
        default=Path.home()
        / ".cache"
        / "physics-agent"
        / f"IsaacLab-{ISAACLAB_COMMIT[:12]}",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=Path.cwd() / "byor-isaaclab.json",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Clone and pin IsaacLab without creating/installing its uv environment.",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Do not run the post-install IsaacLab import smoke test.",
    )
    args = parser.parse_args()

    checkout = args.checkout_dir.expanduser().resolve()
    config_out = args.config_out.expanduser().resolve()
    adapter = Path(__file__).with_name("isaaclab_bounce_trial.py").resolve()
    if not args.skip_smoke:
        _require_eula_consent()
    _prepare_checkout(checkout)
    if not args.skip_install:
        _install(checkout)
    if not args.skip_smoke:
        _smoke(checkout)
    config_out.parent.mkdir(parents=True, exist_ok=True)
    config_out.write_text(
        json.dumps(_config(checkout, adapter), indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"IsaacLab commit: {ISAACLAB_COMMIT}")
    print(f"Runtime config: {config_out}")
    print()
    print(f"physics-agent tune-external {config_out} --output-dir output/byor")
    print(
        f"physics-agent refine-external {config_out} --output-dir output/byor-refine "
        '--user-prompt "make the cube bounce to about 0.55 metres"'
    )


if __name__ == "__main__":
    main()
