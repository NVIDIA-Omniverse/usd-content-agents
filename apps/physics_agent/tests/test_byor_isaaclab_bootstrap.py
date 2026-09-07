# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the reproducible IsaacLab BYOR bootstrap."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import ModuleType

import pytest


def _load_bootstrap() -> ModuleType:
    path = Path(__file__).parents[1] / "examples" / "byor_isaaclab" / "bootstrap.py"
    spec = importlib.util.spec_from_file_location("byor_isaaclab_bootstrap", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_install_activates_new_uv_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    checkout.mkdir()
    calls: list[tuple[list[str], dict[str, str]]] = []

    def capture(
        argv: list[str],
        *,
        cwd: Path | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        assert cwd == checkout
        assert environment is not None
        calls.append((argv, dict(environment)))

    monkeypatch.setattr(bootstrap, "_run", capture)
    monkeypatch.setenv("VIRTUAL_ENV", "/unrelated/active-env")

    bootstrap._install(checkout)

    assert calls[0][0] == [str(checkout / "isaaclab.sh"), "--uv", "env_isaaclab"]
    assert "VIRTUAL_ENV" not in calls[0][1]
    assert calls[1][0][:3] == ["uv", "pip", "install"]
    assert f"isaacsim[all,extscache]=={bootstrap.ISAACSIM_VERSION}" in calls[1][0]
    assert calls[2][0] == [str(checkout / "isaaclab.sh"), "--install", "none"]
    assert calls[1][1]["VIRTUAL_ENV"] == str(checkout / "env_isaaclab")
    executable_dir = "Scripts" if os.name == "nt" else "bin"
    assert calls[1][1]["PATH"].split(os.pathsep)[0] == str(
        checkout / "env_isaaclab" / executable_dir
    )


def test_eula_consent_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    bootstrap = _load_bootstrap()
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)

    with pytest.raises(RuntimeError, match="OMNI_KIT_ACCEPT_EULA=YES"):
        bootstrap._require_eula_consent()

    monkeypatch.setenv("OMNI_KIT_ACCEPT_EULA", "YES")
    bootstrap._require_eula_consent()


def test_prepare_checkout_skips_dirty_check_for_fresh_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    commands: list[tuple[list[str], Path | None]] = []
    revisions = iter(["previous", bootstrap.ISAACLAB_COMMIT])

    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda argv, *, cwd=None, environment=None: commands.append((argv, cwd)),
    )

    def output(argv: list[str], *, cwd: Path | None = None) -> str:
        assert cwd == checkout
        assert argv != ["git", "status", "--porcelain"]
        assert argv == ["git", "rev-parse", "HEAD"]
        return next(revisions)

    monkeypatch.setattr(bootstrap, "_output", output)

    bootstrap._prepare_checkout(checkout)

    assert commands[0][0][:4] == ["git", "clone", "--filter=blob:none", "--no-checkout"]
    assert commands[1] == (
        ["git", "fetch", "origin", bootstrap.ISAACLAB_COMMIT],
        checkout,
    )
    assert commands[2] == (
        ["git", "checkout", "--detach", bootstrap.ISAACLAB_COMMIT],
        checkout,
    )


def test_prepare_checkout_materializes_fresh_clone_already_at_pin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    commands: list[tuple[list[str], Path | None]] = []
    revisions = iter([bootstrap.ISAACLAB_COMMIT, bootstrap.ISAACLAB_COMMIT])

    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda argv, *, cwd=None, environment=None: commands.append((argv, cwd)),
    )

    def output(_argv: list[str], *, cwd: Path) -> str:
        assert cwd == checkout
        return next(revisions)

    monkeypatch.setattr(bootstrap, "_output", output)

    bootstrap._prepare_checkout(checkout)

    assert commands[0][0][:4] == ["git", "clone", "--filter=blob:none", "--no-checkout"]
    assert commands[1] == (
        ["git", "checkout", "--detach", bootstrap.ISAACLAB_COMMIT],
        checkout,
    )
    assert not any(command[:2] == ["git", "fetch"] for command, _cwd in commands)


def test_prepare_checkout_rejects_dirty_existing_clone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    (checkout / ".git").mkdir(parents=True)
    monkeypatch.setattr(bootstrap, "_output", lambda *_args, **_kwargs: " M source.py")

    with pytest.raises(RuntimeError, match="dirty IsaacLab checkout"):
        bootstrap._prepare_checkout(checkout)


def test_generated_config_uses_headless_default_and_explicit_consent(
    tmp_path: Path,
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    adapter = tmp_path / "trial.py"

    config = bootstrap._config(checkout, adapter)
    runtime = config["runtime"]
    evidence = config["evidence"]

    assert runtime["extra_args"] == []
    assert runtime["pass_env"] == ["OMNI_KIT_ACCEPT_EULA", "PRIVACY_CONSENT"]
    assert runtime["fingerprint_paths"] == [
        str(checkout / "source" / "isaaclab"),
        str(checkout / "source" / "isaaclab_physx"),
        str(adapter),
    ]
    assert evidence["renderer"] == "isaac_sim_kit_rtx"
    assert evidence["recording_artifact_name"] == "recording_usd"
    assert evidence["playback_renderer"] == "ovrtx"
    assert evidence["max_duration_seconds"] == 4.0
    assert config["publish_artifacts"] == ["trajectory"]


def test_smoke_launches_and_closes_kit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bootstrap = _load_bootstrap()
    captured: list[list[str]] = []
    monkeypatch.setattr(
        bootstrap,
        "_run",
        lambda argv, **_kwargs: captured.append(argv),
    )

    bootstrap._smoke(tmp_path)

    assert "AppLauncher()" in captured[0][3]
    assert "launcher.app.close()" in captured[0][3]


def test_bootstrap_prints_supported_tuning_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bootstrap = _load_bootstrap()
    checkout = tmp_path / "IsaacLab"
    config_out = tmp_path / "runtime.json"
    monkeypatch.setattr(bootstrap, "_prepare_checkout", lambda _checkout: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "bootstrap.py",
            "--checkout-dir",
            str(checkout),
            "--config-out",
            str(config_out),
            "--skip-install",
            "--skip-smoke",
        ],
    )

    bootstrap.main()

    output = capsys.readouterr().out
    assert "physics-agent tune-external" in output
    assert "physics-agent refine-external" in output
