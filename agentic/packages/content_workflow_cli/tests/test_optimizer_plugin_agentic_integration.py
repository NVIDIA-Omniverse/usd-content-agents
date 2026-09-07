# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic pipeline integration tests for plugin optimizer support.

Verifies that the agentic pipeline accepts a registered plugin optimizer when
it is available, rejects it with the plugin's own message when unavailable,
and rejects completely unknown optimizer names — all through the public
optimizer registry, without depending on any specific optimizer extension.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from content_workflow_cli.cli import main
from content_workflow_cli.tuning_broker import check_tune_optimizer_available

_FAKE_PLUGIN_NAME = "fake-tuner"
_FAKE_UNAVAILABLE_MESSAGE = (
    "fake-tuner requires the fake-tuner-client package and "
    "FAKE_TUNER_API_KEY to be set."
)


def _register_fake_plugin(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    """Inject a fake optimizer plugin into the public registry."""
    from world_understanding.optimization import registry as reg

    fake_plugin = reg.OptimizerPlugin(
        runner=lambda *a, **kw: None,
        is_available=lambda: available,
        unavailable_message=_FAKE_UNAVAILABLE_MESSAGE,
    )
    monkeypatch.setitem(reg._optimizer_plugins, _FAKE_PLUGIN_NAME, fake_plugin)
    monkeypatch.setattr(reg, "_optimizer_plugins_scanned", True)


def test_builtin_optimizers_always_pass() -> None:
    for name in ("auto", "botorch", "random", "cma-es"):
        check_tune_optimizer_available(name)


def test_plugin_optimizer_passes_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_fake_plugin(monkeypatch, available=True)
    check_tune_optimizer_available(_FAKE_PLUGIN_NAME)


def test_plugin_optimizer_fails_with_plugin_message_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _register_fake_plugin(monkeypatch, available=False)
    with pytest.raises(ValueError, match="fake-tuner-client"):
        check_tune_optimizer_available(_FAKE_PLUGIN_NAME)


def test_unknown_optimizer_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from world_understanding.optimization import registry as reg

    monkeypatch.setattr(reg, "_optimizer_plugins_scanned", True)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        check_tune_optimizer_available("does-not-exist")


def test_physics_apply_dry_run_accepts_plugin_optimizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A registered plugin optimizer must pass argparse and config validation.

    Guards the removal of ``choices=(...)`` on ``--optimizer``: restoring the
    built-in-only choices tuple would make argparse reject a plugin name with
    exit code 2 before validation is ever reached.
    """
    _register_fake_plugin(monkeypatch, available=True)
    usd = tmp_path / "asset.usdc"
    usd.write_text("placeholder", encoding="utf-8")
    scenario = tmp_path / "drop_settle.yaml"
    scenario.write_text("name: drop_settle\n", encoding="utf-8")
    run_dir = tmp_path / "physics-plugin-run"

    exit_code = main(
        [
            "physics",
            "apply",
            "--usd",
            str(usd),
            "--repo-root",
            str(Path(__file__).resolve().parents[4]),
            "--output-dir",
            str(run_dir),
            "--behavior-prompt",
            "make the asset settle without sliding",
            "--scenario",
            str(scenario),
            "--refine",
            "--tune-engine",
            "newton",
            "--optimizer",
            _FAKE_PLUGIN_NAME,
            "--dry-run",
        ]
    )

    assert exit_code == 0
    contract = json.loads(
        (run_dir / "raw" / "physics_agentic_contract.json").read_text(encoding="utf-8")
    )
    assert contract["tuning"]["optimizer"] == _FAKE_PLUGIN_NAME
