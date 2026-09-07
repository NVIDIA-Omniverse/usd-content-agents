# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the backend protocol + fake backend determinism + ovphysx loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from physics_agent.tuning import backend as backend_mod
from physics_agent.tuning.backend import (
    ENGINE_FAKE,
    ENGINE_OVPHYSX,
    SUPPORTED_ENGINES,
    FakeBackend,
    TuningBackend,
    get_backend,
    load_ovphysx_backend,
)
from physics_agent.tuning.errors import OvPhysXUnavailableError
from physics_agent.tuning.scenario import parse_scenario


def _scenario():
    return parse_scenario(
        {
            "name": "drop_settle",
            "parameters": [
                {"name": "mass_scale", "min": 0.5, "max": 2.0},
                {"name": "static_friction", "min": 0.05, "max": 1.0},
            ],
        }
    )


def test_supported_engines_contains_fake_and_ovphysx() -> None:
    assert ENGINE_FAKE in SUPPORTED_ENGINES
    assert ENGINE_OVPHYSX in SUPPORTED_ENGINES


def test_get_backend_returns_fake_for_fake_engine() -> None:
    b = get_backend(ENGINE_FAKE)
    assert isinstance(b, FakeBackend)
    assert isinstance(b, TuningBackend)


def test_get_backend_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError, match="Unknown engine"):
        get_backend("mujoco")


def test_fake_backend_is_deterministic() -> None:
    sc = _scenario()
    backend = FakeBackend()
    physics_usd = Path("/tmp/dummy.usda")
    a = backend.evaluate(
        params={"mass_scale": 1.2, "static_friction": 0.5},
        scenario=sc,
        physics_usd=physics_usd,
        seed=42,
    )
    b = backend.evaluate(
        params={"mass_scale": 1.2, "static_friction": 0.5},
        scenario=sc,
        physics_usd=physics_usd,
        seed=42,
    )
    assert a == b
    assert "score" in a
    assert isinstance(a["score"], float)
    assert a["objective_value"] == a["score"]


def test_fake_backend_score_minimised_at_seeded_optimum() -> None:
    sc = _scenario()
    backend = FakeBackend()
    physics_usd = Path("/tmp/dummy.usda")
    # The seeded "true optimum" is recorded in target_params; evaluating it
    # exactly should give score == 0.
    res = backend.evaluate(
        params={"mass_scale": 1.0, "static_friction": 0.5},
        scenario=sc,
        physics_usd=physics_usd,
        seed=99,
    )
    optimum = res["target_params"]
    res_at_optimum = backend.evaluate(
        params=optimum,
        scenario=sc,
        physics_usd=physics_usd,
        seed=99,
    )
    assert res_at_optimum["score"] == pytest.approx(0.0, abs=1e-12)
    assert res_at_optimum["objective_value"] == res_at_optimum["score"]
    assert res["score"] > 0.0


def test_load_ovphysx_does_not_import_ovphysx_in_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The daemon-isolation contract: the parent process must NOT import
    ``ovphysx`` when ``--engine ovphysx`` is selected. ovphysx ships a
    bundled OpenUSD that conflicts with the parent's OpenUSD provider, so any
    parent-side import either crashes or rejects daemon-only installs (where ovphysx
    lives only in the daemon venv). This test pins the contract by
    blocking ``ovphysx`` at import time and verifying the loader still
    succeeds — the daemon's startup handshake is the authoritative
    availability gate.
    """
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        if name == "ovphysx" or name.startswith("ovphysx."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # Should succeed: load_ovphysx_backend now only imports the in-tree
    # adapter module, never ovphysx itself.
    backend = load_ovphysx_backend()
    assert backend is not None


def test_load_ovphysx_raises_install_hint_when_adapter_module_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the in-tree ``ovphysx_backend`` adapter module fails to import
    (broken install of physics_agent itself), the canonical install hint
    is preserved so users know what to do."""
    import builtins
    import sys

    # Drop any cached version of the adapter module so the patched
    # ``__import__`` actually intercepts the next ``from .ovphysx_backend``.
    monkeypatch.delitem(
        sys.modules, "physics_agent.tuning.ovphysx_backend", raising=False
    )

    real_import = builtins.__import__

    def fake_import(name: str, *args, **kwargs):
        # Both the absolute form (importlib spec) and the relative form
        # (``from .ovphysx_backend import OvPhysXBackend`` calls
        # ``__import__("ovphysx_backend", level=1, ...)``) need to fail
        # so the loader's ``except ImportError`` branch fires.
        if name in ("physics_agent.tuning.ovphysx_backend", "ovphysx_backend"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(OvPhysXUnavailableError) as ei:
        load_ovphysx_backend()
    msg = str(ei.value)
    assert "OvPhysX backend requires the tuning extra" in msg
    assert 'uv pip install -e "apps/physics_agent[tuning]"' in msg


def test_get_backend_ovphysx_unavailable_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """get_backend('ovphysx') must propagate OvPhysXUnavailableError."""
    monkeypatch.setattr(
        backend_mod,
        "load_ovphysx_backend",
        lambda: (_ for _ in ()).throw(OvPhysXUnavailableError()),
    )
    with pytest.raises(OvPhysXUnavailableError) as ei:
        get_backend(ENGINE_OVPHYSX)
    assert "OvPhysX backend requires the tuning extra" in str(ei.value)
