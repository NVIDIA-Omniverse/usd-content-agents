# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TuningBackend implementation for ovphysx via a daemon subprocess.

PR #43 originally shipped a ``getattr(ovphysx, f"run_{scenario.name}")``
adapter, but OvPhysX does not provide those scenario helpers. This module
replaces that broken adapter:

* ovphysx is hosted in its own venv (``~/.cache/wu/ovphysx_venv``)
  spawned as a long-running JSON-line subprocess by
  :class:`world_understanding.functions.physics.ovphysx_daemon._OvPhysXDaemon`.
* Per scenario, we dispatch to the corresponding evaluator module in
  ``physics_agent.tuning.scenarios`` (``drop_settle.py`` or
  ``freeform.py``). Each evaluator authors its scene USD with pxr in
  the parent process, sends one daemon ``evaluate`` request, gets a
  trajectory back, writes a time-sampled ``recording.usd``, and
  returns the score the runner expects.

Per-trial state reset (USD release + binding teardown) is handled by
the daemon — the evaluator just calls ``daemon.evaluate(...)`` and the
daemon enforces the contract.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backend import ENGINE_OVPHYSX, validate_engine_supports_params
from .capabilities import BindingCapability, usd_physics_capabilities

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .types import Scenario


class OvPhysXBackend:
    """Forward-only adapter from TuningBackend → ovphysx daemon evaluators.

    The daemon is created lazily on the first ``evaluate`` call and
    reused for the life of the backend instance (and therefore the life
    of the tune session). Two callbacks may be set externally
    (typically by ``backend.load_ovphysx_backend``):

    * ``judge_callback``: a freeform-trial VLM judge over rendered
      recording frames. ``None`` → freeform falls back to
      programmatic-only scoring.
    * ``final_state_judge``: an optional drop_settle end-state VLM
      verifier (only invoked when ``target.vlm_check != "off"``).
    """

    name = ENGINE_OVPHYSX

    def __init__(self) -> None:
        # Lazy-init on first evaluate so importing this module is cheap
        # for code paths that never run a tune (CLI --help, etc.).
        self._daemon: Any | None = None
        self.judge_callback: Any | None = None
        self.final_state_judge: Any | None = None

    def _get_daemon(self) -> Any:
        if self._daemon is None:
            from world_understanding.functions.physics.ovphysx_daemon import (
                _OvPhysXDaemon,
            )

            self._daemon = _OvPhysXDaemon()
        return self._daemon

    def warmup(self) -> None:
        """Eager-start the daemon subprocess.

        Round 14 (Codex CX P2#3): on the ``user_prompt`` path,
        ``run_tune`` calls the NL interpreter (a paid LLM request) BEFORE
        the first ``evaluate``. Without a warmup, a box missing the
        ovphysx venv only surfaces ``OvPhysXDaemonUnavailableError`` on
        the first trial — after the LLM bill has already been spent.
        Calling ``warmup`` right after backend construction trades a
        single eager daemon startup for a guarantee that the local
        precondition fails fast.

        Safe to call multiple times; the underlying daemon's
        ``ensure_running`` is idempotent.
        """
        self._get_daemon().ensure_running()

    def tuning_capabilities(self) -> tuple[BindingCapability, ...]:
        """OvPhysX consumes authored UsdPhysics mass/material attributes."""
        return usd_physics_capabilities()

    def evaluate(
        self,
        params: dict[str, float],
        scenario: Scenario,
        physics_usd: Path,
        *,
        seed: int,
    ) -> dict[str, Any]:
        """Run one trial. Dispatches by ``scenario.name`` to the matching
        evaluator module.

        Returns the evaluator's result dict; runner code consumes the
        ``score`` key.
        """
        from physics_agent.tuning.scenario_resolution import (
            get_resolved_bindings,
            resolve_scenario_bindings,
        )
        from physics_agent.tuning.scenarios import resolve

        validate_engine_supports_params(self.name, scenario)
        evaluator = resolve(scenario.name)
        if get_resolved_bindings(scenario) is None:
            scenario = resolve_scenario_bindings(
                scenario,
                physics_usd=physics_usd,
                backend=self,
            )
        # The daemon class structurally satisfies the engine-agnostic
        # ``physics_agent.tuning.simulator.Simulator`` protocol; we hand
        # it over to the scenario evaluator as ``simulator=`` so both
        # OvPhysX (here) and Newton can reuse the same evaluator code.
        kwargs: dict[str, Any] = {"simulator": self._get_daemon()}
        if scenario.name == "freeform":
            kwargs["judge_callback"] = self.judge_callback
        elif scenario.name == "drop_settle":
            kwargs["final_state_judge"] = self.final_state_judge
        return evaluator(
            params=dict(params),
            scenario=scenario,
            physics_usd=Path(physics_usd),
            seed=int(seed),
            **kwargs,
        )

    def shutdown(self) -> None:
        """Optional explicit shutdown — daemon also has an atexit hook."""
        if self._daemon is not None:
            self._daemon.shutdown()
            self._daemon = None


__all__ = ["OvPhysXBackend"]
