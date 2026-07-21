# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TuningBackend implementation for NVIDIA Newton.

Newton is installed through ``apps/physics_agent[newton]``, which enables
Newton's PyPI ``sim`` and ``importers`` extras. Unlike OvPhysX it lives in the
parent venv and does not require daemon isolation. The runtime cost is a Warp
kernel compile on first call — :meth:`NewtonBackend.warmup` flushes that
up-front so NL ``--user-prompt`` LLM calls don't burn money before a local
precondition check fails.

Architecture mirrors :class:`physics_agent.tuning.ovphysx_backend.OvPhysXBackend`:
both back-ends dispatch by ``scenario.name`` to the same scenario evaluator
modules (``drop_settle.py``, ``freeform.py``), passing themselves (or their
held simulator) as ``simulator=`` — the engine-agnostic
:class:`physics_agent.tuning.simulator.Simulator` protocol.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from .backend import ENGINE_NEWTON, validate_engine_supports_params
from .capabilities import BindingCapability, newton_mujoco_capabilities

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from .newton_simulator import NewtonSimulator
    from .types import Scenario


class NewtonBackend:
    """Forward-only adapter from TuningBackend → NewtonSimulator → scenarios.

    Same callback contract as :class:`OvPhysXBackend`:

    * ``judge_callback``: freeform-trial VLM judge over rendered recording
      frames. ``None`` → freeform falls back to programmatic-only scoring.
    * ``final_state_judge``: optional drop_settle end-state VLM verifier
      (only invoked when ``target.vlm_check != "off"``).
    """

    name = ENGINE_NEWTON

    def __init__(self) -> None:
        # Lazy-init on first evaluate so importing this module is cheap for
        # CLI --help etc. The simulator constructor is light but newton's
        # first kernel compile is slow — held back until ``evaluate``.
        self._simulator: NewtonSimulator | None = None
        self.judge_callback: Any | None = None
        self.final_state_judge: Any | None = None

    def _get_simulator(self) -> NewtonSimulator:
        if self._simulator is None:
            from .newton_simulator import NewtonSimulator

            self._simulator = NewtonSimulator()
        return self._simulator

    def warmup(self) -> None:
        """Eagerly probe Newton+Warp+CUDA before any paid LLM call.

        Mirrors :meth:`OvPhysXBackend.warmup`: on the ``user_prompt`` path
        the NL interpreter runs BEFORE the first ``evaluate``. Warmup
        guarantees a clean install-hint error if newton or CUDA is missing,
        rather than a cryptic failure after the LLM bill has been spent.

        Repeated calls deliberately re-run the lightweight probe so callers can
        surface a fresh local precondition error after environment changes.
        """
        self._get_simulator().warmup()

    def tuning_capabilities(self) -> tuple[BindingCapability, ...]:
        """Newton MuJoCo consumes contact stiffness/damping, not restitution."""
        return newton_mujoco_capabilities()

    def evaluate(
        self,
        params: dict[str, float],
        scenario: Scenario,
        physics_usd: Path,
        *,
        seed: int,
    ) -> dict[str, Any]:
        """Run one trial. Dispatches to the scenario evaluator and returns
        the evaluator's result dict (runner consumes the ``score`` key)."""
        from physics_agent.tuning.scenario_resolution import (
            get_resolved_bindings,
            resolve_scenario_bindings,
        )
        from physics_agent.tuning.scenarios import resolve

        validate_engine_supports_params(self.name, scenario)
        if get_resolved_bindings(scenario) is None:
            scenario = resolve_scenario_bindings(
                scenario,
                physics_usd=physics_usd,
                backend=self,
            )
        evaluator = resolve(scenario.name)
        kwargs: dict[str, Any] = {"simulator": self._get_simulator()}
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
        """Optional explicit shutdown. Newton has no persistent state
        outside Warp's kernel cache so this is essentially a no-op; we
        drop the simulator handle for symmetry with OvPhysXBackend."""
        if self._simulator is not None:
            self._simulator.shutdown()
            self._simulator = None
