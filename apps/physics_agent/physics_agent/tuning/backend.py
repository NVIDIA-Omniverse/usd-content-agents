# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend protocol + factory for tuning.

The backend abstracts over "evaluate this candidate parameter set against a
physics simulator and return a scalar metric". Three implementations:

* :class:`FakeBackend` — deterministic, in-process; used by unit tests and the
  default ``--engine fake`` path. No external dependencies.
* OvPhysX — lazy-loaded via :func:`load_ovphysx_backend` when the user opts in
  with ``--engine ovphysx``. Runs through a daemon subprocess (separate venv)
  because its bundled OpenUSD conflicts with the parent's ``usd-core``.
* Newton — lazy-loaded via :func:`load_newton_backend` when the user opts in
  with ``--engine newton``. NVIDIA Newton (open-source GPU/Warp + MuJoCo-warp).
  Installable via the ``apps/physics_agent[newton]`` extra; no daemon needed
  because it runs in the parent venv.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .capabilities import (
    BindingCapability,
    capabilities_for_backend,
    usd_physics_capabilities,
)
from .errors import NewtonUnavailableError, OvPhysXUnavailableError, TuningError
from .types import Scenario

# Engine identifiers exposed to users via CLI / REST.
ENGINE_OVPHYSX = "ovphysx"
ENGINE_NEWTON = "newton"
ENGINE_FAKE = "fake"
SUPPORTED_ENGINES: tuple[str, ...] = (ENGINE_OVPHYSX, ENGINE_NEWTON, ENGINE_FAKE)

NEWTON_UNSUPPORTED_PARAM_REASONS: dict[str, str] = {
    "restitution": (
        "NewtonSimulator currently runs SolverMuJoCo with Newton contacts, "
        "and that path does not apply shape_material_restitution, so "
        "bouncy/max_bounce_height trials would be ineffective."
    ),
    "static_friction": (
        "The current Newton USD importer builds its effective contact "
        "friction from UsdPhysics.MaterialAPI.dynamicFriction, not "
        "staticFriction, so static_friction-only trials would be ineffective."
    ),
}


@runtime_checkable
class TuningBackend(Protocol):
    """Backend contract for evaluating one parameter set in simulation.

    Implementations must be deterministic for a given (params, seed) pair when
    used for unit tests. The runner enforces this only for the FakeBackend.
    """

    name: str

    def tuning_capabilities(self) -> tuple[BindingCapability, ...]:
        """Return backend-supported parameter binding capabilities."""
        ...  # pragma: no cover - protocol declaration

    def evaluate(
        self,
        params: dict[str, float],
        scenario: Scenario,
        physics_usd: Path,
        *,
        seed: int,
    ) -> dict[str, Any]:
        """Run one trial and return raw backend metrics.

        Required output keys:
            ``score`` (float, lower is better) — the optimization objective.

        Optional keys (passed through to ``trial.backend_metrics``):
            ``trajectory`` (path to a per-trial trajectory file),
            ``raw_log`` (path to a raw simulator log),
            anything scenario-specific.
        """
        ...  # pragma: no cover - protocol declaration


class FakeBackend:
    """Deterministic in-process backend for tests and CLI smoke runs.

    The objective is a smooth quadratic centered on a per-scenario / per-seed
    optimum that lives strictly inside the configured parameter bounds. A
    well-behaved optimizer should descend toward that optimum even with very
    small trial budgets, and the *exact* optimum is reproducible from the
    seed alone — that gives us strong, deterministic test signal without
    needing a real simulator.
    """

    name = ENGINE_FAKE

    def tuning_capabilities(self) -> tuple[BindingCapability, ...]:
        """Fake backend keeps the same USD patching surface as OvPhysX."""
        return usd_physics_capabilities()

    def evaluate(
        self,
        params: dict[str, float],
        scenario: Scenario,
        physics_usd: Path,
        *,
        seed: int,
    ) -> dict[str, Any]:
        # Per-param target = the midpoint of [min, max] biased by seed. This
        # makes the optimum reproducible for a (scenario, seed) pair.
        score = 0.0
        contribution: dict[str, float] = {}
        for tp in scenario.params:
            target = _scenario_target(tp.name, tp.min_value, tp.max_value, seed)
            value = float(params.get(tp.name, (tp.min_value + tp.max_value) / 2.0))
            scale = max(tp.max_value - tp.min_value, 1e-6)
            normalized = (value - target) / scale
            term = normalized * normalized
            score += term
            contribution[tp.name] = term
        return {
            "score": float(score),
            "target_params": {
                tp.name: _scenario_target(tp.name, tp.min_value, tp.max_value, seed)
                for tp in scenario.params
            },
            "per_param_squared_error": contribution,
            "physics_usd": str(physics_usd),
            "metric": scenario.metric,
        }


def _scenario_target(name: str, lo: float, hi: float, seed: int) -> float:
    """Pick a deterministic 'true optimum' inside [lo, hi] from name+seed.

    Hash-mixing keeps the target well inside the interval (uses the central
    50%) so an optimizer that simply samples the corners won't accidentally
    win, but still leaves enough room to converge in 30 trials.

    Uses :func:`hashlib.sha256` rather than the builtin :func:`hash` so the
    output is identical across Python interpreter restarts and ``PYTHONHASHSEED``
    settings (Python's ``hash()`` is randomised per-process for str/tuples
    since 3.3, which would silently break test reproducibility).
    """
    import hashlib

    digest_bytes = hashlib.sha256(f"{name}|{seed}".encode()).digest()
    digest = int.from_bytes(digest_bytes[:4], "big") % 10_000
    fraction = 0.25 + (digest / 10_000.0) * 0.5  # [0.25, 0.75]
    return lo + fraction * (hi - lo)


def validate_engine_supports_param_names(
    engine: str, param_names: Iterable[str]
) -> None:
    """Reject tunable parameter names that the chosen engine cannot apply."""
    requested = set(param_names)
    if not requested:
        return
    if engine not in SUPPORTED_ENGINES:
        raise ValueError(
            f"Unknown engine {engine!r}. Supported: {sorted(SUPPORTED_ENGINES)}"
        )

    capability_names = {c.param_name for c in capabilities_for_backend(engine)}
    unsupported = sorted(requested - capability_names)
    if not unsupported:
        return

    reasons: list[str] = []
    if engine == ENGINE_NEWTON:
        reasons.extend(
            reason
            for name, reason in NEWTON_UNSUPPORTED_PARAM_REASONS.items()
            if name in unsupported
        )
    if not reasons:
        supported = ", ".join(sorted(capability_names)) or "none"
        reasons.append(f"Supported parameters for engine {engine!r}: {supported}.")

    names = ", ".join(unsupported)
    if len(reasons) == 1:
        details = reasons[0]
    else:
        details = " ".join(
            f"Reason {idx}: {reason}" for idx, reason in enumerate(reasons, start=1)
        )
    hint = "or select an engine that supports them."
    if engine == ENGINE_NEWTON and set(unsupported) & {
        "restitution",
        "static_friction",
    }:
        hint = "or use --engine ovphysx for static-friction or restitution tuning."
    elif engine == ENGINE_OVPHYSX and set(unsupported) & {"contact_ke", "contact_kd"}:
        hint = "or use --engine newton for contact_ke/contact_kd bounce tuning."

    raise TuningError(
        f"Engine {engine!r} does not support tuning {names} yet. "
        f"{details} Remove unsupported parameters from scenario.parameters "
        f"{hint}"
    )


def validate_engine_supports_params(engine: str, scenario: Scenario) -> None:
    """Reject tunable parameters that the chosen engine cannot apply."""
    validate_engine_supports_param_names(
        engine, (param.name for param in scenario.params)
    )


def load_ovphysx_backend() -> TuningBackend:
    """Construct the OvPhysX backend adapter.

    The adapter forwards every trial through the daemon subprocess
    (:class:`world_understanding.functions.physics.ovphysx_daemon._OvPhysXDaemon`),
    which lives in its own venv (``WU_OVPHYSX_VENV_DIR``, default
    ``~/.cache/wu/ovphysx_venv``) precisely because ovphysx ships a
    bundled OpenUSD that conflicts with the parent's ``usd-core``. We
    therefore must NOT ``import ovphysx`` in the parent process — that
    would (a) defeat the daemon-isolation contract, (b) trigger the
    USD-version conflict the daemon is meant to avoid, and (c) reject
    correct daemon-only installations where the parent venv has no
    ovphysx at all. The daemon's startup handshake is the
    authoritative availability gate; it raises
    :class:`OvPhysXDaemonUnavailableError` with the canonical install
    hint when its venv is missing or empty.

    Raises:
        OvPhysXUnavailableError: when the in-tree backend adapter
            module cannot be imported. The daemon's own runtime errors
            surface separately on the first trial.
    """
    try:
        from .ovphysx_backend import OvPhysXBackend
    except ImportError as e:
        raise OvPhysXUnavailableError(
            f"{OvPhysXUnavailableError.DEFAULT_MESSAGE}\n(underlying import error: {e})"
        ) from e
    return OvPhysXBackend()


def load_newton_backend() -> TuningBackend:
    """Construct the Newton backend adapter.

    Unlike OvPhysX, Newton lives in the parent venv through the
    ``apps/physics_agent[newton]`` extra with ``sim``/``importers`` enabled, so
    we just import the in-tree adapter directly. The first ``evaluate`` call
    pays for Warp kernel compilation;
    :meth:`NewtonBackend.warmup` flushes that up-front so the NL interpreter
    LLM path doesn't burn money before a missing-CUDA failure surfaces.

    Raises:
        NewtonUnavailableError: when the ``newton`` package is not installed.
    """
    try:
        from .newton_backend import NewtonBackend
    except ImportError as e:
        raise NewtonUnavailableError(
            f"{NewtonUnavailableError.DEFAULT_MESSAGE}\n(underlying import error: {e})"
        ) from e
    return NewtonBackend()


def get_backend(engine: str) -> TuningBackend:
    """Resolve an engine name to a backend instance.

    Args:
        engine: One of :data:`SUPPORTED_ENGINES`.

    Raises:
        ValueError: for unknown engine names.
        OvPhysXUnavailableError: when ``ovphysx`` is requested but missing.
        NewtonUnavailableError: when ``newton`` is requested but missing.
    """
    if engine == ENGINE_FAKE:
        return FakeBackend()
    if engine == ENGINE_OVPHYSX:
        return load_ovphysx_backend()
    if engine == ENGINE_NEWTON:
        return load_newton_backend()
    raise ValueError(
        f"Unknown engine {engine!r}. Supported: {sorted(SUPPORTED_ENGINES)}"
    )


__all__ = [
    "ENGINE_FAKE",
    "ENGINE_NEWTON",
    "ENGINE_OVPHYSX",
    "NEWTON_UNSUPPORTED_PARAM_REASONS",
    "SUPPORTED_ENGINES",
    "FakeBackend",
    "TuningBackend",
    "get_backend",
    "load_newton_backend",
    "load_ovphysx_backend",
    "validate_engine_supports_param_names",
    "validate_engine_supports_params",
]
