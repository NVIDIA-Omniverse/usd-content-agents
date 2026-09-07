# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Helpers for wrapper-orchestrated external refinement sessions.

An external wrapper (for example the agentic ``content-workflow-cli``
refine-external workflow) replaces :func:`run_external_refine`'s built-in VLM
judge and LLM refiner with its own outer loop while reusing the exact same
inner mechanics: per-iteration active-search derivation and qualification
approval validation. These helpers expose those two mechanics publicly so a
wrapper never has to import private runner symbols or re-implement the spec
validation rules.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from physics_agent.tuning.types import TunableParam

from .fingerprint import build_runtime_fingerprint
from .runner import _validate_approved_qualification
from .types import ExternalTuneSpec


def derive_iteration_spec(
    spec: ExternalTuneSpec,
    *,
    active_search: Mapping[str, Mapping[str, float]],
    iteration: int,
) -> ExternalTuneSpec:
    """Return the spec for one refinement iteration's fixed tuning run.

    ``active_search`` maps parameter names to ``{"min": ..., "max": ...}``
    bounds, exactly as :mod:`physics_agent.tuning.external.refine` records in
    each iteration's ``search.json``. Every named parameter must belong to the
    qualified parameter catalog; numeric types are copied from the catalog so
    an integer parameter cannot silently become continuous. The optimizer
    seed schedule matches the built-in refine loop: ``seed + iteration - 1``
    with the replica seed pinned to the base spec's schedule.

    Raises ``ValueError`` for an unknown parameter, malformed bounds, or an
    iteration below 1. Full contract validation (min < max, catalog
    consistency) happens in ``ExternalTuneSpec.__post_init__``.
    """

    if iteration < 1:
        raise ValueError(f"iteration must be at least 1, got {iteration}")
    if not active_search:
        raise ValueError("active_search must name at least one parameter")
    catalog = {parameter.name: parameter for parameter in spec.parameter_catalog}
    params: list[TunableParam] = []
    for name in active_search:
        bounds = active_search[name]
        qualified = catalog.get(name)
        if qualified is None:
            known = ", ".join(sorted(catalog))
            raise ValueError(
                f"active_search parameter {name!r} is not in the qualified "
                f"parameter catalog ({known})"
            )
        if not isinstance(bounds, Mapping) or {"min", "max"} - set(bounds):
            raise ValueError(
                f"active_search parameter {name!r} must provide 'min' and 'max'"
            )
        try:
            min_value = float(bounds["min"])
            max_value = float(bounds["max"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"active_search parameter {name!r} bounds must be numbers"
            ) from exc
        params.append(
            TunableParam(
                name=name,
                min_value=min_value,
                max_value=max_value,
                integer=qualified.integer,
            )
        )
    replica_seed = (
        spec.optimizer.seed
        if spec.optimizer.replica_seed is None
        else spec.optimizer.replica_seed
    )
    iteration_optimizer = replace(
        spec.optimizer,
        seed=spec.optimizer.seed + iteration - 1,
        replica_seed=replica_seed,
    )
    return replace(spec, params=tuple(params), optimizer=iteration_optimizer)


def validate_qualification_approval(
    spec: ExternalTuneSpec,
    *,
    qualification_dir: Path,
    approval_digest: str,
) -> Path:
    """Validate an approval digest against the stored qualification.

    Re-fingerprints the runtime and checks the approval digest, evidence
    bytes, and runtime identity exactly as a tuning run would, without
    mutating anything. Returns the ``qualification.json`` path on success and
    raises ``ValueError`` (with the runner's diagnostic message) on any
    mismatch, so a wrapper can fail fast before starting an agent session.
    """

    fingerprint = build_runtime_fingerprint(spec)
    try:
        _, qualification_path, _, _ = _validate_approved_qualification(
            spec=spec,
            qualification_dir=qualification_dir,
            approval_digest=approval_digest,
            fingerprint=fingerprint,
        )
    except Exception as exc:
        raise ValueError(str(exc)) from exc
    return qualification_path


__all__ = [
    "derive_iteration_spec",
    "validate_qualification_approval",
]
