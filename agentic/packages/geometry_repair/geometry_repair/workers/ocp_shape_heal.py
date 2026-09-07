# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded OCP/OCCT native CAD healing worker."""

from __future__ import annotations

from pathlib import Path

from ..brep import BREP_SUFFIXES, heal_brep, ocp_available
from ..models import RepairOperation
from .base import WorkerResult


class OcpShapeHealWorker:
    """Apply ShapeFix with source-scale tolerance growth limits."""

    name = "ocp_shape_heal"
    operations = frozenset({"brep_check", "shape_fix", "bounded_sewing"})

    def available(self) -> tuple[bool, str | None]:
        return ocp_available()

    def execute(
        self,
        *,
        source: Path,
        output: Path,
        operation: RepairOperation,
    ) -> WorkerResult:
        if source.suffix.lower() not in BREP_SUFFIXES:
            return WorkerResult(
                status="unavailable",
                failures=["OCP shape healing requires STEP, IGES, or BREP source"],
            )
        precision = float(operation.parameters.get("precision", 1e-6))
        maximum_tolerance = float(operation.parameters.get("maximum_tolerance", precision * 10.0))
        sew = bool(operation.parameters.get("sew", True))
        if maximum_tolerance < precision or maximum_tolerance > precision * 100.0:
            return WorkerResult(
                status="failed",
                failures=["maximum_tolerance must be between precision and 100x precision"],
            )
        try:
            report = heal_brep(
                source,
                output,
                precision=precision,
                maximum_tolerance=maximum_tolerance,
                sew=sew,
            )
        except Exception as exc:
            return WorkerResult(
                status="failed",
                failures=[f"{type(exc).__name__}: {exc}"],
            )
        return WorkerResult(
            status="completed",
            output_path=str(output),
            changed=True,
            operations=[*(["ocp_bounded_sewing"] if sew else []), "ocp_shape_fix"],
            metadata=report,
        )
