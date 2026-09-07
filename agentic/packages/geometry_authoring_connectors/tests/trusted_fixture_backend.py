# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic constants-only backend for connector tests.

This backend never evaluates the request prompt or source code. It writes fixed
fixture bytes and is accepted only through the worker runner's explicit test opt-in.
"""

from __future__ import annotations

from pathlib import Path

from geometry_authoring_connectors import (
    AuthoringRequest,
    Build123dWorkerResult,
    WirePart,
    WireSemanticParameter,
    WireVerificationAssertion,
    WorkerArtifact,
    WorkerIsolationKind,
)

TRUSTED_STEP = b"""ISO-10303-21;
HEADER;
FILE_DESCRIPTION(('trusted fixture'),'2;1');
ENDSEC;
DATA;
ENDSEC;
END-ISO-10303-21;
"""
TRUSTED_SOURCE = b"""# Trusted fixture provenance; never evaluated by the test backend.
from build123d import Box
result = Box(10, 20, 30)
"""


class TrustedFixtureBackend:
    isolation_kind: WorkerIsolationKind = "trusted-test-fixture"

    def execute(
        self,
        request: AuthoringRequest,
        *,
        workspace: Path,
    ) -> Build123dWorkerResult:
        del request
        source = workspace / "trusted_fixture.py"
        geometry = workspace / "trusted_fixture.step"
        source.write_bytes(TRUSTED_SOURCE)
        geometry.write_bytes(TRUSTED_STEP)
        return Build123dWorkerResult(
            provider_version="trusted-fixture-1",
            source_revision="trusted-fixture-revision-1",
            units="millimeter",
            up_axis="Z",
            artifacts=(
                WorkerArtifact(
                    path=source,
                    filename=source.name,
                    role="native_source",
                    media_type="text/x-python",
                ),
                WorkerArtifact(
                    path=geometry,
                    filename=geometry.name,
                    role="cad_geometry",
                    media_type="model/step",
                ),
            ),
            parts=(
                WirePart(
                    part_id="body",
                    name="Body",
                    artifact_filenames=(geometry.name,),
                ),
            ),
            parameters=(
                WireSemanticParameter(
                    name="width_mm",
                    value=10.0,
                    unit="mm",
                    minimum=5.0,
                    maximum=100.0,
                    step=1.0,
                    semantic_role="overall_width",
                    effects=("exact_geometry",),
                    affects=("body",),
                ),
            ),
            verification_assertions=(
                WireVerificationAssertion(
                    assertion_id="fixture-build",
                    status="passed",
                    summary="Trusted fixture geometry was emitted.",
                ),
            ),
            metadata={"fixture": True},
        )
