# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Immutable source capture for one admitted articulation-v2 static run.

This boundary validates the #1015 plan against already-loaded #1012 authority
before it asks a resolver for the selected source request.  It neither
interprets joint semantics nor creates an authoring or evidence destination.
A resolver supplies an already size-bound opaque capture; this module detaches
it, derives the Gate 3 v3 inventory from the retained bytes, and compares that
inventory with the admitted selector lineage.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from world_understanding.functions.physics.joint_rigger import ArtifactIdentityV1
from world_understanding.functions.physics.joint_rigger.reference import (
    retain_usd_artifact_inspection,
)
from world_understanding.utils.captured_artifacts import (
    CapturedArtifactError,
    CapturedOpaqueArtifactResolver,
    CapturedOpaqueFile,
    OpaqueArtifactRequest,
    capture_resolved_opaque_file,
)

from joint_agent.articulation_v2_static_artifact_identity import (
    ARTICULATION_V2_STATIC_ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION,
    ArticulationV2StaticArtifactIdentityError,
    ArticulationV2StaticGate3DependencyManifestV1,
    gate3_v3_dependency_manifest_from_capture,
)
from joint_agent.articulation_v2_static_run_plan import (
    ArticulationV2StaticControlledDistanceRunPlanV2,
    ArticulationV2StaticControlledDistanceSelectedSourceV2,
    ArticulationV2StaticRunPlanV1,
    ArticulationV2StaticSelectedSourceV1,
    canonical_articulation_v2_static_controlled_distance_run_plan_bytes,
    canonical_articulation_v2_static_run_plan_bytes,
    validate_articulation_v2_static_controlled_distance_run_plan,
    validate_articulation_v2_static_run_plan,
)
from joint_agent.articulation_v2_static_semantics import (
    ArticulationV2StaticDependencyBundleV3,
)
from joint_agent.capability_manifest import (
    CapabilityManifestError,
    LoadedCapabilityManifest,
)


class ArticulationV2StaticSourceCaptureError(RuntimeError):
    """Fail-closed source capture error for the 0.6 static lane."""


@dataclass(frozen=True, slots=True)
class ArticulationV2StaticSourceCaptureRequestV1:
    """The only source request a resolver may use for one admitted plan."""

    locator: str
    root_sha256: str
    max_bytes: int


@runtime_checkable
class ArticulationV2StaticSourceCaptureResolver(
    CapturedOpaqueArtifactResolver,
    Protocol,
):
    """Resolver extension that supplies a size-bound opaque request.

    The run plan deliberately owns the maximum, root hash, and locator but not
    a transport-specific byte length.  The provider supplies the exact length
    in an ``OpaqueArtifactRequest`` which is then checked before capture.
    """

    def source_request(
        self,
        request: ArticulationV2StaticSourceCaptureRequestV1,
    ) -> OpaqueArtifactRequest: ...


@dataclass(frozen=True, slots=True)
class CapturedArticulationV2StaticSourceV1:
    """One private source snapshot held only for the surrounding context."""

    source: ArticulationV2StaticSelectedSourceV1
    capture: CapturedOpaqueFile
    gate3_dependency_manifest: ArticulationV2StaticGate3DependencyManifestV1

    @property
    def gate3_dependency_bundle(self) -> ArticulationV2StaticDependencyBundleV3:
        """Return the admitted-v3-shaped identity computed from retained bytes."""

        return ArticulationV2StaticDependencyBundleV3(
            schema_version=ARTICULATION_V2_STATIC_ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION,
            sha256=self.gate3_dependency_manifest.sha256,
            entry_count=len(self.gate3_dependency_manifest.entries),
        )

    def require_intact(self) -> None:
        """Recheck the source snapshot before a consumer reads it."""

        self.capture.require_intact()


@dataclass(frozen=True, slots=True)
class CapturedArticulationV2StaticControlledDistanceSourceV2:
    """One exact retained source-contract snapshot for controlled-distance v2."""

    source: ArticulationV2StaticControlledDistanceSelectedSourceV2
    capture: CapturedOpaqueFile
    source_artifact: ArtifactIdentityV1
    gate3_dependency_manifest: ArticulationV2StaticGate3DependencyManifestV1

    @property
    def gate3_dependency_bundle(self) -> ArticulationV2StaticDependencyBundleV3:
        return ArticulationV2StaticDependencyBundleV3(
            schema_version=ARTICULATION_V2_STATIC_ARTIFACT_DEPENDENCY_BUNDLE_SCHEMA_VERSION,
            sha256=self.gate3_dependency_manifest.sha256,
            entry_count=len(self.gate3_dependency_manifest.entries),
        )

    def require_intact(self) -> None:
        self.capture.require_intact()


@contextmanager
def capture_articulation_v2_static_source(
    authority: LoadedCapabilityManifest,
    plan: ArticulationV2StaticRunPlanV1,
    resolver: ArticulationV2StaticSourceCaptureResolver,
) -> Iterator[CapturedArticulationV2StaticSourceV1]:
    """Capture and verify the exact selector-scoped source named by ``plan``.

    ``authority`` is the already-loaded #1012 manifest used only to admit the
    #1015 plan before resolver I/O.  No writable evidence root is opened here
    or by any helper it calls.  The returned object is valid only inside this
    context, which keeps the captured descriptors private and immutable through
    the downstream authoring call.
    """

    canonical_plan = _revalidate_exact_plan(plan)
    try:
        validate_articulation_v2_static_run_plan(authority, canonical_plan)
    except CapabilityManifestError as exc:
        raise ArticulationV2StaticSourceCaptureError(
            "source_plan_unadmitted: source capture requires #1012 authority "
            "for the #1015 run plan"
        ) from exc
    if not isinstance(resolver, ArticulationV2StaticSourceCaptureResolver):
        raise TypeError("resolver must implement the static source-capture protocol")
    source = canonical_plan.run.selected_source
    request = ArticulationV2StaticSourceCaptureRequestV1(
        locator=source.locator,
        root_sha256=source.root_sha256,
        max_bytes=source.max_bytes,
    )
    admitted_locator = request.locator
    admitted_root_sha256 = request.root_sha256
    admitted_max_bytes = request.max_bytes
    opaque_request = resolver.source_request(request)
    _require_opaque_request_matches_source(
        opaque_request,
        locator=admitted_locator,
        root_sha256=admitted_root_sha256,
        max_bytes=admitted_max_bytes,
    )
    caller_exception = False
    try:
        with capture_resolved_opaque_file(resolver, opaque_request) as capture:
            _require_captured_source_matches(capture, source)
            manifest = gate3_v3_dependency_manifest_from_capture(
                capture,
                representation=source.representation,
            )
            expected = source.gate3_dependency_bundle
            if (
                manifest.sha256 != expected.sha256
                or len(manifest.entries) != expected.entry_count
            ):
                raise ArticulationV2StaticSourceCaptureError(
                    "source_dependency_closure_mismatch: retained dependency inventory "
                    "differs from the selector-admitted Gate 3 v3 closure"
                )
            retained = CapturedArticulationV2StaticSourceV1(
                source=source,
                capture=capture,
                gate3_dependency_manifest=manifest,
            )
            retained.require_intact()
            caller_exception = True
            yield retained
            caller_exception = False
            retained.require_intact()
    except CapturedArtifactError as exc:
        if caller_exception:
            raise
        raise ArticulationV2StaticSourceCaptureError(
            "source retained capture integrity check failed"
        ) from exc
    except ArticulationV2StaticArtifactIdentityError as exc:
        if not caller_exception and isinstance(exc.__cause__, CapturedArtifactError):
            raise ArticulationV2StaticSourceCaptureError(
                "source retained capture integrity check failed"
            ) from exc
        raise


@contextmanager
def capture_articulation_v2_static_controlled_distance_source(
    authority: LoadedCapabilityManifest,
    plan: ArticulationV2StaticControlledDistanceRunPlanV2,
    resolver: ArticulationV2StaticSourceCaptureResolver,
) -> Iterator[CapturedArticulationV2StaticControlledDistanceSourceV2]:
    """Capture the explicitly discriminated raw-USDA controlled-distance source."""

    canonical_plan = _revalidate_exact_controlled_distance_plan(plan)
    try:
        validate_articulation_v2_static_controlled_distance_run_plan(
            authority,
            canonical_plan,
        )
    except CapabilityManifestError as exc:
        raise ArticulationV2StaticSourceCaptureError(
            "source_plan_unadmitted: controlled-distance source capture requires "
            "the exact admitted v2 run plan"
        ) from exc
    if not isinstance(resolver, ArticulationV2StaticSourceCaptureResolver):
        raise TypeError("resolver must implement the static source-capture protocol")
    source = canonical_plan.run.selected_source
    request = ArticulationV2StaticSourceCaptureRequestV1(
        locator=source.locator,
        root_sha256=source.root_sha256,
        max_bytes=source.max_bytes,
    )
    admitted_locator = request.locator
    admitted_root_sha256 = request.root_sha256
    admitted_max_bytes = request.max_bytes
    opaque_request = resolver.source_request(request)
    _require_opaque_request_matches_source(
        opaque_request,
        locator=admitted_locator,
        root_sha256=admitted_root_sha256,
        max_bytes=admitted_max_bytes,
    )
    caller_exception = False
    try:
        with capture_resolved_opaque_file(resolver, opaque_request) as capture:
            _require_captured_source_matches_v2(capture, source)
            source_artifact = _controlled_distance_source_artifact(
                capture,
                uri=(
                    canonical_plan.run.selector.expected_joint_semantics.controlled_distance_contract.source_artifact.uri
                ),
            )
            gate3_dependency_manifest = gate3_v3_dependency_manifest_from_capture(
                capture,
                # The sibling semantic representation is not a USD container
                # representation. Its selected source contract requires USDA.
                representation="raw_usd",
            )
            expected_authorer = canonical_plan.run.selector.expected_joint_semantics.controlled_distance_contract.source_artifact
            expected = source.gate3_dependency_bundle
            if (
                source_artifact != expected_authorer
                or source_artifact.root_sha256 != source.root_sha256
                or gate3_dependency_manifest.sha256 != expected.sha256
                or len(gate3_dependency_manifest.entries) != expected.entry_count
            ):
                raise ArticulationV2StaticSourceCaptureError(
                    "source_dependency_closure_mismatch: retained dependency inventory "
                    "differs from the controlled-distance v2 authority"
                )
            retained = CapturedArticulationV2StaticControlledDistanceSourceV2(
                source=source,
                capture=capture,
                source_artifact=source_artifact,
                gate3_dependency_manifest=gate3_dependency_manifest,
            )
            retained.require_intact()
            caller_exception = True
            yield retained
            caller_exception = False
            retained.require_intact()
    except CapturedArtifactError as exc:
        if caller_exception:
            raise
        raise ArticulationV2StaticSourceCaptureError(
            "controlled-distance source retained capture integrity check failed"
        ) from exc
    except ArticulationV2StaticArtifactIdentityError as exc:
        if not caller_exception and isinstance(exc.__cause__, CapturedArtifactError):
            raise ArticulationV2StaticSourceCaptureError(
                "controlled-distance source retained capture integrity check failed"
            ) from exc
        raise


def _controlled_distance_source_artifact(
    capture: CapturedOpaqueFile,
    *,
    uri: str,
) -> ArtifactIdentityV1:
    """Recompute the admitted authorer-domain source identity from sealed bytes."""

    capture.require_intact()
    with tempfile.TemporaryDirectory(
        prefix="joint-agent-controlled-distance-source-"
    ) as root:
        path = Path(root) / "controlled_distance_contract.usda"
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        size = 0
        try:
            for chunk in capture.iter_chunks():
                view = memoryview(chunk)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError(
                            "controlled-distance source write made no progress"
                        )
                    view = view[written:]
                digest.update(chunk)
                size += len(chunk)
            os.fchmod(descriptor, 0o400)
        finally:
            os.close(descriptor)
        if size != capture.size_bytes or digest.hexdigest() != capture.sha256:
            raise ArticulationV2StaticSourceCaptureError(
                "controlled-distance source materialization differs from capture"
            )
        with retain_usd_artifact_inspection(
            path,
            uri=uri,
            expected_root_sha256=capture.sha256,
            recheck_source_content_on_exit=True,
        ) as inspection:
            identity = inspection.identity
    capture.require_intact()
    return identity


def _revalidate_exact_plan(
    plan: ArticulationV2StaticRunPlanV1,
) -> ArticulationV2StaticRunPlanV1:
    if type(plan) is not ArticulationV2StaticRunPlanV1:
        raise TypeError("plan must be an exact ArticulationV2StaticRunPlanV1")
    try:
        return ArticulationV2StaticRunPlanV1.model_validate_json(
            canonical_articulation_v2_static_run_plan_bytes(plan),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArticulationV2StaticSourceCaptureError(
            "source_plan_invalid: static source capture requires a canonical run plan"
        ) from exc


def _revalidate_exact_controlled_distance_plan(
    plan: ArticulationV2StaticControlledDistanceRunPlanV2,
) -> ArticulationV2StaticControlledDistanceRunPlanV2:
    if type(plan) is not ArticulationV2StaticControlledDistanceRunPlanV2:
        raise TypeError(
            "plan must be an exact ArticulationV2StaticControlledDistanceRunPlanV2"
        )
    try:
        return ArticulationV2StaticControlledDistanceRunPlanV2.model_validate_json(
            canonical_articulation_v2_static_controlled_distance_run_plan_bytes(plan),
            strict=True,
        )
    except (TypeError, ValueError) as exc:
        raise ArticulationV2StaticSourceCaptureError(
            "source_plan_invalid: controlled-distance source capture requires a "
            "canonical v2 run plan"
        ) from exc


def _require_opaque_request_matches_source(
    opaque: OpaqueArtifactRequest,
    *,
    locator: str,
    root_sha256: str,
    max_bytes: int,
) -> None:
    if type(opaque) is not OpaqueArtifactRequest:
        raise ArticulationV2StaticSourceCaptureError(
            "source_request_invalid: resolver did not return an OpaqueArtifactRequest"
        )
    if (
        opaque.uri != locator
        or opaque.sha256 != root_sha256
        or opaque.max_bytes != max_bytes
        or opaque.size_bytes > max_bytes
    ):
        raise ArticulationV2StaticSourceCaptureError(
            "source_request_mismatch: resolver request differs from the admitted source"
        )


def _require_captured_source_matches(
    capture: CapturedOpaqueFile,
    source: ArticulationV2StaticSelectedSourceV1,
) -> None:
    if type(capture) is not CapturedOpaqueFile:
        raise ArticulationV2StaticSourceCaptureError(
            "source_capture_invalid: resolver did not provide an exact captured file"
        )
    capture.require_intact()
    if (
        capture.uri != source.locator
        or capture.sha256 != source.root_sha256
        or capture.max_bytes != source.max_bytes
        or capture.size_bytes > source.max_bytes
    ):
        raise ArticulationV2StaticSourceCaptureError(
            "source_capture_mismatch: retained source differs from the admitted plan"
        )


def _require_captured_source_matches_v2(
    capture: CapturedOpaqueFile,
    source: ArticulationV2StaticControlledDistanceSelectedSourceV2,
) -> None:
    if type(capture) is not CapturedOpaqueFile:
        raise ArticulationV2StaticSourceCaptureError(
            "source_capture_invalid: resolver did not provide an exact captured file"
        )
    capture.require_intact()
    if (
        capture.uri != source.locator
        or capture.sha256 != source.root_sha256
        or capture.max_bytes != source.max_bytes
        or capture.size_bytes > source.max_bytes
    ):
        raise ArticulationV2StaticSourceCaptureError(
            "source_capture_mismatch: retained source differs from the admitted v2 plan"
        )


__all__ = [
    "ArticulationV2StaticSourceCaptureError",
    "ArticulationV2StaticSourceCaptureRequestV1",
    "ArticulationV2StaticSourceCaptureResolver",
    "CapturedArticulationV2StaticControlledDistanceSourceV2",
    "CapturedArticulationV2StaticSourceV1",
    "capture_articulation_v2_static_controlled_distance_source",
    "capture_articulation_v2_static_source",
]
