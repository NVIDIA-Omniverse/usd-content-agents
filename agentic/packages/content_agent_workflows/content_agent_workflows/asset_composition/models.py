# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed contracts for one durable, composed single-asset workflow."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import PurePath
from typing import Any, Final, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v1"
)
PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v2"
)
PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v3"
)
# Request.v3 and run.v4 are the canonical agentic graph identities that landed
# on main. Fixed-compatibility contracts continue at later versions.
ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v4"
)
PRE_SELECTED_MODE_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v5"
)
PRE_ARTICULATION_REVIEW_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    PRE_SELECTED_MODE_ASSET_COMPOSITION_RUN_SCHEMA_VERSION
)
COMPATIBILITY_FIXED_ASSET_COMPOSITION_RUN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-composition-run.v6"
)
LEGACY_ASSET_STAGE_HANDOFF_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-stage-handoff.v1"
)
ASSET_STAGE_HANDOFF_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-stage-handoff.v2"
)
ASSET_TERMINAL_VALIDATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-terminal-validation.v2"
)
ASSET_COMBINED_REPORT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-combined-report.v1"
)
ASSET_COORDINATOR_PLAN_DRAFT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-coordinator-plan-draft.v1"
)
ASSET_COORDINATOR_PLAN_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-coordinator-plan.v1"
)
ASSET_COORDINATOR_REVIEW_DRAFT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-coordinator-review-draft.v1"
)
ASSET_COORDINATOR_REVIEW_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-coordinator-review.v1"
)
ASSET_CROSS_STAGE_VALIDATION_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-cross-stage-validation.v1"
)
LEGACY_ASSET_REQUEST_SCHEMA_VERSION: Final = (
    "content-agents.asset-composition-request.v1"
)
PRE_CAD_MODELING_ASSET_REQUEST_SCHEMA_VERSION: Final = (
    "content-agents.asset-composition-request.v2"
)
ASSET_REQUEST_SCHEMA_VERSION: Final = "content-agents.asset-composition-request.v3"
PRE_PROVIDER_AUTHORING_ASSET_REQUEST_SCHEMA_VERSION: Final = (
    "content-agents.asset-composition-request.v4"
)
COMPATIBILITY_FIXED_ASSET_REQUEST_SCHEMA_VERSION: Final = (
    "content-agents.asset-composition-request.v5"
)
ASSET_CAD_MODELING_STAGE_RESULT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-cad-modeling-stage-result.v2"
)
ASSET_GEOMETRY_STAGE_RESULT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-geometry-stage-result.v1"
)
LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION: Final = (
    "content-agents.asset-execution-graph.v1"
)
ASSET_EXECUTION_GRAPH_SCHEMA_VERSION: Final = "content-agents.asset-execution-graph.v2"
LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION: Final = "content-agents.asset-leaf-catalog.v1"
LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION: Final = (
    "content-agents.asset-leaf-descriptor.v1"
)
ASSET_LEAF_CATALOG_SCHEMA_VERSION: Final = "content-agents.asset-leaf-catalog.v3"
ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION: Final = "content-agents.asset-leaf-descriptor.v3"
ASSET_LEAF_PROJECTOR_API_VERSION: Final = "content-agents.asset-leaf-projector-api.v2"
ASSET_LEAF_PROJECTION_CONTEXT_SCHEMA_VERSION: Final = (
    "content-agents.asset-leaf-projection-context.v1"
)
ASSET_LEAF_PROJECTION_SCHEMA_VERSION: Final = "content-agents.asset-leaf-projection.v2"
ASSET_LEAF_BUNDLE_IDENTITY_SCHEMA_VERSION: Final = (
    "content-agents.asset-leaf-bundle-identity.v1"
)
ASSET_LEAF_REGISTRAR_IDENTITY_SCHEMA_VERSION: Final = (
    "content-agents.asset-leaf-registrar-identity.v1"
)
ASSET_SOLE_COORDINATOR_IDENTITY_SCHEMA_VERSION: Final = (
    "content-agents.asset-sole-coordinator-identity.v1"
)
LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-leaf-receipt.v1"
)
ASSET_LEAF_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-leaf-receipt.v2"
)
LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-graph-terminal-receipt.v1"
)
ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION: Final = (
    "content-agent-workflows.asset-graph-terminal-receipt.v2"
)
MAX_COORDINATOR_REVIEW_EVIDENCE: Final = 1024

StageName = Literal[
    "cad_modeling",
    "geometry",
    "articulation",
    "material",
    "texture",
    "physics",
    "validation",
    "finalization",
]
StageStatus = Literal[
    "pending",
    "ready",
    "running",
    "needs_review",
    "completed",
    "failed",
    "cancelled",
]
TerminalStatus = Literal["active", "completed", "failed", "cancelled"]
CoordinatorDecision = Literal[
    "accept",
    "await_review",
    "refine",
    "revisit",
    "stop_failed",
    "stop_cancelled",
]
CoordinatorMode = Literal["legacy", "single_reasoning_loop"]
SelectedMode = Literal["agentic", "compatibility_fixed"]
LeafRequirement = Literal["required", "optional"]
LeafReceiptArtifactCategory = Literal[
    "operation_index",
    "evidence_index",
    "evidence",
    "saved_stage_readback",
    "resource_release",
]
LeafNativeDisposition = Literal[
    "pending",
    "running",
    "passed",
    "failed",
    "cancelled",
    "not_requested",
    "not_evaluated",
    "superseded",
]
LeafStateStatus = Literal[
    "pending",
    "ready",
    "running",
    "passed",
    "failed",
    "cancelled",
    "not_evaluated",
]
PhysicsValidationMode = Literal["runtime_required", "schema_readback"]
CoordinatorNextAction = Literal[
    "freeze_graph",
    "plan",
    "begin_stage",
    "execute_stage",
    "begin_leaf",
    "execute_leaf",
    "await_human_review",
    "complete_stage",
    "finalize_receipts",
    "stopped",
    "terminal",
]

LEGACY_STAGE_ORDER: tuple[StageName, ...] = (
    "articulation",
    "material",
    "texture",
    "physics",
    "validation",
    "finalization",
)
STAGE_ORDER: tuple[StageName, ...] = LEGACY_STAGE_ORDER
GEOMETRY_STAGE_ORDER: tuple[StageName, ...] = (
    "geometry",
    *LEGACY_STAGE_ORDER,
)
CAD_MODELING_STAGE_ORDER: tuple[StageName, ...] = (
    "cad_modeling",
    *GEOMETRY_STAGE_ORDER,
)
SUPPORTED_STAGE_ORDERS: Final = frozenset(
    {LEGACY_STAGE_ORDER, GEOMETRY_STAGE_ORDER, CAD_MODELING_STAGE_ORDER}
)
ALL_STAGE_NAMES: tuple[StageName, ...] = CAD_MODELING_STAGE_ORDER

CrossStageClaimName = Literal[
    "joint_graph",
    "appearance",
    "physics_behavior",
    "render_and_package",
    "non_target_preservation",
]
CrossStageClaimStatus = Literal["pass", "warn"]
CrossStageHandoffName = Literal[
    "articulation",
    "material",
    "texture",
    "physics",
]


def _canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_asset_digest(payload: Any) -> str:
    """Return the public canonical SHA-256 used by graph identity contracts."""

    return _canonical_digest(payload)


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _digest_model_without(model: BaseModel, *excluded: str) -> str:
    payload = model.model_dump(mode="json")
    for field in excluded:
        payload.pop(field, None)
    return _canonical_digest(payload)


class ArtifactBinding(BaseModel):
    """Immutable identity for one regular-file workflow artifact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


class AssetSegmentationRunBinding(BaseModel):
    """Exact immutable file closure for one staged mesh-segmentation run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-segmentation-run-binding.v1"
    ] = "content-agent-workflows.asset-segmentation-run-binding.v1"
    producer_run_id: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    run_dir: str = Field(min_length=1, max_length=4096)
    artifacts: tuple[ArtifactBinding, ...] = Field(
        min_length=1,
        max_length=20_000,
    )

    @model_validator(mode="after")
    def validate_artifact_closure(self) -> AssetSegmentationRunBinding:
        root = PurePath(self.run_dir)
        if not root.is_absolute() or ".." in root.parts:
            raise ValueError("segmentation run_dir must be an absolute normalized path")
        paths = [item.path for item in self.artifacts]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("segmentation artifacts must be unique and path-sorted")
        for value in paths:
            path = PurePath(value)
            if (
                not path.is_absolute()
                or ".." in path.parts
                or not path.is_relative_to(root)
            ):
                raise ValueError("segmentation artifacts must be inside run_dir")
        return self


class AssetSourceStaging(BaseModel):
    """Immutable custody bridge from an approved source closure into the run."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    original_source: ArtifactBinding
    original_dependencies: list[ArtifactBinding] = Field(min_length=1)
    staged_source: ArtifactBinding
    staged_dependencies: list[ArtifactBinding] = Field(min_length=1)
    manifest: ArtifactBinding
    dependency_digest_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_dependency_identity(self) -> AssetSourceStaging:
        original_digests = sorted(
            dependency.sha256 for dependency in self.original_dependencies
        )
        staged_digests = sorted(
            dependency.sha256 for dependency in self.staged_dependencies
        )
        if original_digests != staged_digests:
            raise ValueError(
                "staged source dependencies differ from the approved source bytes"
            )
        if self.original_source.sha256 not in original_digests:
            raise ValueError("original source is missing from its dependency closure")
        if self.staged_source.sha256 not in staged_digests:
            raise ValueError("staged source is missing from its dependency closure")
        return self


class AssetSoleCoordinatorIdentity(BaseModel):
    """Durable identity for the only reasoner allowed to own one graph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.asset-sole-coordinator-identity.v1"] = (
        ASSET_SOLE_COORDINATOR_IDENTITY_SCHEMA_VERSION
    )
    coordinator_id: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    invocation_id: str = Field(
        min_length=1,
        max_length=240,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    actor: str = Field(min_length=1, max_length=240)
    implementation: str = Field(min_length=1, max_length=240)
    identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        coordinator_id: str,
        invocation_id: str,
        actor: str,
        implementation: str,
    ) -> AssetSoleCoordinatorIdentity:
        """Build a self-digested durable coordinator identity."""

        payload = {
            "schema_version": ASSET_SOLE_COORDINATOR_IDENTITY_SCHEMA_VERSION,
            "coordinator_id": coordinator_id,
            "invocation_id": invocation_id,
            "actor": actor,
            "implementation": implementation,
        }
        return cls.model_validate(
            {**payload, "identity_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_identity_digest(self) -> AssetSoleCoordinatorIdentity:
        if self.identity_digest != _digest_model_without(self, "identity_digest"):
            raise ValueError("sole coordinator identity digest is stale")
        return self


class AssetLeafDescriptor(BaseModel):
    """Opaque public entrypoint identity available to the outer coordinator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agents.asset-leaf-descriptor.v1",
        "content-agents.asset-leaf-descriptor.v3",
    ] = ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
    leaf_id: str = Field(
        min_length=3,
        max_length=240,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*\.v[1-9][0-9]*$",
    )
    entrypoint: str = Field(min_length=1, max_length=1000)
    invocation_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_schema_digest: str = Field(
        default="0" * 64,
        pattern=r"^[0-9a-f]{64}$",
    )
    projector_id: str = Field(
        default="asset.projector.unbound.v1",
        min_length=3,
        max_length=240,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*\.v[1-9][0-9]*$",
    )
    projector_digest: str = Field(default="0" * 64, pattern=r"^[0-9a-f]{64}$")
    required_artifact_categories: list[LeafReceiptArtifactCategory] = Field(
        default_factory=list,
        max_length=5,
    )
    required_dependencies: list[str] = Field(default_factory=list, max_length=64)
    required_dependents: list[str] = Field(default_factory=list, max_length=64)
    incompatible_leaf_ids: list[str] = Field(default_factory=list, max_length=64)
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        leaf_id: str,
        entrypoint: str,
        invocation_schema_digest: str,
        result_schema_digest: str,
        projection_schema_digest: str = "0" * 64,
        projector_id: str = "asset.projector.unbound.v1",
        projector_digest: str = "0" * 64,
        required_artifact_categories: list[LeafReceiptArtifactCategory] | None = None,
        required_dependencies: list[str] | None = None,
        required_dependents: list[str] | None = None,
        incompatible_leaf_ids: list[str] | None = None,
    ) -> AssetLeafDescriptor:
        """Build a self-digested, domain-neutral public leaf descriptor."""

        legacy = (
            projection_schema_digest == "0" * 64
            and projector_id == "asset.projector.unbound.v1"
            and projector_digest == "0" * 64
            and not required_artifact_categories
            and not required_dependents
        )
        payload = {
            "schema_version": (
                LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
                if legacy
                else ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
            ),
            "leaf_id": leaf_id,
            "entrypoint": entrypoint,
            "invocation_schema_digest": invocation_schema_digest,
            "result_schema_digest": result_schema_digest,
            "required_dependencies": sorted(required_dependencies or []),
            "incompatible_leaf_ids": sorted(incompatible_leaf_ids or []),
        }
        if not legacy:
            payload.update(
                {
                    "projection_schema_digest": projection_schema_digest,
                    "projector_id": projector_id,
                    "projector_digest": projector_digest,
                    "required_artifact_categories": sorted(
                        required_artifact_categories or []
                    ),
                    "required_dependents": sorted(required_dependents or []),
                }
            )
        return cls.model_validate(
            {**payload, "descriptor_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_descriptor(self) -> AssetLeafDescriptor:
        if self.schema_version == LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION and (
            self.projection_schema_digest != "0" * 64
            or self.projector_id != "asset.projector.unbound.v1"
            or self.projector_digest != "0" * 64
            or self.required_artifact_categories
            or self.required_dependents
        ):
            raise ValueError("legacy leaf descriptor carries v3 projection fields")
        for label, constraints in (
            ("required artifact categories", self.required_artifact_categories),
            ("required dependencies", self.required_dependencies),
            ("required dependents", self.required_dependents),
            ("incompatible leaf IDs", self.incompatible_leaf_ids),
        ):
            if constraints != sorted(constraints) or len(constraints) != len(
                set(constraints)
            ):
                raise ValueError(f"leaf {label} must be sorted and unique")
            if self.leaf_id in constraints:
                raise ValueError(f"leaf cannot list itself in {label}")
        if self.descriptor_digest != _canonical_digest(self._identity_payload()):
            raise ValueError(f"leaf descriptor digest is stale for {self.leaf_id}")
        return self

    @model_serializer(mode="wrap")
    def _serialize_wire_version(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Retain the exact v1 descriptor surface for existing domain catalogs."""

        serialized = handler(self)
        if self.schema_version == LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION:
            for field in (
                "projection_schema_digest",
                "projector_id",
                "projector_digest",
                "required_artifact_categories",
                "required_dependents",
            ):
                serialized.pop(field, None)
        return serialized

    def _identity_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "leaf_id": self.leaf_id,
            "entrypoint": self.entrypoint,
            "invocation_schema_digest": self.invocation_schema_digest,
            "result_schema_digest": self.result_schema_digest,
            "required_dependencies": self.required_dependencies,
            "incompatible_leaf_ids": self.incompatible_leaf_ids,
        }
        if self.schema_version == ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION:
            payload.update(
                {
                    "projection_schema_digest": self.projection_schema_digest,
                    "projector_id": self.projector_id,
                    "projector_digest": self.projector_digest,
                    "required_artifact_categories": (self.required_artifact_categories),
                    "required_dependents": self.required_dependents,
                }
            )
        return payload

    def _catalog_payload(self) -> dict[str, Any]:
        return {**self._identity_payload(), "descriptor_digest": self.descriptor_digest}


class AssetLeafBundleIdentity(BaseModel):
    """Repository-derived identity of one domain-owned runtime bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.asset-leaf-bundle-identity.v1"] = (
        ASSET_LEAF_BUNDLE_IDENTITY_SCHEMA_VERSION
    )
    registrar_id: str = Field(min_length=1, max_length=240)
    bundle_id: str = Field(min_length=1, max_length=240)
    implementation: str = Field(min_length=1, max_length=1000)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaf_descriptor_digests: dict[str, str] = Field(min_length=1, max_length=256)
    bundle_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        registrar_id: str,
        bundle_id: str,
        implementation: str,
        source_sha256: str,
        leaf_descriptor_digests: dict[str, str],
    ) -> AssetLeafBundleIdentity:
        ordered = {
            leaf_id: leaf_descriptor_digests[leaf_id]
            for leaf_id in sorted(leaf_descriptor_digests)
        }
        payload = {
            "schema_version": ASSET_LEAF_BUNDLE_IDENTITY_SCHEMA_VERSION,
            "registrar_id": registrar_id,
            "bundle_id": bundle_id,
            "implementation": implementation,
            "source_sha256": source_sha256,
            "leaf_descriptor_digests": ordered,
        }
        return cls.model_validate(
            {**payload, "bundle_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_identity(self) -> AssetLeafBundleIdentity:
        if list(self.leaf_descriptor_digests) != sorted(self.leaf_descriptor_digests):
            raise ValueError("bundle leaf identities must be lexically ordered")
        if any(
            not _is_sha256(digest) for digest in self.leaf_descriptor_digests.values()
        ):
            raise ValueError("bundle leaf descriptor digest is invalid")
        if self.bundle_digest != _digest_model_without(self, "bundle_digest"):
            raise ValueError("asset leaf bundle identity digest is stale")
        return self


class AssetLeafRegistrarIdentity(BaseModel):
    """Approved package registrar and every mandatory returned bundle."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.asset-leaf-registrar-identity.v1"] = (
        ASSET_LEAF_REGISTRAR_IDENTITY_SCHEMA_VERSION
    )
    registrar_id: str = Field(min_length=1, max_length=240)
    distribution_version: str = Field(min_length=1, max_length=240)
    implementation: str = Field(min_length=1, max_length=1000)
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bundles: list[AssetLeafBundleIdentity] = Field(min_length=1, max_length=256)
    registrar_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        registrar_id: str,
        distribution_version: str,
        bundles: list[AssetLeafBundleIdentity],
    ) -> AssetLeafRegistrarIdentity:
        ordered = sorted(bundles, key=lambda bundle: bundle.bundle_id)
        implementation = "|".join(bundle.implementation for bundle in ordered)
        source_sha256 = _canonical_digest([bundle.source_sha256 for bundle in ordered])
        payload = {
            "schema_version": ASSET_LEAF_REGISTRAR_IDENTITY_SCHEMA_VERSION,
            "registrar_id": registrar_id,
            "distribution_version": distribution_version,
            "implementation": implementation,
            "source_sha256": source_sha256,
            "bundles": [bundle.model_dump(mode="json") for bundle in ordered],
        }
        return cls.model_validate(
            {**payload, "registrar_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_identity(self) -> AssetLeafRegistrarIdentity:
        bundle_ids = [bundle.bundle_id for bundle in self.bundles]
        if bundle_ids != sorted(bundle_ids) or len(bundle_ids) != len(set(bundle_ids)):
            raise ValueError("registrar bundle identities must be sorted and unique")
        if any(bundle.registrar_id != self.registrar_id for bundle in self.bundles):
            raise ValueError("registrar contains a foreign bundle identity")
        expected_implementation = "|".join(
            bundle.implementation for bundle in self.bundles
        )
        if self.implementation != expected_implementation:
            raise ValueError("registrar implementation identity differs from bundles")
        expected_source_sha256 = _canonical_digest(
            [bundle.source_sha256 for bundle in self.bundles]
        )
        if self.source_sha256 != expected_source_sha256:
            raise ValueError("registrar source identity differs from bundles")
        if self.registrar_digest != _digest_model_without(self, "registrar_digest"):
            raise ValueError("asset leaf registrar identity digest is stale")
        return self


class AssetLeafCatalog(BaseModel):
    """Frozen, domain-neutral catalog of opaque public entrypoints."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agents.asset-leaf-catalog.v1",
        "content-agents.asset-leaf-catalog.v3",
    ] = ASSET_LEAF_CATALOG_SCHEMA_VERSION
    registrars: list[AssetLeafRegistrarIdentity] = Field(
        default_factory=list,
        max_length=64,
    )
    descriptors: list[AssetLeafDescriptor] = Field(min_length=1, max_length=256)
    catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        descriptors: list[AssetLeafDescriptor],
        *,
        registrars: list[AssetLeafRegistrarIdentity] | None = None,
    ) -> AssetLeafCatalog:
        """Build a sorted, self-digested catalog without selecting any leaf."""

        ordered = sorted(descriptors, key=lambda descriptor: descriptor.leaf_id)
        ordered_registrars = [
            item.model_dump(mode="json")
            for item in sorted(
                registrars or [],
                key=lambda registrar: registrar.registrar_id,
            )
        ]
        legacy = not ordered_registrars and all(
            item.schema_version == LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
            for item in ordered
        )
        payload = {
            "schema_version": (
                LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
                if legacy
                else ASSET_LEAF_CATALOG_SCHEMA_VERSION
            ),
            "descriptors": [item._catalog_payload() for item in ordered],
        }
        if not legacy:
            payload["registrars"] = ordered_registrars
        return cls.model_validate(
            {**payload, "catalog_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_catalog(self) -> AssetLeafCatalog:
        legacy = self.schema_version == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION
        if legacy and (
            self.registrars
            or any(
                descriptor.schema_version != LEGACY_ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
                for descriptor in self.descriptors
            )
        ):
            raise ValueError("legacy leaf catalog carries repository runtime identity")
        if not legacy and any(
            descriptor.schema_version != ASSET_LEAF_DESCRIPTOR_SCHEMA_VERSION
            for descriptor in self.descriptors
        ):
            raise ValueError("repository leaf catalog contains a legacy descriptor")
        registrar_ids = [registrar.registrar_id for registrar in self.registrars]
        if registrar_ids != sorted(registrar_ids) or len(registrar_ids) != len(
            set(registrar_ids)
        ):
            raise ValueError("leaf catalog registrars must be sorted and unique")
        leaf_ids = [descriptor.leaf_id for descriptor in self.descriptors]
        if leaf_ids != sorted(leaf_ids):
            raise ValueError("leaf catalog descriptors must be sorted by leaf_id")
        if len(leaf_ids) != len(set(leaf_ids)):
            raise ValueError("leaf catalog contains duplicate leaf IDs")
        known = set(leaf_ids)
        for descriptor in self.descriptors:
            unknown = sorted(
                set(
                    [
                        *descriptor.required_dependencies,
                        *descriptor.required_dependents,
                        *descriptor.incompatible_leaf_ids,
                    ]
                ).difference(known)
            )
            if unknown:
                raise ValueError(
                    f"leaf {descriptor.leaf_id} has unknown constraints: {unknown}"
                )
        if self.registrars:
            registered_entries = [
                (leaf_id, descriptor_digest)
                for registrar in self.registrars
                for bundle in registrar.bundles
                for leaf_id, descriptor_digest in (
                    bundle.leaf_descriptor_digests.items()
                )
            ]
            registered = {
                leaf_id: descriptor_digest
                for leaf_id, descriptor_digest in registered_entries
            }
            if len(registered) != len(registered_entries):
                raise ValueError("registrar bundles contain duplicate leaf IDs")
            if registered != {
                descriptor.leaf_id: descriptor.descriptor_digest
                for descriptor in self.descriptors
            }:
                raise ValueError(
                    "leaf catalog descriptors differ from registrar bundle identities"
                )
        payload = {
            "schema_version": self.schema_version,
            "descriptors": [
                descriptor._catalog_payload() for descriptor in self.descriptors
            ],
        }
        if not legacy:
            payload["registrars"] = [
                registrar.model_dump(mode="json") for registrar in self.registrars
            ]
        if self.catalog_digest != _canonical_digest(payload):
            raise ValueError("leaf catalog digest is stale")
        return self

    @model_serializer(mode="wrap")
    def _serialize_wire_version(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Retain the exact v1 catalog surface when no runtime registrar exists."""

        serialized = handler(self)
        if self.schema_version == LEGACY_ASSET_LEAF_CATALOG_SCHEMA_VERSION:
            serialized.pop("registrars", None)
        return serialized


class AssetLeafProjectionPayload(BaseModel):
    """Deterministic native-to-composition facts returned by one projector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    native_disposition: Literal[
        "passed",
        "failed",
        "cancelled",
        "not_evaluated",
    ]
    native_status: str = Field(min_length=1, max_length=240)
    native_terminal_receipt: ArtifactBinding
    operation_indexes: tuple[ArtifactBinding, ...] = ()
    evidence_indexes: tuple[ArtifactBinding, ...] = ()
    evidence: tuple[ArtifactBinding, ...] = ()
    saved_stage_readbacks: tuple[ArtifactBinding, ...] = ()
    resource_claims: tuple[str, ...] = ()
    resource_release_receipts: tuple[ArtifactBinding, ...] = ()
    summary: str = Field(min_length=1)
    error: str | None = None

    @model_validator(mode="after")
    def validate_projection_payload(self) -> AssetLeafProjectionPayload:
        if self.native_status == "not_requested":
            raise ValueError(
                "not_requested is graph omission and cannot project a selected leaf"
            )
        for label, values in (
            ("operation indexes", self.operation_indexes),
            ("evidence indexes", self.evidence_indexes),
            ("evidence", self.evidence),
            ("saved-stage readbacks", self.saved_stage_readbacks),
            ("resource releases", self.resource_release_receipts),
        ):
            identities = [
                (binding.path, binding.sha256, binding.size_bytes) for binding in values
            ]
            if len(identities) != len(set(identities)):
                raise ValueError(f"projected {label} must be unique")
        if len(self.resource_claims) != len(set(self.resource_claims)):
            raise ValueError("projected resource claims must be unique")
        if self.native_disposition in {"failed", "cancelled"}:
            if not self.error:
                raise ValueError("failed or cancelled projection requires an error")
        elif self.error is not None:
            raise ValueError(
                "successful or unevaluated projection cannot carry an error"
            )
        return self


class AssetLeafProjectionContext(BaseModel):
    """Immutable selected-artifact identity supplied to a leaf projector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.asset-leaf-projection-context.v1"] = (
        ASSET_LEAF_PROJECTION_CONTEXT_SCHEMA_VERSION
    )
    leaf_id: str
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projector_id: str
    projector_api_version: Literal["content-agents.asset-leaf-projector-api.v2"] = (
        ASSET_LEAF_PROJECTOR_API_VERSION
    )
    projector_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    context_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_artifact: ArtifactBinding
    result_artifact: ArtifactBinding
    context_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        leaf_id: str,
        descriptor_digest: str,
        invocation_schema_digest: str,
        result_schema_digest: str,
        projector_id: str,
        projector_api_version: Literal["content-agents.asset-leaf-projector-api.v2"],
        projector_digest: str,
        context_schema_digest: str,
        invocation_artifact: ArtifactBinding,
        result_artifact: ArtifactBinding,
    ) -> AssetLeafProjectionContext:
        """Freeze the exact invocation/result artifacts selected for projection."""

        data = {
            "schema_version": ASSET_LEAF_PROJECTION_CONTEXT_SCHEMA_VERSION,
            "leaf_id": leaf_id,
            "descriptor_digest": descriptor_digest,
            "invocation_schema_digest": invocation_schema_digest,
            "result_schema_digest": result_schema_digest,
            "projector_id": projector_id,
            "projector_api_version": projector_api_version,
            "projector_digest": projector_digest,
            "context_schema_digest": context_schema_digest,
            "invocation_artifact": invocation_artifact.model_dump(mode="json"),
            "result_artifact": result_artifact.model_dump(mode="json"),
        }
        return cls.model_validate({**data, "context_digest": _canonical_digest(data)})

    def require_native_artifact_chain(
        self,
        *,
        invocation_artifact: ArtifactBinding,
        result_artifact: ArtifactBinding,
    ) -> None:
        """Reject a native receipt chain that identifies other selected artifacts."""

        self.require_invocation_artifact(invocation_artifact)
        self.require_result_artifact(result_artifact)

    def require_invocation_artifact(self, artifact: ArtifactBinding) -> None:
        """Require an exact path/digest/size match for the selected invocation."""

        if artifact != self.invocation_artifact:
            raise ValueError(
                "native invocation artifact differs from frozen projection context"
            )

    def require_result_artifact(self, artifact: ArtifactBinding) -> None:
        """Require an exact path/digest/size match for the selected result."""

        if artifact != self.result_artifact:
            raise ValueError(
                "native result artifact differs from frozen projection context"
            )

    @model_validator(mode="after")
    def validate_context_digest(self) -> AssetLeafProjectionContext:
        if self.context_schema_digest != canonical_asset_digest(
            type(self).model_json_schema(mode="validation")
        ):
            raise ValueError("asset leaf projection context schema digest drifted")
        if self.context_digest != _digest_model_without(self, "context_digest"):
            raise ValueError("asset leaf projection context digest is stale")
        return self


class AssetLeafProjection(BaseModel):
    """Descriptor-bound deterministic projection retained with a leaf receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agents.asset-leaf-projection.v2"] = (
        ASSET_LEAF_PROJECTION_SCHEMA_VERSION
    )
    leaf_id: str
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    result_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_context_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projection_schema_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    projector_id: str
    projector_api_version: Literal["content-agents.asset-leaf-projector-api.v2"] = (
        ASSET_LEAF_PROJECTOR_API_VERSION
    )
    projector_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    invocation: ArtifactBinding
    result: ArtifactBinding
    context: AssetLeafProjectionContext
    payload: AssetLeafProjectionPayload
    projection_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        leaf_id: str,
        descriptor_digest: str,
        invocation_schema_digest: str,
        result_schema_digest: str,
        projection_context_schema_digest: str,
        projection_schema_digest: str,
        projector_id: str,
        projector_api_version: Literal["content-agents.asset-leaf-projector-api.v2"],
        projector_digest: str,
        invocation: ArtifactBinding,
        result: ArtifactBinding,
        context: AssetLeafProjectionContext,
        payload: AssetLeafProjectionPayload,
    ) -> AssetLeafProjection:
        data = {
            "schema_version": ASSET_LEAF_PROJECTION_SCHEMA_VERSION,
            "leaf_id": leaf_id,
            "descriptor_digest": descriptor_digest,
            "invocation_schema_digest": invocation_schema_digest,
            "result_schema_digest": result_schema_digest,
            "projection_context_schema_digest": projection_context_schema_digest,
            "projection_schema_digest": projection_schema_digest,
            "projector_id": projector_id,
            "projector_api_version": projector_api_version,
            "projector_digest": projector_digest,
            "invocation": invocation.model_dump(mode="json"),
            "result": result.model_dump(mode="json"),
            "context": context.model_dump(mode="json"),
            "payload": payload.model_dump(mode="json"),
        }
        return cls.model_validate(
            {**data, "projection_digest": _canonical_digest(data)}
        )

    @model_validator(mode="after")
    def validate_projection_digest(self) -> AssetLeafProjection:
        if (
            self.context.leaf_id != self.leaf_id
            or self.context.descriptor_digest != self.descriptor_digest
            or self.context.invocation_schema_digest != self.invocation_schema_digest
            or self.context.result_schema_digest != self.result_schema_digest
            or self.context.context_schema_digest
            != self.projection_context_schema_digest
            or self.context.projector_id != self.projector_id
            or self.context.projector_api_version != self.projector_api_version
            or self.context.projector_digest != self.projector_digest
            or self.context.invocation_artifact != self.invocation
            or self.context.result_artifact != self.result
        ):
            raise ValueError(
                "asset leaf projection differs from its frozen projection context"
            )
        if self.projection_digest != _digest_model_without(self, "projection_digest"):
            raise ValueError("asset leaf projection digest is stale")
        return self


class AssetExecutionNode(BaseModel):
    """One outer-selected opaque leaf and its prompt-specific dependencies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    leaf_id: str = Field(
        min_length=3,
        max_length=240,
        pattern=r"^[a-z0-9]+(?:[._-][a-z0-9]+)*\.v[1-9][0-9]*$",
    )
    depends_on: list[str] = Field(default_factory=list, max_length=128)
    requirement: LeafRequirement
    descriptor_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    terminal_output: bool = False
    selection_rationale: str | None = Field(default=None, min_length=1, max_length=1000)

    @model_validator(mode="after")
    def validate_dependencies(self) -> AssetExecutionNode:
        if self.leaf_id in self.depends_on:
            raise ValueError("leaf cannot depend on itself")
        if len(self.depends_on) != len(set(self.depends_on)):
            raise ValueError("leaf dependencies must be unique")
        return self

    @model_serializer(mode="wrap")
    def _serialize_selection_rationale(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep legacy graph bytes stable when no rationale was requested."""

        serialized = handler(self)
        if self.selection_rationale is None:
            serialized.pop("selection_rationale", None)
        return serialized


class AssetExecutionGraph(BaseModel):
    """Immutable prompt-specific graph supplied by the sole outer reasoner."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agents.asset-execution-graph.v1",
        "content-agents.asset-execution-graph.v2",
    ] = ASSET_EXECUTION_GRAPH_SCHEMA_VERSION
    selected_mode: Literal["agentic"] = "agentic"
    sole_coordinator_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaf_catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_leaf_ids: list[str] = Field(min_length=1, max_length=256)
    nodes: list[AssetExecutionNode] = Field(min_length=1, max_length=256)
    omitted_leaf_ids: list[str] = Field(default_factory=list, max_length=256)
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def create(
        cls,
        *,
        sole_coordinator_identity_digest: str,
        prompt_digest: str,
        source_digest: str,
        configuration_digest: str,
        reference_digest: str,
        leaf_catalog_digest: str,
        nodes: list[AssetExecutionNode],
        omitted_leaf_ids: list[str],
    ) -> AssetExecutionGraph:
        """Build a self-digested DAG with canonical, non-execution ordering."""

        ordered_nodes = sorted(
            (
                node.model_copy(update={"depends_on": sorted(node.depends_on)})
                for node in nodes
            ),
            key=lambda node: node.leaf_id,
        )
        ordered_omissions = sorted(omitted_leaf_ids)
        payload = {
            "schema_version": ASSET_EXECUTION_GRAPH_SCHEMA_VERSION,
            "selected_mode": "agentic",
            "sole_coordinator_identity_digest": sole_coordinator_identity_digest,
            "prompt_digest": prompt_digest,
            "source_digest": source_digest,
            "configuration_digest": configuration_digest,
            "reference_digest": reference_digest,
            "leaf_catalog_digest": leaf_catalog_digest,
            "selected_leaf_ids": [node.leaf_id for node in ordered_nodes],
            "nodes": [node.model_dump(mode="json") for node in ordered_nodes],
            "omitted_leaf_ids": ordered_omissions,
        }
        return cls.model_validate(
            {**payload, "graph_digest": _canonical_digest(payload)}
        )

    @model_validator(mode="after")
    def validate_graph(self) -> AssetExecutionGraph:
        node_ids = [node.leaf_id for node in self.nodes]
        if node_ids != self.selected_leaf_ids:
            raise ValueError("graph nodes must follow selected_leaf_ids exactly")
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("graph contains duplicate selected leaf IDs")
        if len(self.omitted_leaf_ids) != len(set(self.omitted_leaf_ids)):
            raise ValueError("graph contains duplicate omitted leaf IDs")
        if set(node_ids).intersection(self.omitted_leaf_ids):
            raise ValueError("selected and omitted leaf IDs must be disjoint")
        selected = set(node_ids)
        if self.schema_version == LEGACY_ASSET_EXECUTION_GRAPH_SCHEMA_VERSION:
            positions = {leaf_id: index for index, leaf_id in enumerate(node_ids)}
            for node in self.nodes:
                unknown = sorted(set(node.depends_on).difference(positions))
                if unknown:
                    raise ValueError(
                        f"leaf {node.leaf_id} has missing dependencies: {unknown}"
                    )
                if any(
                    positions[dependency] >= positions[node.leaf_id]
                    for dependency in node.depends_on
                ):
                    raise ValueError(
                        "selected_leaf_ids must be the outer-supplied topological order"
                    )
        else:
            if node_ids != sorted(node_ids):
                raise ValueError("graph selected leaf IDs must be lexically sorted")
            if self.omitted_leaf_ids != sorted(self.omitted_leaf_ids):
                raise ValueError("graph omitted leaf IDs must be lexically sorted")
            if any(node.depends_on != sorted(node.depends_on) for node in self.nodes):
                raise ValueError("graph v2 leaf dependencies must be lexically sorted")
            remaining: dict[str, set[str]] = {}
            for node in self.nodes:
                unknown = sorted(set(node.depends_on).difference(selected))
                if unknown:
                    raise ValueError(
                        f"leaf {node.leaf_id} has missing dependencies: {unknown}"
                    )
                remaining[node.leaf_id] = set(node.depends_on)
            while remaining:
                ready = sorted(
                    leaf_id
                    for leaf_id, dependencies in remaining.items()
                    if not dependencies
                )
                if not ready:
                    raise ValueError("graph dependencies must be acyclic")
                for leaf_id in ready:
                    remaining.pop(leaf_id)
                for dependencies in remaining.values():
                    dependencies.difference_update(ready)
        if not any(node.terminal_output for node in self.nodes):
            raise ValueError(
                "graph requires at least one outer-selected terminal output"
            )
        if self.graph_digest != _digest_model_without(self, "graph_digest"):
            raise ValueError("execution graph digest is stale")
        return self


class AssetLeafReceipt(BaseModel):
    """Immutable state-authored receipt for one selected opaque leaf attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-leaf-receipt.v1",
        "content-agent-workflows.asset-leaf-receipt.v2",
    ] = ASSET_LEAF_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    run_revision: int = Field(ge=0)
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaf_catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sole_coordinator_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaf_id: str
    descriptor_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    invocation_schema_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    result_schema_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    projection_schema_digest: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    projector_id: str | None = Field(default=None, min_length=1)
    projector_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    required_artifact_categories: list[LeafReceiptArtifactCategory] = Field(
        default_factory=list,
        max_length=5,
    )
    requirement: LeafRequirement
    depends_on: list[str]
    dependency_receipts: dict[str, ArtifactBinding]
    attempt: int = Field(ge=1)
    invocation: ArtifactBinding
    result: ArtifactBinding | None = None
    projection: ArtifactBinding | None = None
    native_terminal_receipt: ArtifactBinding | None = None
    operation_indexes: list[ArtifactBinding] = Field(default_factory=list)
    evidence_indexes: list[ArtifactBinding] = Field(default_factory=list)
    evidence: list[ArtifactBinding] = Field(default_factory=list)
    saved_stage_readbacks: list[ArtifactBinding] = Field(default_factory=list)
    native_disposition: LeafNativeDisposition
    native_status: str | None = Field(default=None, min_length=1, max_length=240)
    started_at: str
    finished_at: str
    duration_ms: int = Field(ge=0)
    resource_claims: list[str] = Field(default_factory=list, max_length=128)
    resource_release_receipts: list[ArtifactBinding] = Field(default_factory=list)
    supersedes: ArtifactBinding | None = None
    summary: str = Field(min_length=1)
    error: str | None = None
    actor: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def validate_wire_version_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        schema_version = data.get("schema_version", ASSET_LEAF_RECEIPT_SCHEMA_VERSION)
        v2_only = {
            "descriptor_digest",
            "invocation_schema_digest",
            "result_schema_digest",
            "projection_schema_digest",
            "projector_id",
            "projector_digest",
            "required_artifact_categories",
            "projection",
            "native_status",
        }
        if schema_version == LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION:
            mixed = sorted(v2_only.intersection(data))
            if mixed:
                raise ValueError(f"legacy leaf receipt carries v2 fields: {mixed}")
        elif schema_version == ASSET_LEAF_RECEIPT_SCHEMA_VERSION:
            missing = sorted(v2_only.difference(data))
            if missing:
                raise ValueError(f"v2 leaf receipt omits required fields: {missing}")
        return data

    @model_validator(mode="after")
    def validate_receipt(self) -> AssetLeafReceipt:
        if set(self.dependency_receipts) != set(self.depends_on):
            raise ValueError("leaf receipt must bind every dependency receipt")
        if len(self.resource_claims) != len(set(self.resource_claims)):
            raise ValueError("leaf resource claims must be unique")
        for label, bindings in (
            ("operation indexes", self.operation_indexes),
            ("evidence indexes", self.evidence_indexes),
            ("evidence", self.evidence),
            ("saved-stage readbacks", self.saved_stage_readbacks),
            ("resource releases", self.resource_release_receipts),
        ):
            identities = [
                (binding.path, binding.sha256, binding.size_bytes)
                for binding in bindings
            ]
            if len(identities) != len(set(identities)):
                raise ValueError(f"leaf receipt {label} must be unique")
        if self.resource_claims and not self.resource_release_receipts:
            raise ValueError("claimed resources require release receipts")
        if self.schema_version == LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION:
            if self.native_disposition == "passed":
                if (
                    self.result is None
                    or not self.evidence
                    or not self.saved_stage_readbacks
                ):
                    raise ValueError(
                        "passed legacy leaf requires result, evidence, and "
                        "saved-stage readback"
                    )
                if self.error is not None:
                    raise ValueError("passed leaf cannot carry an error")
            elif self.native_disposition == "not_evaluated":
                if self.requirement != "optional":
                    raise ValueError("required leaf cannot be not_evaluated")
                if self.result is None or not self.evidence:
                    raise ValueError(
                        "not_evaluated legacy leaf requires result and evidence"
                    )
                if self.error is not None:
                    raise ValueError("not_evaluated leaf cannot carry an error")
            elif self.native_disposition in {"failed", "cancelled"}:
                if not self.error:
                    raise ValueError("failed or cancelled leaf requires an error")
            else:
                raise ValueError("leaf receipt requires a terminal native disposition")
            return self
        if any(
            item is None
            for item in (
                self.descriptor_digest,
                self.invocation_schema_digest,
                self.result_schema_digest,
                self.projection_schema_digest,
                self.projector_id,
                self.projector_digest,
                self.result,
                self.projection,
                self.native_terminal_receipt,
                self.native_status,
            )
        ):
            raise ValueError("v2 leaf receipt lacks descriptor-resolved identity")
        observed_categories = {
            category
            for category, bindings in (
                ("operation_index", self.operation_indexes),
                ("evidence_index", self.evidence_indexes),
                ("evidence", self.evidence),
                ("saved_stage_readback", self.saved_stage_readbacks),
                ("resource_release", self.resource_release_receipts),
            )
            if bindings
        }
        missing_categories = sorted(
            set(self.required_artifact_categories).difference(observed_categories)
        )
        if missing_categories:
            raise ValueError(
                "leaf receipt lacks descriptor-required artifact categories: "
                f"{missing_categories}"
            )
        if self.native_disposition == "passed":
            if self.error is not None:
                raise ValueError("passed leaf cannot carry an error")
        elif self.native_disposition == "not_evaluated":
            if self.requirement != "optional":
                raise ValueError("required leaf cannot be not_evaluated")
            if self.error is not None:
                raise ValueError("not_evaluated leaf cannot carry an error")
        elif self.native_disposition in {"failed", "cancelled"}:
            if not self.error:
                raise ValueError("failed or cancelled leaf requires an error")
        else:
            raise ValueError("leaf receipt requires a terminal native disposition")
        return self

    @model_serializer(mode="wrap")
    def _serialize_wire_version(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        serialized = handler(self)
        if self.schema_version == LEGACY_ASSET_LEAF_RECEIPT_SCHEMA_VERSION:
            for field in (
                "descriptor_digest",
                "invocation_schema_digest",
                "result_schema_digest",
                "projection_schema_digest",
                "projector_id",
                "projector_digest",
                "required_artifact_categories",
                "projection",
                "native_status",
            ):
                serialized.pop(field, None)
        return serialized


class AssetLeafState(BaseModel):
    """Durable progress for one selected opaque leaf."""

    model_config = ConfigDict(extra="forbid")

    leaf_id: str
    requirement: LeafRequirement
    depends_on: list[str] = Field(default_factory=list)
    terminal_output: bool = False
    status: LeafStateStatus = "pending"
    attempt_count: int = Field(default=0, ge=0)
    started_at: str | None = None
    receipt: ArtifactBinding | None = None
    superseded_receipts: list[ArtifactBinding] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def validate_state(self) -> AssetLeafState:
        if self.status == "running" and self.started_at is None:
            raise ValueError("running leaf requires started_at")
        if self.status in {"passed", "failed", "cancelled", "not_evaluated"}:
            if self.receipt is None:
                raise ValueError("terminal leaf requires a receipt")
        elif self.receipt is not None:
            raise ValueError("nonterminal leaf cannot carry a receipt")
        if self.status in {"failed", "cancelled"} and not self.error:
            raise ValueError("failed or cancelled leaf requires an error")
        return self


class AssetLeafTransition(BaseModel):
    """Append-only transition for one graph-selected opaque leaf."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: str
    leaf_id: str
    from_status: LeafStateStatus
    to_status: LeafStateStatus
    reason: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    attempt_count: int = Field(ge=0)


class AssetGraphTerminalReceipt(BaseModel):
    """Comprehensive graph-bound terminal and resource-release receipt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-graph-terminal-receipt.v1",
        "content-agent-workflows.asset-graph-terminal-receipt.v2",
    ] = ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    graph: ArtifactBinding
    graph_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    leaf_catalog_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    sole_coordinator_identity_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    prompt_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    reference_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_leaf_ids: list[str] = Field(min_length=1, max_length=256)
    selected_leaf_receipts: dict[str, ArtifactBinding]
    omitted_leaf_ids: list[str] = Field(default_factory=list, max_length=256)
    omitted_leaf_dispositions: dict[str, Literal["not_requested"]]
    terminal_artifacts: list[ArtifactBinding] = Field(min_length=1)
    resource_release_receipts: list[ArtifactBinding] = Field(default_factory=list)
    parent_release_receipt: ArtifactBinding | None = None
    parent_command_receipt_journal: ArtifactBinding | None = None
    parent_command_receipt_checkpoint: ArtifactBinding | None = None
    started_at: str
    finished_at: str
    duration_ms: int = Field(ge=0)
    actor: str = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def validate_wire_version_fields(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        schema_version = data.get(
            "schema_version",
            ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION,
        )
        parent_fields = {
            "parent_release_receipt",
            "parent_command_receipt_journal",
            "parent_command_receipt_checkpoint",
        }
        if schema_version == LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION:
            mixed = sorted(parent_fields.intersection(data))
            if mixed:
                raise ValueError(
                    f"legacy graph terminal receipt carries v2 fields: {mixed}"
                )
        elif schema_version == ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION:
            missing = sorted(parent_fields.difference(data))
            if missing:
                raise ValueError(
                    f"v2 graph terminal receipt omits required fields: {missing}"
                )
        return data

    @model_validator(mode="after")
    def validate_complete_ledger(self) -> AssetGraphTerminalReceipt:
        if set(self.selected_leaf_receipts) != set(self.selected_leaf_ids) or len(
            self.selected_leaf_ids
        ) != len(set(self.selected_leaf_ids)):
            raise ValueError("terminal receipt selected ledger identity changed")
        if set(self.omitted_leaf_dispositions) != set(self.omitted_leaf_ids) or len(
            self.omitted_leaf_ids
        ) != len(set(self.omitted_leaf_ids)):
            raise ValueError("terminal receipt omitted ledger identity changed")
        if set(self.selected_leaf_ids).intersection(self.omitted_leaf_ids):
            raise ValueError("terminal selected and omitted ledgers overlap")
        if self.schema_version == LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION:
            return self
        parent_evidence = (
            self.parent_release_receipt,
            self.parent_command_receipt_journal,
            self.parent_command_receipt_checkpoint,
        )
        if any(item is None for item in parent_evidence):
            raise ValueError(
                "v2 terminal receipt requires parent release, journal, and checkpoint"
            )
        if (
            self.parent_release_receipt is not None
            and self.parent_release_receipt not in self.resource_release_receipts
        ):
            raise ValueError("parent release is missing from the resource ledger")
        return self

    @model_serializer(mode="wrap")
    def _serialize_wire_version(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        serialized = handler(self)
        if self.schema_version == LEGACY_ASSET_GRAPH_TERMINAL_RECEIPT_SCHEMA_VERSION:
            for field in (
                "parent_release_receipt",
                "parent_command_receipt_journal",
                "parent_command_receipt_checkpoint",
            ):
                serialized.pop(field, None)
        return serialized


class AssetRuntimeRequest(BaseModel):
    """Frozen child-agent and low-level scene-tool runtime settings."""

    model_config = ConfigDict(extra="forbid")

    runner: str
    model: str | None = None
    model_reasoning_effort: str | None = None
    scene_tool_timeout_seconds: float = Field(gt=0)
    child_timeout_seconds: float = Field(ge=0)
    codex_base_url: str | None = None
    codex_sandbox_mode: str
    codex_config: dict[str, Any] = Field(default_factory=dict)
    claude_config: dict[str, Any] = Field(default_factory=dict)
    claude_permission_mode: str
    claude_max_turns: int | None = Field(default=None, gt=0)
    claude_execution_mode: str


class AssetGeometryRequest(BaseModel):
    """Frozen policy for the deterministic Geometry stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_profile: str = Field(min_length=1)
    target_runtime: str = Field(default="isaac-lab", min_length=1)
    optimization_policy: Literal[
        "skip",
        "preserve_correspondence",
        "runtime_efficiency",
    ] = "preserve_correspondence"
    optimizer_backend: Literal["local", "remote"] = "local"
    repair_mode: Literal["off", "diagnose", "auto"] = "off"
    repair_profile: Literal[
        "visual_only",
        "static_environment",
        "rigid_pick_place",
        "articulated_rigid",
        "contact_rich",
        "deformable_or_cae",
    ] = "visual_only"
    render_evidence: bool = True
    segmentation_run: AssetSegmentationRunBinding | None = None
    segmentation_required: bool = False
    segmentation_required_parts: list[str] = Field(default_factory=list)
    runtime_validation_mode: Literal[
        "skip",
        "authored_physics",
        "temporary_loadability_proxy",
    ] = "skip"
    simready_mode: Literal["skip", "validate", "validate_and_route_conformance"] = (
        "skip"
    )

    @model_validator(mode="after")
    def validate_geometry_policy(self) -> AssetGeometryRequest:
        if len(self.segmentation_required_parts) != len(
            set(self.segmentation_required_parts)
        ):
            raise ValueError("segmentation_required_parts must not contain duplicates")
        for name in self.segmentation_required_parts:
            if (
                not name
                or len(name) > 2048
                or name in {".", ".."}
                or "/" in name
                or "\\" in name
                or any(ord(character) < 32 for character in name)
            ):
                raise ValueError(
                    "segmentation_required_parts must contain bounded plain names"
                )
        if self.segmentation_required_parts and not self.segmentation_required:
            raise ValueError(
                "segmentation_required_parts require segmentation_required=true"
            )
        return self


CadParameterScalar = str | int | float | bool


class AssetCadParameterVariant(BaseModel):
    """One named semantic parameter row evaluated during CAD modeling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(
        min_length=1,
        max_length=120,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    )
    parameter_values: dict[str, CadParameterScalar] = Field(min_length=1)

    @field_validator("parameter_values")
    @classmethod
    def validate_finite_parameter_values(
        cls,
        values: dict[str, CadParameterScalar],
    ) -> dict[str, CadParameterScalar]:
        return _finite_cad_parameter_values(values)


class LegacyAssetCadModelingRequest(BaseModel):
    """Read-only v4 policy retained for durable request compatibility."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = Field(min_length=1, max_length=240)
    model_backend: str | None = Field(default=None, min_length=1, max_length=120)
    quality_mode: Literal["fast", "balanced", "maximum"] = "balanced"
    repair_budget: int = Field(default=3, ge=0, le=8)
    compose_retry_budget: int | None = Field(default=None, ge=0, le=10)
    parameter_values: dict[str, CadParameterScalar] = Field(default_factory=dict)
    parameter_variants: list[AssetCadParameterVariant] = Field(
        default_factory=list,
        max_length=100,
    )
    required_outputs: list[Literal["step", "stl", "usd", "usda"]] = Field(
        default_factory=lambda: ["usd"],
        min_length=1,
        max_length=4,
    )

    @field_validator("parameter_values")
    @classmethod
    def validate_finite_parameter_values(
        cls,
        values: dict[str, CadParameterScalar],
    ) -> dict[str, CadParameterScalar]:
        return _finite_cad_parameter_values(values)

    @model_validator(mode="after")
    def validate_cad_modeling_policy(self) -> LegacyAssetCadModelingRequest:
        _validate_cad_parameter_policy(
            required_outputs=self.required_outputs,
            parameter_values=self.parameter_values,
            parameter_variants=self.parameter_variants,
        )
        return self


class AssetCadModelingRequest(BaseModel):
    """Frozen policy for provider-delegated CAD authoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    target_profile: str = Field(
        default="geometry-agent.static-visual-asset.v1",
        min_length=1,
        max_length=256,
    )
    parameter_values: dict[str, CadParameterScalar] = Field(default_factory=dict)
    parameter_variants: list[AssetCadParameterVariant] = Field(
        default_factory=list,
        max_length=100,
    )
    required_outputs: list[Literal["step", "stl", "usd", "usda"]] = Field(
        default_factory=lambda: ["usd"],
        min_length=1,
        max_length=4,
    )

    @field_validator("parameter_values")
    @classmethod
    def validate_finite_parameter_values(
        cls,
        values: dict[str, CadParameterScalar],
    ) -> dict[str, CadParameterScalar]:
        return _finite_cad_parameter_values(values)

    @model_validator(mode="after")
    def validate_cad_modeling_policy(self) -> AssetCadModelingRequest:
        _validate_cad_parameter_policy(
            required_outputs=self.required_outputs,
            parameter_values=self.parameter_values,
            parameter_variants=self.parameter_variants,
        )
        return self


def _finite_cad_parameter_values(
    values: dict[str, CadParameterScalar],
) -> dict[str, CadParameterScalar]:
    for name, value in values.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"CAD parameter {name!r} must be finite")
    return values


def _validate_cad_parameter_policy(
    *,
    required_outputs: list[str],
    parameter_values: dict[str, CadParameterScalar],
    parameter_variants: list[AssetCadParameterVariant],
) -> None:
    if len(required_outputs) != len(set(required_outputs)):
        raise ValueError("required_outputs must not contain duplicates")
    if not {"usd", "usda"}.intersection(required_outputs):
        raise ValueError("CAD modeling requires a USD output for Geometry")
    variant_ids = [variant.id for variant in parameter_variants]
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("parameter variant ids must be unique")
    canonical_rows = [
        tuple(sorted(variant.parameter_values.items()))
        for variant in parameter_variants
    ]
    if len(canonical_rows) != len(set(canonical_rows)):
        raise ValueError("parameter variants must contain distinct values")
    parameter_names = [
        *parameter_values,
        *[name for variant in parameter_variants for name in variant.parameter_values],
    ]
    if any(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}", name) is None
        for name in parameter_names
    ):
        raise ValueError("CAD parameter names must satisfy the authoring contract")


class AssetRunRequest(BaseModel):
    """Frozen user intent for agentic graphs or explicit fixed compatibility."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agents.asset-composition-request.v1",
        "content-agents.asset-composition-request.v2",
        "content-agents.asset-composition-request.v3",
        "content-agents.asset-composition-request.v4",
        "content-agents.asset-composition-request.v5",
    ] = PRE_CAD_MODELING_ASSET_REQUEST_SCHEMA_VERSION
    created_at: str
    workflow: str = "asset.run"
    coordinator_mode: Literal["single_reasoning_loop"] = "single_reasoning_loop"
    selected_mode: SelectedMode = "compatibility_fixed"
    run_id: str
    run_dir: str
    run_state: str
    repository_root: str
    source_asset: str
    source_mode: Literal["provided", "cad_modeling"] = "provided"
    source_staging: AssetSourceStaging | None = None
    cad_modeling: LegacyAssetCadModelingRequest | AssetCadModelingRequest | None = None
    source_images: list[str] = Field(default_factory=list, max_length=32)
    source_image_bindings: list[ArtifactBinding] = Field(
        default_factory=list,
        max_length=32,
    )
    prompt: str = Field(min_length=1)
    geometry: AssetGeometryRequest | None = None
    prompt_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    configuration_digest: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    reference_digest: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    sole_coordinator_identity: AssetSoleCoordinatorIdentity | None = None
    leaf_catalog: AssetLeafCatalog | None = None
    required_leaf_ids: list[str] = Field(default_factory=list, max_length=256)
    required_terminal_leaf_ids: list[str] = Field(
        default_factory=list,
        max_length=256,
    )
    required_leaf_dependencies: dict[str, list[str]] = Field(
        default_factory=dict,
        max_length=256,
    )
    exact_leaf_scope: bool = False
    requires_parent_resource_release: bool = False
    physics_validation_mode: PhysicsValidationMode = "runtime_required"
    joint_config: str | None = None
    joint_config_binding: ArtifactBinding | None = None
    materials_yaml: str | None = None
    materials_yaml_binding: ArtifactBinding | None = None
    materials_usd: str | None = None
    materials_usd_binding: ArtifactBinding | None = None
    materials_usd_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    reference_images: list[str] = Field(default_factory=list)
    reference_files: list[str] = Field(default_factory=list)
    reference_bindings: list[ArtifactBinding] = Field(default_factory=list)
    runtime: AssetRuntimeRequest

    @field_validator("required_leaf_ids", "required_terminal_leaf_ids")
    @classmethod
    def validate_required_leaf_ids(cls, values: list[str]) -> list[str]:
        if values != sorted(values) or len(values) != len(set(values)):
            raise ValueError("required leaf IDs must be unique and lexically sorted")
        return values

    @field_validator("required_leaf_dependencies")
    @classmethod
    def validate_required_leaf_dependencies(
        cls,
        values: dict[str, list[str]],
    ) -> dict[str, list[str]]:
        if list(values) != sorted(values):
            raise ValueError("required dependency owners must be lexically sorted")
        for leaf_id, dependencies in values.items():
            if dependencies != sorted(dependencies) or len(dependencies) != len(
                set(dependencies)
            ):
                raise ValueError(
                    "required leaf dependencies must be unique and lexically sorted: "
                    f"{leaf_id}"
                )
            if leaf_id in dependencies:
                raise ValueError(f"required leaf cannot depend on itself: {leaf_id}")
        return values

    @model_validator(mode="after")
    def validate_request_contract(self) -> AssetRunRequest:
        if self.source_images != [item.path for item in self.source_image_bindings]:
            raise ValueError("source images must match their frozen bindings")
        if len(self.source_images) != len(set(self.source_images)):
            raise ValueError("source images must not contain duplicates")
        if self.source_mode == "provided" and self.source_images:
            raise ValueError("provided-source requests cannot contain source images")
        if (
            self.source_staging is not None
            and self.source_asset != self.source_staging.staged_source.path
        ):
            raise ValueError(
                "asset request source must be the staged run-confined source"
            )
        if self.schema_version == ASSET_REQUEST_SCHEMA_VERSION:
            if self.source_staging is None:
                raise ValueError("asset request v3 requires source staging identity")
            if self.selected_mode != "agentic":
                raise ValueError("asset request v3 requires selected_mode=agentic")
            if (
                self.source_mode != "provided"
                or self.geometry is not None
                or self.cad_modeling is not None
                or self.source_images
                or self.source_image_bindings
            ):
                raise ValueError(
                    "agentic graph request cannot embed fixed Geometry or CAD policy"
                )
            if self.sole_coordinator_identity is None or self.leaf_catalog is None:
                raise ValueError(
                    "agentic request requires coordinator identity and leaf catalog"
                )
            catalog_ids = {
                descriptor.leaf_id for descriptor in self.leaf_catalog.descriptors
            }
            unknown_required = sorted(
                set(self.required_leaf_ids).difference(catalog_ids)
            )
            if unknown_required:
                raise ValueError(
                    "agentic request requires leaves absent from its frozen catalog: "
                    f"{unknown_required}"
                )
            nonrequired_terminals = sorted(
                set(self.required_terminal_leaf_ids).difference(self.required_leaf_ids)
            )
            if nonrequired_terminals:
                raise ValueError(
                    "required terminal leaves must also be required leaves: "
                    f"{nonrequired_terminals}"
                )
            required_dependency_ids = {
                leaf_id
                for owner, dependencies in self.required_leaf_dependencies.items()
                for leaf_id in (owner, *dependencies)
            }
            unknown_dependencies = sorted(
                required_dependency_ids.difference(catalog_ids)
            )
            if unknown_dependencies:
                raise ValueError(
                    "agentic request requires dependency leaves absent from its frozen "
                    f"catalog: {unknown_dependencies}"
                )
            nonrequired_dependencies = sorted(
                required_dependency_ids.difference(self.required_leaf_ids)
            )
            if nonrequired_dependencies:
                raise ValueError(
                    "required dependency edges must connect request-required leaves: "
                    f"{nonrequired_dependencies}"
                )
            if self.exact_leaf_scope and not self.required_leaf_ids:
                raise ValueError(
                    "exact leaf scope requires at least one request-required leaf"
                )
            configuration_identity: dict[str, Any] = {
                "repository_root": self.repository_root,
                "runtime": self.runtime.model_dump(mode="json"),
                "requires_parent_resource_release": (
                    self.requires_parent_resource_release
                ),
            }
            if (
                self.required_leaf_ids
                or self.required_terminal_leaf_ids
                or self.required_leaf_dependencies
            ):
                configuration_identity.update(
                    {
                        "required_leaf_ids": self.required_leaf_ids,
                        "required_terminal_leaf_ids": (self.required_terminal_leaf_ids),
                        "required_leaf_dependencies": (self.required_leaf_dependencies),
                    }
                )
            if self.exact_leaf_scope:
                configuration_identity["exact_leaf_scope"] = True
            expected_digests = {
                "prompt_digest": hashlib.sha256(
                    self.prompt.encode("utf-8")
                ).hexdigest(),
                "source_digest": self.source_staging.staged_source.sha256,
                "configuration_digest": _canonical_digest(configuration_identity),
                "reference_digest": _canonical_digest(
                    [
                        binding.model_dump(mode="json")
                        for binding in self.reference_bindings
                    ]
                ),
            }
            for name, expected in expected_digests.items():
                if getattr(self, name) != expected:
                    raise ValueError(f"asset request {name} is stale")
            if (
                any(
                    value is not None
                    for value in (
                        self.joint_config,
                        self.joint_config_binding,
                        self.materials_yaml,
                        self.materials_yaml_binding,
                        self.materials_usd,
                        self.materials_usd_binding,
                    )
                )
                or self.materials_usd_dependencies
            ):
                raise ValueError(
                    "agentic graph request cannot embed fixed domain configuration"
                )
            return self

        if self.selected_mode != "compatibility_fixed":
            raise ValueError(
                "fixed compatibility requests require selected_mode=compatibility_fixed"
            )
        if (
            any(
                value is not None
                for value in (
                    self.prompt_digest,
                    self.source_digest,
                    self.configuration_digest,
                    self.reference_digest,
                    self.sole_coordinator_identity,
                    self.leaf_catalog,
                )
            )
            or self.required_leaf_ids
            or self.required_terminal_leaf_ids
            or self.required_leaf_dependencies
            or self.requires_parent_resource_release
        ):
            raise ValueError(
                "fixed compatibility request cannot carry agentic graph identity"
            )
        if (
            self.joint_config is None
            or self.joint_config_binding is None
            or self.materials_yaml is None
            or self.materials_yaml_binding is None
        ):
            raise ValueError(
                "fixed compatibility request requires Joint and Material inputs"
            )
        if self.source_mode == "provided" and self.cad_modeling is not None:
            raise ValueError(
                "provided-source fixed requests cannot contain CAD modeling policy"
            )
        if self.source_mode == "cad_modeling" and self.cad_modeling is None:
            raise ValueError("CAD-modeling fixed requests require CAD modeling policy")

        if self.schema_version == LEGACY_ASSET_REQUEST_SCHEMA_VERSION:
            if self.source_staging is not None:
                raise ValueError("legacy asset request cannot carry v2 source staging")
            if self.geometry is not None:
                raise ValueError("v1 asset requests cannot contain Geometry policy")
            if self.source_mode != "provided" or self.cad_modeling is not None:
                raise ValueError("v1 asset requests require a provided source")
        elif self.schema_version == PRE_CAD_MODELING_ASSET_REQUEST_SCHEMA_VERSION:
            if self.source_staging is None:
                raise ValueError("asset request v2 requires source staging identity")
            if self.geometry is not None or self.cad_modeling is not None:
                raise ValueError(
                    "asset request v2 cannot contain Geometry or CAD modeling policy"
                )
            if self.source_mode != "provided":
                raise ValueError("asset request v2 requires a provided source")
        else:
            if self.geometry is None:
                raise ValueError("fixed compatibility requests require Geometry policy")
            if self.source_mode == "provided":
                if self.source_staging is None:
                    raise ValueError(
                        "provided-source fixed requests require source staging identity"
                    )
            elif self.source_staging is not None:
                raise ValueError(
                    "CAD-modeling fixed requests cannot carry provided-source staging"
                )
            if (
                self.schema_version
                == PRE_PROVIDER_AUTHORING_ASSET_REQUEST_SCHEMA_VERSION
                and isinstance(self.cad_modeling, AssetCadModelingRequest)
            ):
                raise ValueError(
                    "asset request v4 requires the pre-provider CAD modeling policy"
                )
            if (
                self.schema_version == COMPATIBILITY_FIXED_ASSET_REQUEST_SCHEMA_VERSION
                and isinstance(self.cad_modeling, LegacyAssetCadModelingRequest)
            ):
                raise ValueError(
                    "asset request v5 requires provider-delegated CAD modeling policy"
                )
        return self

    @model_serializer(mode="wrap")
    def _serialize_mode_shape(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep agentic and fixed compatibility request shapes disjoint."""

        serialized = handler(self)
        agentic_fields = (
            "selected_mode",
            "prompt_digest",
            "source_digest",
            "configuration_digest",
            "reference_digest",
            "sole_coordinator_identity",
            "leaf_catalog",
            "required_leaf_ids",
            "required_terminal_leaf_ids",
            "required_leaf_dependencies",
            "exact_leaf_scope",
            "requires_parent_resource_release",
        )
        fixed_fields = (
            "source_mode",
            "cad_modeling",
            "source_images",
            "source_image_bindings",
            "geometry",
            "physics_validation_mode",
            "joint_config",
            "joint_config_binding",
            "materials_yaml",
            "materials_yaml_binding",
            "materials_usd",
            "materials_usd_binding",
            "materials_usd_dependencies",
        )
        if self.schema_version == ASSET_REQUEST_SCHEMA_VERSION:
            for field in fixed_fields:
                serialized.pop(field, None)
            if not self.required_leaf_ids:
                serialized.pop("required_leaf_ids", None)
            if not self.required_terminal_leaf_ids:
                serialized.pop("required_terminal_leaf_ids", None)
            if not self.required_leaf_dependencies:
                serialized.pop("required_leaf_dependencies", None)
            if not self.exact_leaf_scope:
                serialized.pop("exact_leaf_scope", None)
        else:
            for field in agentic_fields:
                if field == "selected_mode" and self.schema_version != (
                    LEGACY_ASSET_REQUEST_SCHEMA_VERSION
                ):
                    continue
                serialized.pop(field, None)
            if (
                self.schema_version
                in {
                    LEGACY_ASSET_REQUEST_SCHEMA_VERSION,
                    PRE_CAD_MODELING_ASSET_REQUEST_SCHEMA_VERSION,
                }
                and self.source_mode == "provided"
            ):
                serialized.pop("source_mode", None)
            if self.source_staging is None:
                serialized.pop("source_staging", None)
            if self.cad_modeling is None:
                serialized.pop("cad_modeling", None)
            if not self.source_images:
                serialized.pop("source_images", None)
            if not self.source_image_bindings:
                serialized.pop("source_image_bindings", None)
            if self.geometry is None:
                serialized.pop("geometry", None)
        return serialized


class AssetCoordinatorPlanStep(BaseModel):
    """One concise, evidence-verifiable step in a coordinator plan."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: StageName
    objective: str = Field(min_length=1)
    acceptance_evidence: list[str] = Field(min_length=1, max_length=32)
    may_revisit: bool = True


class AssetCoordinatorPlanDraft(BaseModel):
    """Agent-authored plan input before run identities are sealed around it."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.asset-coordinator-plan-draft.v1"
    ] = ASSET_COORDINATOR_PLAN_DRAFT_SCHEMA_VERSION
    stage: StageName
    objective: str = Field(min_length=1)
    steps: list[AssetCoordinatorPlanStep] = Field(min_length=1, max_length=24)
    evidence_paths: list[str] = Field(min_length=1, max_length=64)
    revision_reason: str = Field(min_length=1)


class AssetCoordinatorPlan(BaseModel):
    """Immutable plan revision bound to the frozen run and inspected evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.asset-coordinator-plan.v1"] = (
        ASSET_COORDINATOR_PLAN_SCHEMA_VERSION
    )
    plan_revision: int = Field(ge=1)
    run_revision: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    request: ArtifactBinding
    source_asset: ArtifactBinding
    stage: StageName
    stage_attempt: int = Field(ge=1)
    objective: str = Field(min_length=1)
    steps: list[AssetCoordinatorPlanStep] = Field(min_length=1, max_length=24)
    evidence: list[ArtifactBinding] = Field(min_length=1, max_length=64)
    revision_reason: str = Field(min_length=1)
    prior_plan: ArtifactBinding | None = None
    timestamp: str
    actor: str = Field(min_length=1)


class AssetCoordinatorReviewDraft(BaseModel):
    """Agent-authored evidence review and bounded next-action decision."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.asset-coordinator-review-draft.v1"
    ] = ASSET_COORDINATOR_REVIEW_DRAFT_SCHEMA_VERSION
    stage: StageName
    output_asset_path: str | None = None
    evidence_paths: list[str] = Field(
        min_length=1, max_length=MAX_COORDINATOR_REVIEW_EVIDENCE
    )
    findings: list[str] = Field(min_length=1, max_length=64)
    decision: CoordinatorDecision
    target_stage: StageName | None = None
    decision_summary: str = Field(min_length=1)
    repair_scope: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def validate_decision_contract(self) -> AssetCoordinatorReviewDraft:
        if self.decision == "revisit" and self.target_stage is None:
            raise ValueError("revisit decision requires target_stage")
        if self.decision != "revisit" and self.target_stage is not None:
            raise ValueError("target_stage is accepted only for revisit")
        if self.decision == "accept" and self.output_asset_path is None:
            raise ValueError("accept decision requires output_asset_path")
        return self


class AssetCoordinatorEvidenceReview(BaseModel):
    """Immutable evidence review produced by the single reasoning loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["content-agent-workflows.asset-coordinator-review.v1"] = (
        ASSET_COORDINATOR_REVIEW_SCHEMA_VERSION
    )
    review_index: int = Field(ge=1)
    run_revision: int = Field(ge=0)
    run_id: str = Field(min_length=1)
    request: ArtifactBinding
    stage: StageName
    stage_attempt: int = Field(ge=1)
    input_asset: ArtifactBinding
    output_asset: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    evidence: list[ArtifactBinding] = Field(
        min_length=1, max_length=MAX_COORDINATOR_REVIEW_EVIDENCE
    )
    plan: ArtifactBinding
    articulation_review_decisions: ArtifactBinding | None = None
    findings: list[str] = Field(min_length=1, max_length=64)
    decision: CoordinatorDecision
    target_stage: StageName | None = None
    decision_summary: str = Field(min_length=1)
    repair_scope: list[str] = Field(default_factory=list, max_length=32)
    prior_review: ArtifactBinding | None = None
    timestamp: str
    actor: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_decision_contract(self) -> AssetCoordinatorEvidenceReview:
        if self.decision == "revisit" and self.target_stage is None:
            raise ValueError("revisit decision requires target_stage")
        if self.decision != "revisit" and self.target_stage is not None:
            raise ValueError("target_stage is accepted only for revisit")
        if self.decision == "accept" and self.output_asset is None:
            raise ValueError("accept decision requires output_asset")
        if self.output_asset is None and self.output_dependencies:
            raise ValueError("output dependencies require output_asset")
        return self


class SupersededArticulationReview(BaseModel):
    """One immutable prior human-review cycle superseded by a graph revision."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    superseded_at: str
    revision_receipt: ArtifactBinding
    review_candidates: ArtifactBinding
    review_decisions: ArtifactBinding
    actor: str = Field(min_length=1)


class SupersededStageAttempt(BaseModel):
    """Accepted or partial stage state archived before a bounded revisit."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    archived_at: str
    reason_review: ArtifactBinding
    status: StageStatus
    attempt_count: int = Field(ge=0)
    input_asset: ArtifactBinding | None = None
    input_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    input_readiness: Literal["yes", "conditional"] = "yes"
    output_asset: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    handoff: ArtifactBinding | None = None
    evidence: list[ArtifactBinding] = Field(default_factory=list)
    review_candidates: ArtifactBinding | None = None
    review_decisions: ArtifactBinding | None = None
    superseded_reviews: list[SupersededArticulationReview] = Field(default_factory=list)
    error: str | None = None


class AssetCoordinatorState(BaseModel):
    """Durable state owned by the single cross-domain reasoning loop."""

    model_config = ConfigDict(extra="forbid")

    mode: CoordinatorMode = "legacy"
    max_plan_revisions: int = Field(default=48, ge=1, le=128)
    max_evidence_reviews: int = Field(default=48, ge=1, le=128)
    max_refinements_per_stage: int = Field(default=3, ge=0, le=16)
    max_revisits: int = Field(default=3, ge=0, le=16)
    plan_revisions: list[ArtifactBinding] = Field(default_factory=list)
    evidence_reviews: list[ArtifactBinding] = Field(default_factory=list)
    refinement_counts: dict[StageName, int] = Field(default_factory=dict)
    revisit_count: int = Field(default=0, ge=0)
    next_action: CoordinatorNextAction = "plan"
    stop_reason: str | None = None

    @model_validator(mode="after")
    def validate_budget_contract(self) -> AssetCoordinatorState:
        if len(self.plan_revisions) > self.max_plan_revisions:
            raise ValueError("coordinator plan revision budget exceeded")
        if len(self.evidence_reviews) > self.max_evidence_reviews:
            raise ValueError("coordinator evidence review budget exceeded")
        if self.revisit_count > self.max_revisits:
            raise ValueError("coordinator revisit budget exceeded")
        if any(
            count < 0 or count > self.max_refinements_per_stage
            for count in self.refinement_counts.values()
        ):
            raise ValueError("coordinator refinement budget exceeded")
        if self.next_action == "stopped" and not self.stop_reason:
            raise ValueError("stopped coordinator requires stop_reason")
        return self


class AssetStageHandoff(BaseModel):
    """Digest-bound output contract published by one completed stage."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.asset-stage-handoff.v1",
        "content-agent-workflows.asset-stage-handoff.v2",
    ] = ASSET_STAGE_HANDOFF_SCHEMA_VERSION
    stage: StageName
    input_asset: ArtifactBinding
    input_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    output_asset: ArtifactBinding
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    evidence: list[ArtifactBinding] = Field(min_length=1)
    readiness: Literal["yes", "conditional"] = "yes"
    summary: str = Field(min_length=1)

    @model_serializer(mode="wrap")
    def _serialize_v1_shape(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        serialized = handler(self)
        if self.schema_version == LEGACY_ASSET_STAGE_HANDOFF_SCHEMA_VERSION:
            serialized.pop("readiness", None)
        return serialized


class AssetCadModelingStageResult(BaseModel):
    """Digest-bound result emitted by a selected authoring provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-cad-modeling-stage-result.v2"
    ] = ASSET_CAD_MODELING_STAGE_RESULT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    run_revision: int = Field(ge=0)
    stage_attempt: int = Field(ge=1)
    request: ArtifactBinding
    plan: ArtifactBinding
    input_asset: ArtifactBinding
    input_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    result_path: str
    authoring_request: ArtifactBinding
    source_manifest: ArtifactBinding | None = None
    parameter_family: ArtifactBinding | None = None
    output_asset: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    artifacts: dict[str, ArtifactBinding] = Field(default_factory=dict)
    source_bundle_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    source_revision: str | None = None
    source_provider_id: str | None = None
    source_representation_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$",
    )
    parameter_values: dict[str, CadParameterScalar] = Field(default_factory=dict)
    success: bool
    error: str | None = None
    timestamp: str
    actor: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result_contract(self) -> AssetCadModelingStageResult:
        if self.success:
            required = {
                "authoring_request",
                "generated_usd",
                "source_manifest",
            }
            missing = sorted(required.difference(self.artifacts))
            if missing:
                raise ValueError(
                    f"successful CAD modeling result lacks required artifacts: {missing}"
                )
            if (
                self.source_manifest is None
                or self.output_asset is None
                or self.source_bundle_id is None
                or self.source_revision is None
                or self.source_provider_id is None
                or self.source_representation_id is None
            ):
                raise ValueError(
                    "successful CAD modeling result requires source and output identities"
                )
            if self.artifacts["authoring_request"] != self.authoring_request:
                raise ValueError("Authoring request artifact is inconsistent")
            if self.artifacts["source_manifest"] != self.source_manifest:
                raise ValueError("Authoring source manifest is inconsistent")
            if self.artifacts["generated_usd"] != self.output_asset:
                raise ValueError("CAD generated USD artifact is inconsistent")
        elif not self.error:
            raise ValueError("failed CAD modeling result requires an error")
        if self.output_asset is None and self.output_dependencies:
            raise ValueError("CAD modeling output dependencies require output_asset")
        return self


class AssetGeometryStageResult(BaseModel):
    """Digest-bound result emitted by the composed Geometry executor."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-geometry-stage-result.v1"
    ] = ASSET_GEOMETRY_STAGE_RESULT_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    run_revision: int = Field(ge=0)
    stage_attempt: int = Field(ge=1)
    request: ArtifactBinding
    plan: ArtifactBinding
    input_asset: ArtifactBinding
    input_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    result_path: str
    workflow_result: ArtifactBinding
    output_asset: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    artifacts: dict[str, ArtifactBinding] = Field(default_factory=dict)
    success: bool
    validation_status: Literal["pass", "conditional", "fail", "not_evaluated"]
    handoff_ready: Literal["yes", "conditional", "no"]
    error: str | None = None
    timestamp: str
    actor: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_result_contract(self) -> AssetGeometryStageResult:
        required = {
            "handoff_manifest",
            "validation_evidence",
            "evidence_bundle",
            "optimization_metadata",
            "usd_validation",
        }
        if self.success:
            if self.output_asset is None:
                raise ValueError("successful Geometry result requires output_asset")
            if self.validation_status not in {"pass", "conditional"}:
                raise ValueError(
                    "successful Geometry result requires pass or conditional validation"
                )
            if self.handoff_ready not in {"yes", "conditional"}:
                raise ValueError(
                    "successful Geometry result requires an admissible handoff"
                )
            missing = sorted(required.difference(self.artifacts))
            if missing:
                raise ValueError(
                    f"successful Geometry result lacks required artifacts: {missing}"
                )
        else:
            if self.handoff_ready != "no" or self.validation_status not in {
                "fail",
                "not_evaluated",
            }:
                raise ValueError("failed Geometry result must remain non-admissible")
            if not self.error:
                raise ValueError("failed Geometry result requires an error")
        if self.output_asset is None and self.output_dependencies:
            raise ValueError("Geometry output dependencies require output_asset")
        return self


class AssetCrossStageClaim(BaseModel):
    """One coordinator-reviewed claim backed by sealed domain evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: CrossStageClaimName
    status: CrossStageClaimStatus
    summary: str = Field(min_length=1)
    evidence: list[ArtifactBinding] = Field(min_length=1, max_length=32)


class AssetCrossStageValidation(BaseModel):
    """Typed cross-domain receipt authored by the single coordinator loop."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-cross-stage-validation.v1"
    ] = ASSET_CROSS_STAGE_VALIDATION_SCHEMA_VERSION
    run_id: str = Field(min_length=1)
    validation_input: ArtifactBinding
    accepted_handoffs: dict[CrossStageHandoffName, ArtifactBinding]
    claims: list[AssetCrossStageClaim] = Field(min_length=5, max_length=5)
    source_unchanged: Literal[True] = True

    @model_validator(mode="after")
    def validate_complete_claim_scope(self) -> AssetCrossStageValidation:
        required_handoffs = {"articulation", "material", "texture", "physics"}
        if set(self.accepted_handoffs) != required_handoffs:
            raise ValueError(
                "accepted_handoffs must cover articulation through physics"
            )
        required_claims = {
            "joint_graph",
            "appearance",
            "physics_behavior",
            "render_and_package",
            "non_target_preservation",
        }
        claim_names = [claim.name for claim in self.claims]
        if len(claim_names) != len(set(claim_names)) or set(claim_names) != (
            required_claims
        ):
            raise ValueError("claims must cover every cross-stage acceptance claim")
        return self


class StageState(BaseModel):
    """Current durable state and accepted evidence for one stage."""

    model_config = ConfigDict(extra="forbid")

    status: StageStatus = "pending"
    attempt_count: int = Field(default=0, ge=0)
    continue_current_attempt: bool = False
    input_asset: ArtifactBinding | None = None
    input_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    input_readiness: Literal["yes", "conditional"] = "yes"
    output_asset: ArtifactBinding | None = None
    output_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    handoff: ArtifactBinding | None = None
    evidence: list[ArtifactBinding] = Field(default_factory=list)
    review_candidates: ArtifactBinding | None = None
    review_decisions: ArtifactBinding | None = None
    error: str | None = None
    superseded_reviews: list[SupersededArticulationReview] = Field(default_factory=list)
    superseded_attempts: list[SupersededStageAttempt] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_status_contract(self) -> StageState:
        if self.status in {"ready", "running", "needs_review", "completed"}:
            if self.input_asset is None:
                raise ValueError(f"{self.status} stage requires input_asset")
        if self.status == "needs_review" and self.review_candidates is None:
            raise ValueError("needs_review stage requires review_candidates")
        if self.continue_current_attempt and (
            self.status not in {"ready", "failed", "cancelled"}
            or self.attempt_count < 1
        ):
            raise ValueError(
                "attempt continuation requires a resumable attempted stage"
            )
        if self.status == "completed" and (
            self.output_asset is None or self.handoff is None
        ):
            raise ValueError("completed stage requires output_asset and handoff")
        if self.status == "completed" and not self.evidence:
            raise ValueError("completed stage requires evidence")
        if self.status in {"failed", "cancelled"} and not self.error:
            raise ValueError(f"{self.status} stage requires an error or reason")
        return self


class StageTransition(BaseModel):
    """Append-only audit entry for one coordinator state change."""

    model_config = ConfigDict(extra="forbid")

    timestamp: str
    stage: StageName
    from_status: StageStatus
    to_status: StageStatus
    reason: str = Field(min_length=1)
    actor: str = Field(min_length=1)
    attempt_count: int = Field(ge=0)
    input_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    output_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class AssetCompositionRun(BaseModel):
    """Durable top-level coordinator state shared across agent sessions."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.asset-composition-run.v1",
        "content-agent-workflows.asset-composition-run.v2",
        "content-agent-workflows.asset-composition-run.v3",
        "content-agent-workflows.asset-composition-run.v4",
        "content-agent-workflows.asset-composition-run.v5",
        "content-agent-workflows.asset-composition-run.v6",
    ] = COMPATIBILITY_FIXED_ASSET_COMPOSITION_RUN_SCHEMA_VERSION
    revision: int = Field(default=0, ge=0)
    run_id: str = Field(min_length=1)
    request: ArtifactBinding
    source_asset: ArtifactBinding
    source_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    stage_order: tuple[StageName, ...] = LEGACY_STAGE_ORDER
    selected_mode: SelectedMode = "compatibility_fixed"
    execution_graph: ArtifactBinding | None = None
    graph_started_at: str | None = None
    current_leaf_id: str | None = None
    leaf_states: dict[str, AssetLeafState] = Field(default_factory=dict)
    leaf_transitions: list[AssetLeafTransition] = Field(default_factory=list)
    graph_terminal_receipt: ArtifactBinding | None = None
    current_stage: StageName | None = "articulation"
    terminal_status: TerminalStatus = "active"
    stages: dict[StageName, StageState] = Field(default_factory=dict)
    transitions: list[StageTransition] = Field(default_factory=list)
    coordinator: AssetCoordinatorState = Field(default_factory=AssetCoordinatorState)

    @model_validator(mode="before")
    @classmethod
    def infer_unserialized_legacy_stage_order(cls, value: Any) -> Any:
        """Recover fixed stage order omitted by the original strict schemas."""

        if not isinstance(value, dict) or "stage_order" in value:
            return value
        schema_version = value.get("schema_version")
        if schema_version in {
            LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
        }:
            inferred_order = LEGACY_STAGE_ORDER
        elif schema_version == PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION:
            stages = value.get("stages")
            stage_names = set(stages) if isinstance(stages, dict) else set()
            inferred_order = (
                GEOMETRY_STAGE_ORDER
                if stage_names == set(GEOMETRY_STAGE_ORDER)
                else LEGACY_STAGE_ORDER
            )
        else:
            return value
        return {**value, "stage_order": inferred_order}

    @model_validator(mode="after")
    def validate_run_contract(self) -> AssetCompositionRun:
        if self.schema_version == ASSET_COMPOSITION_RUN_SCHEMA_VERSION:
            if self.selected_mode != "agentic":
                raise ValueError("run-state v4 requires selected_mode=agentic")
            if self.coordinator.mode != "single_reasoning_loop":
                raise ValueError(
                    "agentic graph run requires its sole outer coordinator"
                )
            if self.stage_order != LEGACY_STAGE_ORDER:
                raise ValueError("agentic graph run cannot redefine fixed stage_order")
            if self.stages or self.current_stage is not None:
                raise ValueError("agentic graph run cannot contain fixed stages")
            if self.transitions:
                raise ValueError(
                    "agentic graph run cannot contain fixed stage transitions"
                )
            if self.execution_graph is None:
                if self.leaf_states or self.current_leaf_id is not None:
                    raise ValueError("unfrozen graph run cannot contain leaf progress")
                if self.graph_started_at is not None:
                    raise ValueError("unfrozen graph run cannot have graph_started_at")
                if self.graph_terminal_receipt is not None:
                    raise ValueError("unfrozen graph run cannot have terminal receipt")
                if self.terminal_status != "active":
                    raise ValueError("unfrozen graph run must remain active")
                if self.coordinator.next_action != "freeze_graph":
                    raise ValueError("unfrozen graph run must request freeze_graph")
            else:
                if self.graph_started_at is None:
                    raise ValueError("frozen graph run requires graph_started_at")
                if not self.leaf_states:
                    raise ValueError("frozen graph run requires selected leaf states")
                if (
                    self.current_leaf_id is not None
                    and self.current_leaf_id not in self.leaf_states
                ):
                    raise ValueError("current graph leaf is absent from leaf states")
                if self.terminal_status == "completed":
                    if self.current_leaf_id is not None:
                        raise ValueError("completed graph run cannot have current leaf")
                    if self.graph_terminal_receipt is None:
                        raise ValueError(
                            "completed graph run requires terminal receipt"
                        )
                    if self.coordinator.next_action != "terminal":
                        raise ValueError("completed graph coordinator must be terminal")
                elif self.terminal_status in {"failed", "cancelled"}:
                    if self.current_leaf_id is None:
                        raise ValueError("stopped graph run requires current leaf")
                    if (
                        self.leaf_states[self.current_leaf_id].status
                        != self.terminal_status
                    ):
                        raise ValueError(
                            "stopped graph status differs from current leaf"
                        )
                    if self.coordinator.next_action != "stopped":
                        raise ValueError("stopped graph coordinator must be stopped")
                    if self.graph_terminal_receipt is not None:
                        raise ValueError(
                            "stopped graph run cannot have terminal receipt"
                        )
                elif self.coordinator.next_action == "finalize_receipts":
                    if self.current_leaf_id is not None:
                        raise ValueError(
                            "receipt finalization cannot retain a current leaf"
                        )
                elif self.coordinator.next_action == "begin_leaf":
                    if self.current_leaf_id is not None and (
                        self.leaf_states[self.current_leaf_id].status != "ready"
                    ):
                        raise ValueError("begin_leaf current leaf must be ready")
                    if self.current_leaf_id is None and not any(
                        state.status == "ready" for state in self.leaf_states.values()
                    ):
                        raise ValueError(
                            "begin_leaf requires at least one dependency-ready leaf"
                        )
                elif self.coordinator.next_action == "execute_leaf":
                    if self.current_leaf_id is None:
                        raise ValueError("execute_leaf requires a current leaf")
                    if self.leaf_states[self.current_leaf_id].status != "running":
                        raise ValueError("execute_leaf current leaf must be running")
                else:
                    raise ValueError(
                        "active graph coordinator action does not match graph state"
                    )
            return self
        if self.selected_mode != "compatibility_fixed":
            raise ValueError("legacy run schemas require compatibility_fixed mode")
        if (
            self.execution_graph is not None
            or self.graph_started_at is not None
            or self.current_leaf_id is not None
            or self.leaf_states
            or self.leaf_transitions
            or self.graph_terminal_receipt is not None
        ):
            raise ValueError("legacy run cannot carry agentic graph state")
        if self.stage_order not in SUPPORTED_STAGE_ORDERS:
            raise ValueError("stage_order is not a supported composed workflow order")
        if (
            self.schema_version
            in {
                LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
                PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            }
            and self.stage_order != LEGACY_STAGE_ORDER
        ):
            raise ValueError("v1/v2 runs require the legacy stage order")
        if (
            self.schema_version == PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION
            and self.stage_order not in {LEGACY_STAGE_ORDER, GEOMETRY_STAGE_ORDER}
        ):
            raise ValueError("v3 runs cannot contain the CAD modeling stage")
        if self.schema_version in {
            LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
        } and any(
            stage.superseded_reviews
            or any(attempt.superseded_reviews for attempt in stage.superseded_attempts)
            for stage in self.stages.values()
        ):
            raise ValueError(
                "superseded Articulation reviews require run-state v5 or v6"
            )
        if set(self.stages) != set(self.stage_order):
            raise ValueError(f"stages must contain exactly {list(self.stage_order)}")
        if self.terminal_status == "completed":
            if self.current_stage is not None:
                raise ValueError("completed run cannot have current_stage")
            if any(
                self.stages[stage].status != "completed" for stage in self.stage_order
            ):
                raise ValueError("completed run requires every stage completed")
            if (
                self.coordinator.mode == "single_reasoning_loop"
                and self.coordinator.next_action != "terminal"
            ):
                raise ValueError("completed reasoning coordinator must be terminal")
        elif self.current_stage is None:
            raise ValueError("non-completed run requires current_stage")
        elif self.stages[self.current_stage].status == "pending":
            raise ValueError("current_stage cannot be pending")
        return self

    @model_serializer(mode="wrap")
    def _serialize_legacy_run_shapes(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        """Keep legacy bytes readable by their original strict schemas."""

        serialized = handler(self)
        if self.schema_version != ASSET_COMPOSITION_RUN_SCHEMA_VERSION:
            for field in (
                "execution_graph",
                "graph_started_at",
                "current_leaf_id",
                "leaf_states",
                "leaf_transitions",
                "graph_terminal_receipt",
            ):
                serialized.pop(field, None)
        else:
            serialized.pop("stage_order", None)
        if self.schema_version in {
            LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_SELECTED_MODE_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
        }:
            serialized.pop("selected_mode", None)
            if self.schema_version in {
                LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
                PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
                PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            }:
                for stage in serialized["stages"].values():
                    stage.pop("superseded_reviews", None)
                    for attempt in stage.get("superseded_attempts", []):
                        attempt.pop("superseded_reviews", None)
        if self.schema_version in {
            LEGACY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_GEOMETRY_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
            PRE_CAD_MODELING_ASSET_COMPOSITION_RUN_SCHEMA_VERSION,
        }:
            serialized.pop("stage_order", None)
            for stage in serialized["stages"].values():
                stage.pop("input_readiness", None)
                for attempt in stage.get("superseded_attempts", []):
                    attempt.pop("input_readiness", None)
        return serialized


class AssetTerminalValidation(BaseModel):
    """Machine-readable proof that the composed run is terminal and intact."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[
        "content-agent-workflows.asset-terminal-validation.v1",
        "content-agent-workflows.asset-terminal-validation.v2",
    ] = ASSET_TERMINAL_VALIDATION_SCHEMA_VERSION
    valid: bool
    terminal_status: TerminalStatus
    current_stage: StageName | None
    current_leaf_id: str | None = None
    final_asset: ArtifactBinding | None = None
    graph: ArtifactBinding | None = None
    terminal_receipt: ArtifactBinding | None = None
    errors: list[str] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def _serialize_v1_shape(  # type: ignore[no-untyped-def]
        self,
        handler: SerializerFunctionWrapHandler,
    ):
        serialized = handler(self)
        if (
            self.schema_version
            == "content-agent-workflows.asset-terminal-validation.v1"
        ):
            serialized.pop("current_leaf_id", None)
            serialized.pop("graph", None)
            serialized.pop("terminal_receipt", None)
        return serialized


class AssetCombinedReport(BaseModel):
    """Digest-bound index of the final package and every accepted handoff."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["content-agent-workflows.asset-combined-report.v1"] = (
        ASSET_COMBINED_REPORT_SCHEMA_VERSION
    )
    run_id: str = Field(min_length=1)
    request: ArtifactBinding
    source_asset: ArtifactBinding
    source_dependencies: list[ArtifactBinding] = Field(default_factory=list)
    final_asset: ArtifactBinding
    stage_handoffs: dict[StageName, ArtifactBinding]
    validation_summary: ArtifactBinding
