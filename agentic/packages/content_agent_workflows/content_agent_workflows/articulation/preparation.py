# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic provider-neutral Articulation preparation publication.

The public producer consumes one saved, typed inspector readback.  It does not
accept caller-supplied member or owner lists: both are derived from the exact
retained configuration authority and must match the saved per-prim observation.
All inputs are descriptor-confined to one retained root and every publication
is written to a fresh directory.
"""

from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_regular_file,
    write_bytes_to_confined,
)

from content_agent_workflows.common.artifacts import (
    ContainedArtifactRead,
    file_sha256,
    prepare_writable_directory,
    read_contained_artifact,
)
from content_agent_workflows.common.domain_execution import ExecutionArtifactBinding
from content_agent_workflows.common.embedded_domain_decision import (
    ProducerIdentity,
    ProviderNeutralEvidenceRecord,
    canonical_json_digest,
)

from .embedded_decision import (
    EmbeddedArticulationCapabilityLimits,
    EmbeddedArticulationError,
    EmbeddedArticulationPreparation,
    OptionalProposalStatus,
)

ARTICULATION_PREPARATION_READBACK_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-readback.v1"
] = "content-agent-workflows.articulation-preparation-readback.v1"
ARTICULATION_PREPARATION_READBACK_DRAFT_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-readback-draft.v2"
] = "content-agent-workflows.articulation-preparation-readback-draft.v2"
ARTICULATION_PREPARATION_CONFIGURATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-configuration.v1"
] = "content-agent-workflows.articulation-preparation-configuration.v1"
ARTICULATION_PREPARATION_PUBLICATION_SCHEMA_VERSION: Literal[
    "content-agent-workflows.articulation-preparation-publication.v1"
] = "content-agent-workflows.articulation-preparation-publication.v1"
_PUBLISHER_IMPLEMENTATION: Literal[
    "content_agent_workflows.articulation.publish_embedded_articulation_preparation"
] = "content_agent_workflows.articulation.publish_embedded_articulation_preparation"
_MAX_ARTICULATION_INPUT_JSON_BYTES = 16 * 1024 * 1024
_MAX_ARTICULATION_GENERATED_JSON_BYTES = 64 * 1024 * 1024


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _canonical_relative_path(value: str) -> str:
    value = value.strip()
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or candidate.as_posix() != value
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("retained artifact paths must be canonical relative paths")
    return value


def _canonical_prim_path(value: str) -> str:
    value = value.strip()
    if (
        not value.startswith("/")
        or value == "/"
        or any(
            re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", part) is None
            for part in value[1:].split("/")
        )
    ):
        raise ValueError("readback prim paths must be canonical absolute prim paths")
    return value


class ArticulationPreparationRetainedArtifact(_FrozenModel):
    """Expected identity of one retained inspector input."""

    relative_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _canonical_relative_path(value)


class ArticulationPreparationMembershipAuthority(_FrozenModel):
    """One owner/disposition row from retained configuration authority."""

    member_prim: str
    authoritative_owner_prim: str
    disposition: Literal["independent_motion", "co_rigid", "explicit_fixed"]

    @field_validator("member_prim", "authoritative_owner_prim")
    @classmethod
    def validate_prim_path(cls, value: str) -> str:
        return _canonical_prim_path(value)

    @model_validator(mode="after")
    def validate_ownership_shape(self) -> Self:
        if self.disposition == "independent_motion":
            if self.member_prim != self.authoritative_owner_prim:
                raise ValueError(
                    "independent_motion ownership must be rooted at the member"
                )
        elif not (
            self.member_prim == self.authoritative_owner_prim
            or self.member_prim.startswith(f"{self.authoritative_owner_prim}/")
        ):
            raise ValueError("co-rigid and fixed owners must contain their members")
        return self


class ArticulationPreparationInspectorConfiguration(_FrozenModel):
    """Exact configuration bytes that determine preparation capabilities."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-configuration.v1"
    ] = ARTICULATION_PREPARATION_CONFIGURATION_SCHEMA_VERSION
    membership_policy: Literal["retained-explicit-membership-v1"]
    memberships: tuple[ArticulationPreparationMembershipAuthority, ...] = Field(
        min_length=1
    )
    capabilities: EmbeddedArticulationCapabilityLimits

    @model_validator(mode="after")
    def require_standalone_output_evidence(self) -> Self:
        if not self.capabilities.canonical_output_evidence_required:
            raise ValueError(
                "standalone Articulation preparation requires canonical output evidence"
            )
        return self


class ArticulationPreparationHierarchyPrim(_FrozenModel):
    """One exact prim row from the saved hierarchy readback."""

    prim_path: str
    parent_prim_path: str | None = None
    # An empty USD type token is the exact representation of a legal typeless
    # def/over/class prim; it must not be synthesized into a typed prim.
    type_name: str = Field(max_length=1_024)
    active: Literal[True] = True

    @field_validator("prim_path")
    @classmethod
    def validate_prim_path(cls, value: str) -> str:
        return _canonical_prim_path(value)

    @field_validator("parent_prim_path")
    @classmethod
    def validate_parent_prim_path(cls, value: str | None) -> str | None:
        return _canonical_prim_path(value) if value is not None else None


class ArticulationPreparationMembershipReadback(
    ArticulationPreparationMembershipAuthority
):
    """Saved observation that must equal retained configuration authority."""


class ArticulationPreparationInspectionReadback(_FrozenModel):
    """Complete saved readback consumed by the deterministic public producer."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-readback.v1"
    ] = ARTICULATION_PREPARATION_READBACK_SCHEMA_VERSION
    inspector_id: str = Field(min_length=1)
    inspector_implementation: str = Field(min_length=1)
    source: ArticulationPreparationRetainedArtifact
    dependencies: tuple[ArticulationPreparationRetainedArtifact, ...] = ()
    dependency_entry_count: int = Field(ge=0)
    dependency_closure_complete: Literal[True] = True
    source_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    configuration: ArticulationPreparationRetainedArtifact
    inspector_implementation_artifact: ArticulationPreparationRetainedArtifact
    saved_stage: ArticulationPreparationRetainedArtifact
    saved_stage_dependency_bundle_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hierarchy: tuple[ArticulationPreparationHierarchyPrim, ...] = Field(min_length=1)
    memberships: tuple[ArticulationPreparationMembershipReadback, ...] = Field(
        min_length=1
    )
    render_artifacts: tuple[ArticulationPreparationRetainedArtifact, ...] = Field(
        min_length=1
    )
    scene_artifacts: tuple[ArticulationPreparationRetainedArtifact, ...] = Field(
        min_length=1
    )
    proposal_status: OptionalProposalStatus = "not_requested"
    readback_complete: Literal[True] = True

    @model_validator(mode="after")
    def validate_complete_readback(self) -> Self:
        if self.proposal_status == "available":
            raise ValueError(
                "preparation inspection cannot fabricate an available proposal"
            )
        if self.dependency_entry_count != len(self.dependencies):
            raise ValueError("dependency closure entry count is incomplete")
        dependency_paths = tuple(item.relative_path for item in self.dependencies)
        if dependency_paths != tuple(sorted(dependency_paths)) or len(
            dependency_paths
        ) != len(set(dependency_paths)):
            raise ValueError("dependency closure must be unique and canonical")

        hierarchy_by_path = {item.prim_path: item for item in self.hierarchy}
        if len(hierarchy_by_path) != len(self.hierarchy):
            raise ValueError("hierarchy readback contains duplicate prim paths")
        for item in self.hierarchy:
            lexical_parent = item.prim_path.rsplit("/", 1)[0] or None
            if item.parent_prim_path != lexical_parent:
                raise ValueError(
                    "hierarchy readback parent must equal the lexical USD parent"
                )
            if item.parent_prim_path is not None and (
                item.parent_prim_path not in hierarchy_by_path
            ):
                raise ValueError("hierarchy readback contains an unknown parent")
            visited = {item.prim_path}
            parent = item.parent_prim_path
            while parent is not None:
                if parent in visited:
                    raise ValueError("hierarchy readback contains a parent cycle")
                if parent not in hierarchy_by_path:
                    raise ValueError("hierarchy readback contains an unknown ancestor")
                visited.add(parent)
                parent = hierarchy_by_path[parent].parent_prim_path

        membership_by_path = {item.member_prim: item for item in self.memberships}
        if len(membership_by_path) != len(self.memberships):
            raise ValueError("membership readback contains duplicate members")
        hierarchy_paths = set(hierarchy_by_path)
        if set(membership_by_path) != hierarchy_paths:
            raise ValueError(
                "membership readback must exactly cover the hierarchy prims"
            )
        if any(
            item.authoritative_owner_prim not in hierarchy_paths
            for item in self.memberships
        ):
            raise ValueError("membership readback names an unknown owner")

        artifact_paths = [
            self.source.relative_path,
            *(item.relative_path for item in self.dependencies),
            self.configuration.relative_path,
            self.inspector_implementation_artifact.relative_path,
            *(item.relative_path for item in self.render_artifacts),
            *(item.relative_path for item in self.scene_artifacts),
        ]
        # A saved stage may intentionally be the retained source itself.  No
        # other artifact role may alias another path.
        if self.saved_stage.relative_path != self.source.relative_path:
            artifact_paths.append(self.saved_stage.relative_path)
        elif self.saved_stage != self.source:
            raise ValueError(
                "an aliased saved stage must exactly equal the source claim"
            )
        if len(artifact_paths) != len(set(artifact_paths)):
            raise ValueError("retained artifact roles must not alias one path")
        return self

    @property
    def retained_artifacts(
        self,
    ) -> tuple[ArticulationPreparationRetainedArtifact, ...]:
        ordered = (
            self.source,
            *self.dependencies,
            self.configuration,
            self.inspector_implementation_artifact,
            self.saved_stage,
            *self.render_artifacts,
            *self.scene_artifacts,
        )
        return tuple({item.relative_path: item for item in ordered}.values())


class ArticulationPreparationInspectionReadbackDraft(
    ArticulationPreparationInspectionReadback
):
    """Coordinator-authored facts awaiting trusted dependency finalization.

    The outer coordinator can bind retained artifact bytes, hierarchy, and
    membership facts through its supported inspection tools, but it cannot
    reproduce the package-aware Joint Rigger dependency identity.  The
    deterministic publisher owns that derived fact for this versioned draft.
    """

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-readback-draft.v2"
    ] = ARTICULATION_PREPARATION_READBACK_DRAFT_SCHEMA_VERSION
    source_dependency_bundle_sha256: None = None
    saved_stage_dependency_bundle_sha256: None = None


type ArticulationPreparationReadback = (
    ArticulationPreparationInspectionReadback
    | ArticulationPreparationInspectionReadbackDraft
)
_ARTICULATION_PREPARATION_READBACK_ADAPTER = TypeAdapter(
    ArticulationPreparationReadback
)


class ArticulationPreparationPublication(_FrozenModel):
    """Create-only publication and complete retained-input closure."""

    schema_version: Literal[
        "content-agent-workflows.articulation-preparation-publication.v1"
    ] = ARTICULATION_PREPARATION_PUBLICATION_SCHEMA_VERSION
    retained_root: str = Field(min_length=1)
    readback: ExecutionArtifactBinding
    source: ExecutionArtifactBinding
    dependencies: tuple[ExecutionArtifactBinding, ...]
    configuration: ExecutionArtifactBinding
    inspector_implementation: ExecutionArtifactBinding
    saved_stage: ExecutionArtifactBinding
    renders: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    scene_artifacts: tuple[ExecutionArtifactBinding, ...] = Field(min_length=1)
    retained_closure_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    preparation: ExecutionArtifactBinding
    preparation_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    evidence_provider: ProducerIdentity
    publisher_implementation: Literal[
        "content_agent_workflows.articulation.publish_embedded_articulation_preparation"
    ] = _PUBLISHER_IMPLEMENTATION
    publisher_implementation_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    create_only: Literal[True] = True
    complete: Literal[True] = True


def _binding_from_read(read: ContainedArtifactRead) -> ExecutionArtifactBinding:
    return ExecutionArtifactBinding(
        path=str(read.path),
        sha256=read.sha256,
        size_bytes=read.size_bytes,
    )


def _capture_claim(
    root: Path,
    claim: ArticulationPreparationRetainedArtifact,
    *,
    label: str,
) -> ExecutionArtifactBinding:
    try:
        captured = read_contained_artifact(root, claim.relative_path)
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(f"{label} is unsafe: {exc}") from exc
    binding = _binding_from_read(captured)
    if binding.sha256 != claim.sha256 or binding.size_bytes != claim.size_bytes:
        raise EmbeddedArticulationError(f"{label} differs from saved readback")
    return binding


def _capture_readback(
    root: Path,
    readback_path: str | Path,
    *,
    expected: ExecutionArtifactBinding | None = None,
) -> tuple[ExecutionArtifactBinding, ArticulationPreparationReadback]:
    try:
        captured = read_contained_artifact(
            root,
            readback_path,
            max_bytes=_MAX_ARTICULATION_INPUT_JSON_BYTES,
            capture_bytes=True,
        )
        if captured.data is None:  # pragma: no cover - helper contract guard
            raise ValueError("readback capture returned no bytes")
        binding = _binding_from_read(captured)
        if expected is not None and binding != expected:
            raise EmbeddedArticulationError(
                "Articulation readback differs from the selected-leaf invocation"
            )
        readback = _ARTICULATION_PREPARATION_READBACK_ADAPTER.validate_json(
            captured.data
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Articulation preparation readback is unsafe or invalid: {exc}"
        ) from exc
    return binding, readback


def _capture_retained_inputs(
    root: Path,
    readback: ArticulationPreparationReadback,
) -> dict[str, ExecutionArtifactBinding]:
    return {
        item.relative_path: _capture_claim(
            root,
            item,
            label=f"retained Articulation artifact {item.relative_path}",
        )
        for item in readback.retained_artifacts
    }


def _load_configuration(
    root: Path,
    claim: ArticulationPreparationRetainedArtifact,
) -> ArticulationPreparationInspectorConfiguration:
    try:
        captured = read_contained_artifact(
            root,
            claim.relative_path,
            max_bytes=_MAX_ARTICULATION_INPUT_JSON_BYTES,
            capture_bytes=True,
        )
        if captured.data is None:  # pragma: no cover - helper contract guard
            raise ValueError("configuration capture returned no bytes")
        if captured.sha256 != claim.sha256 or captured.size_bytes != claim.size_bytes:
            raise ValueError("configuration identity differs from saved readback")
        return ArticulationPreparationInspectorConfiguration.model_validate_json(
            captured.data
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Articulation preparation configuration is unsafe or invalid: {exc}"
        ) from exc


def _verify_usd_identity(
    path: Path,
    *,
    expected_source_sha256: str,
    expected_dependency_sha256: str | None,
    expected_retained_paths: tuple[Path, ...],
    label: str,
    hierarchy_readback: ArticulationPreparationReadback | None = None,
) -> str:
    try:
        from world_understanding.functions.physics.joint_rigger.reference import (
            retain_usd_artifact_inspection,
        )

        observed_source_sha256 = file_sha256(path)
        with retain_usd_artifact_inspection(
            path,
            uri=path.as_uri(),
            expected_root_sha256=observed_source_sha256,
            recheck_source_content_on_exit=True,
        ) as inspection:
            identity = inspection.identity
            inventory = inspection.dependencies
            if any(item.local_path is None for item in inventory):
                raise EmbeddedArticulationError(
                    f"{label} dependency closure contains an unretained "
                    "resolver opinion"
                )
            observed_paths = {
                path.resolve(strict=True),
                *(
                    item.local_path.resolve(strict=True)
                    for item in inventory
                    if item.local_path
                ),
            }
            expected_paths = {
                item.resolve(strict=True) for item in expected_retained_paths
            }
            if hierarchy_readback is not None:
                _verify_saved_stage_hierarchy(
                    inspection.stage_path,
                    hierarchy_readback,
                )
            inspection.require_stage_unchanged()
    except Exception as exc:
        if isinstance(exc, EmbeddedArticulationError):
            raise
        raise EmbeddedArticulationError(
            f"Cannot establish complete {label} USD dependency closure: {exc}"
        ) from exc
    observed_dependency_sha256 = identity.dependency_bundle_sha256
    if observed_dependency_sha256 is None:  # pragma: no cover - identity contract
        raise EmbeddedArticulationError(
            f"{label} dependency closure omitted its canonical identity"
        )
    if identity.root_sha256 != expected_source_sha256 or (
        expected_dependency_sha256 is not None
        and observed_dependency_sha256 != expected_dependency_sha256
    ):
        raise EmbeddedArticulationError(
            f"{label} source or dependency closure differs from saved readback"
        )
    if observed_paths != expected_paths:
        raise EmbeddedArticulationError(
            f"{label} dependency closure is incomplete or contains extra artifacts"
        )
    return observed_dependency_sha256


def _verify_saved_stage_hierarchy(
    saved_stage_path: Path,
    readback: ArticulationPreparationReadback,
) -> None:
    """Cross-check every claimed row against one retained USD projection."""

    expected = tuple(
        sorted(
            (item.prim_path, item.parent_prim_path, item.type_name, item.active)
            for item in readback.hierarchy
        )
    )
    try:
        from pxr import Usd

        stage = Usd.Stage.Open(str(saved_stage_path), load=Usd.Stage.LoadAll)
        if stage is None:
            raise ValueError("Usd.Stage.Open returned None")
        observed_rows: list[tuple[str, str | None, str, bool]] = []
        for prim in stage.TraverseAll():
            if len(observed_rows) >= len(expected):
                raise ValueError(
                    "saved stage contains more prims than the bounded readback"
                )
            if prim.IsInstance() or prim.IsInstanceable():
                raise ValueError(
                    "saved stage contains an instance or instanceable prim that "
                    f"the v1 hierarchy contract cannot expand: {prim.GetPath()}"
                )
            observed_rows.append(
                (
                    str(prim.GetPath()),
                    (
                        None
                        if str(prim.GetParent().GetPath()) == "/"
                        else str(prim.GetParent().GetPath())
                    ),
                    str(prim.GetTypeName()),
                    prim.IsActive(),
                )
            )
        observed = tuple(sorted(observed_rows))
    except Exception as exc:
        raise EmbeddedArticulationError(
            f"Cannot verify saved-stage hierarchy readback: {exc}"
        ) from exc
    if observed != expected:
        observed_by_path = {item[0]: item[1:] for item in observed}
        expected_by_path = {item[0]: item[1:] for item in expected}
        mismatched_path = next(
            item
            for item in sorted(set(observed_by_path) | set(expected_by_path))
            if observed_by_path.get(item) != expected_by_path.get(item)
        )
        raise EmbeddedArticulationError(
            "saved-stage hierarchy differs from the complete typed readback at "
            f"{mismatched_path}: observed={observed_by_path.get(mismatched_path)!r}, "
            f"expected={expected_by_path.get(mismatched_path)!r}"
        )


def _unique_bindings(
    bindings: tuple[ExecutionArtifactBinding, ...],
) -> tuple[ExecutionArtifactBinding, ...]:
    return tuple({item.path: item for item in bindings}.values())


def _authoritative_memberships(
    readback: ArticulationPreparationReadback,
    configuration: ArticulationPreparationInspectorConfiguration,
) -> tuple[ArticulationPreparationMembershipReadback, ...]:
    """Derive exact rows from retained configuration and verify saved observation."""

    configured = tuple(
        sorted(
            (
                ArticulationPreparationMembershipReadback.model_validate(
                    item.model_dump(mode="json")
                )
                for item in configuration.memberships
            ),
            key=lambda item: item.member_prim,
        )
    )
    configured_by_member = {item.member_prim: item for item in configured}
    hierarchy_paths = {item.prim_path for item in readback.hierarchy}
    if len(configured_by_member) != len(configured):
        raise EmbeddedArticulationError(
            "retained membership configuration contains duplicate members"
        )
    if set(configured_by_member) != hierarchy_paths:
        raise EmbeddedArticulationError(
            "retained membership configuration must exactly cover saved hierarchy"
        )
    if any(item.authoritative_owner_prim not in hierarchy_paths for item in configured):
        raise EmbeddedArticulationError(
            "retained membership configuration names an unknown owner"
        )
    if any(
        configured_by_member[item.authoritative_owner_prim].authoritative_owner_prim
        != item.authoritative_owner_prim
        for item in configured
    ):
        raise EmbeddedArticulationError(
            "retained membership configuration names a transitively owned "
            "authoritative owner"
        )
    observed = tuple(sorted(readback.memberships, key=lambda item: item.member_prim))
    if observed != configured:
        raise EmbeddedArticulationError(
            "saved membership readback differs from retained configuration authority"
        )
    return configured


def _derive_preparation(
    readback_binding: ExecutionArtifactBinding,
    readback: ArticulationPreparationReadback,
    captured: dict[str, ExecutionArtifactBinding],
    configuration_contract: ArticulationPreparationInspectorConfiguration,
    *,
    source_dependency_bundle_sha256: str,
    saved_stage_dependency_bundle_sha256: str,
) -> EmbeddedArticulationPreparation:
    source = captured[readback.source.relative_path]
    dependencies = tuple(captured[item.relative_path] for item in readback.dependencies)
    configuration = captured[readback.configuration.relative_path]
    implementation = captured[readback.inspector_implementation_artifact.relative_path]
    saved_stage = captured[readback.saved_stage.relative_path]
    renders = tuple(captured[item.relative_path] for item in readback.render_artifacts)
    scene_artifacts = tuple(
        captured[item.relative_path] for item in readback.scene_artifacts
    )
    hierarchy = tuple(sorted(readback.hierarchy, key=lambda item: item.prim_path))
    memberships = _authoritative_memberships(readback, configuration_contract)
    source_members = tuple(item.member_prim for item in memberships)
    authoritative_owners = tuple(
        dict.fromkeys(item.authoritative_owner_prim for item in memberships)
    )
    common_inspection = _unique_bindings(
        (
            source,
            *dependencies,
            configuration,
            implementation,
            saved_stage,
            readback_binding,
        )
    )
    producer = ProducerIdentity(
        producer_id=readback.inspector_id,
        role="evidence_provider",
        implementation=readback.inspector_implementation,
        implementation_digest=implementation.sha256,
    )
    return EmbeddedArticulationPreparation(
        source_sha256=source.sha256,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        configuration_sha256=configuration.sha256,
        evidence_provider=producer,
        source_hierarchy=ProviderNeutralEvidenceRecord(
            evidence_id="source-hierarchy-inspection",
            evidence_type="inspection",
            status="available",
            summary="Complete deterministic saved-stage hierarchy readback.",
            artifacts=common_inspection,
            facts={
                "prim_count": len(hierarchy),
                "prims": [item.model_dump(mode="json") for item in hierarchy],
                "readback_sha256": readback_binding.sha256,
            },
        ),
        source_members=ProviderNeutralEvidenceRecord(
            evidence_id="joint-source-member-inspection",
            evidence_type="inspection",
            status="available",
            summary="Source members derived from complete per-prim ownership rows.",
            artifacts=(saved_stage, readback_binding, implementation, configuration),
            facts={"source_member_prims": list(source_members)},
        ),
        authoritative_owners=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoritative-owner-inspection",
            evidence_type="inspection",
            status="available",
            summary="Authoritative owners derived from retained configuration authority.",
            artifacts=(saved_stage, readback_binding, implementation, configuration),
            facts={
                "authoritative_owner_prims": list(authoritative_owners),
                "membership_rows": [
                    item.model_dump(mode="json") for item in memberships
                ],
                "membership_policy": configuration_contract.membership_policy,
            },
        ),
        capabilities=ProviderNeutralEvidenceRecord(
            evidence_id="joint-authoring-capabilities",
            evidence_type="capability",
            status="available",
            summary="Exact configured provider-neutral authoring capabilities.",
            artifacts=(configuration, implementation, readback_binding),
            facts=configuration_contract.capabilities.model_dump(mode="json"),
        ),
        renders=ProviderNeutralEvidenceRecord(
            evidence_id="joint-render-inspection",
            evidence_type="render",
            status="available",
            summary="Exact retained diagnostic render inspection artifacts.",
            artifacts=(*renders, readback_binding, implementation, configuration),
            facts={
                "artifact_count": len(renders),
                "artifact_sha256s": [item.sha256 for item in renders],
            },
        ),
        scene=ProviderNeutralEvidenceRecord(
            evidence_id="joint-scene-inspection",
            evidence_type="inspection",
            status="available",
            summary="Exact saved stage, scene records, and dependency closure.",
            artifacts=_unique_bindings(
                (
                    source,
                    *dependencies,
                    saved_stage,
                    *scene_artifacts,
                    readback_binding,
                    implementation,
                    configuration,
                )
            ),
            facts={
                "source_sha256": source.sha256,
                "source_dependency_bundle_sha256": (source_dependency_bundle_sha256),
                "dependency_entry_count": len(dependencies),
                "saved_stage_sha256": saved_stage.sha256,
                "saved_stage_dependency_bundle_sha256": (
                    saved_stage_dependency_bundle_sha256
                ),
                "scene_artifact_sha256s": [item.sha256 for item in scene_artifacts],
            },
        ),
        proposal_status=readback.proposal_status,
    )


def _fresh_publication_root(path: str | Path) -> Path:
    absolute = Path(os.path.abspath(Path(path).expanduser()))
    try:
        resolved = absolute.resolve(strict=False)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"preparation publication root is unsafe: {absolute}"
        ) from exc
    if resolved != absolute:
        raise EmbeddedArticulationError(
            f"preparation publication root traverses a symlink: {absolute}"
        )
    try:
        prepare_writable_directory(absolute.parent)
        absolute.mkdir(mode=0o700, exist_ok=False)
    except FileExistsError as exc:
        raise EmbeddedArticulationError(
            "preparation publication output already exists; publication is create-only"
        ) from exc
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"preparation publication root is unsafe: {absolute}"
        ) from exc
    try:
        metadata = absolute.lstat()
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or absolute.resolve(strict=True) != absolute
        ):
            raise ValueError("created publication root is not the requested directory")
    except (OSError, ValueError) as exc:  # pragma: no cover - OS race guard
        raise EmbeddedArticulationError(
            "preparation publication root is unsafe after creation"
        ) from exc
    return absolute


def _json_document(payload: BaseModel) -> bytes:
    document = (
        json.dumps(
            payload.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        ).encode("utf-8")
        + b"\n"
    )
    if len(document) > _MAX_ARTICULATION_GENERATED_JSON_BYTES:
        raise EmbeddedArticulationError(
            "generated Articulation publication JSON exceeds the bounded limit"
        )
    return document


def _write_json_create_only(root: Path, name: str, payload: BaseModel) -> Path:
    document = _json_document(payload)
    if os.name == "nt":
        with open_confined_directory(root) as root_descriptor:
            published = write_bytes_to_confined(
                root_descriptor,
                name,
                document,
                overwrite=False,
                file_mode=0o444,
            )
        if not published:
            raise FileExistsError(root / name)
        (root / name).chmod(0o444)
        return root / name

    root_fd = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    file_fd = -1
    try:
        file_fd = os.open(
            name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
            dir_fd=root_fd,
        )
        os.fchmod(file_fd, 0o444)
        view = memoryview(document)
        while view:
            written = os.write(file_fd, view)
            if written <= 0:  # pragma: no cover - defensive OS contract guard
                raise OSError("create-only publication write made no progress")
            view = view[written:]
        os.fsync(file_fd)
        os.fsync(root_fd)
    except BaseException:
        if file_fd >= 0:
            os.close(file_fd)
            file_fd = -1
            try:
                os.unlink(name, dir_fd=root_fd)
            except FileNotFoundError:
                pass
        raise
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        os.close(root_fd)
    return root / name


def _published_binding(root: Path, name: str) -> ExecutionArtifactBinding:
    try:
        return _binding_from_read(read_contained_artifact(root, name))
    except (OSError, ValueError) as exc:  # pragma: no cover - write invariant
        raise EmbeddedArticulationError(
            f"published Articulation artifact is unsafe: {name}: {exc}"
        ) from exc


def _seal_publication_root(root: Path, names: set[str]) -> None:
    if os.name == "nt":
        metadata = root.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or getattr(
            metadata, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise EmbeddedArticulationError(
                f"cannot seal preparation publication root: {root}"
            )
        root.chmod(0o555)
        _validate_sealed_publication_root(root, names)
        return

    root_fd = -1
    try:
        root_fd = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        opened = os.fstat(root_fd)
        path_metadata = root.lstat()
        if (
            not stat.S_ISDIR(opened.st_mode)
            or opened.st_dev != path_metadata.st_dev
            or opened.st_ino != path_metadata.st_ino
        ):
            raise OSError("publication root identity changed before sealing")
        os.fchmod(root_fd, 0o500)
        os.fsync(root_fd)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"cannot seal preparation publication root: {root}"
        ) from exc
    finally:
        if root_fd >= 0:
            os.close(root_fd)
    _validate_sealed_publication_root(root, names)


def _validate_sealed_publication_root(root: Path, names: set[str]) -> None:
    try:
        root_metadata = root.lstat()
        if (
            not stat.S_ISDIR(root_metadata.st_mode)
            or (os.name == "posix" and stat.S_IMODE(root_metadata.st_mode) != 0o500)
            or getattr(root_metadata, "st_file_attributes", 0)
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        ):
            raise ValueError("publication root is not sealed read-only")
        root_identity = (root_metadata.st_dev, root_metadata.st_ino)
        with open_confined_directory(root) as root_descriptor:
            if _publication_entry_names(root) != names:
                raise ValueError("publication contains mutable or extra entries")
            for name in sorted(names):
                with open_confined_regular_file(
                    root_descriptor,
                    name,
                ) as (stream, metadata):
                    current = (root / name).lstat()
                    after = os.fstat(stream.fileno())
                    identity = (metadata.st_dev, metadata.st_ino)
                    if (
                        not stat.S_ISREG(metadata.st_mode)
                        or not stat.S_ISREG(current.st_mode)
                        or metadata.st_nlink != 1
                        or current.st_nlink != 1
                        or (current.st_dev, current.st_ino) != identity
                        or (after.st_dev, after.st_ino) != identity
                        or (
                            os.name == "posix"
                            and stat.S_IMODE(metadata.st_mode) != 0o444
                        )
                        or (os.name == "nt" and stat.S_IMODE(metadata.st_mode) & 0o222)
                    ):
                        raise ValueError(
                            f"publication artifact is mutable or unsafe: {name}"
                        )
            final_root_metadata = root.lstat()
            if (
                final_root_metadata.st_dev,
                final_root_metadata.st_ino,
            ) != root_identity or _publication_entry_names(root) != names:
                raise ValueError("publication contains mutable or extra entries")
    except (ArtifactPathError, OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Articulation preparation publication is not sealed: {exc}"
        ) from exc


def _closure_digest(
    readback_binding: ExecutionArtifactBinding,
    readback: ArticulationPreparationReadback,
    captured: dict[str, ExecutionArtifactBinding],
    *,
    source_dependency_bundle_sha256: str,
    saved_stage_dependency_bundle_sha256: str,
) -> str:
    return canonical_json_digest(
        {
            "schema_version": (
                "content-agent-workflows.articulation-retained-closure.v1"
            ),
            "readback": readback_binding.model_dump(mode="json"),
            "artifacts": [
                captured[item.relative_path].model_dump(mode="json")
                for item in readback.retained_artifacts
            ],
            "source_dependency_bundle_sha256": (source_dependency_bundle_sha256),
            "saved_stage_dependency_bundle_sha256": (
                saved_stage_dependency_bundle_sha256
            ),
        }
    )


def publish_embedded_articulation_preparation(
    readback_path: str | Path,
    *,
    retained_root: str | Path,
    output_dir: str | Path,
    expected_readback: ExecutionArtifactBinding | None = None,
) -> ArticulationPreparationPublication:
    """Derive and create-only publish one exact provider-neutral preparation."""

    retained_absolute = Path(os.path.abspath(Path(retained_root).expanduser()))
    try:
        root = retained_absolute.resolve(strict=True)
    except OSError as exc:
        raise EmbeddedArticulationError(
            f"Articulation retained root does not exist: {retained_root}"
        ) from exc
    if root != retained_absolute or not root.is_dir():
        raise EmbeddedArticulationError(
            f"Articulation retained root is not a symlink-free directory: {root}"
        )
    readback_binding, readback = _capture_readback(
        root,
        readback_path,
        expected=expected_readback,
    )
    if str(readback_binding.path) in {
        str(root / item.relative_path) for item in readback.retained_artifacts
    }:
        raise EmbeddedArticulationError(
            "Articulation readback must not alias a retained artifact role"
        )
    captured = _capture_retained_inputs(root, readback)
    configuration_contract = _load_configuration(root, readback.configuration)
    source = captured[readback.source.relative_path]
    saved_stage = captured[readback.saved_stage.relative_path]
    source_dependency_bundle_sha256 = _verify_usd_identity(
        Path(source.path),
        expected_source_sha256=source.sha256,
        expected_dependency_sha256=(
            None
            if isinstance(readback, ArticulationPreparationInspectionReadbackDraft)
            else readback.source_dependency_bundle_sha256
        ),
        expected_retained_paths=(
            Path(source.path),
            *(
                Path(item.path)
                for item in (
                    captured[claim.relative_path] for claim in readback.dependencies
                )
            ),
        ),
        label="source",
        hierarchy_readback=readback if saved_stage == source else None,
    )
    if saved_stage != source:
        saved_stage_dependency_bundle_sha256 = _verify_usd_identity(
            Path(saved_stage.path),
            expected_source_sha256=saved_stage.sha256,
            expected_dependency_sha256=(
                None
                if isinstance(readback, ArticulationPreparationInspectionReadbackDraft)
                else readback.saved_stage_dependency_bundle_sha256
            ),
            expected_retained_paths=(
                Path(saved_stage.path),
                *(
                    Path(item.path)
                    for item in (
                        captured[claim.relative_path] for claim in readback.dependencies
                    )
                ),
            ),
            label="saved stage",
            hierarchy_readback=readback,
        )
    else:
        if not isinstance(
            readback, ArticulationPreparationInspectionReadbackDraft
        ) and (
            readback.saved_stage_dependency_bundle_sha256
            != readback.source_dependency_bundle_sha256
        ):
            raise EmbeddedArticulationError(
                "aliased source and saved stage require one exact dependency identity"
            )
        saved_stage_dependency_bundle_sha256 = source_dependency_bundle_sha256
    preparation = _derive_preparation(
        readback_binding,
        readback,
        captured,
        configuration_contract,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        saved_stage_dependency_bundle_sha256=(saved_stage_dependency_bundle_sha256),
    )
    # Size-check derived bytes before creating the immutable destination.
    _json_document(preparation)

    # Re-open every retained input after identity and derivation.  No path is
    # reopened for publication if any byte or inode-visible metadata drifted.
    observed_readback_binding, observed_readback = _capture_readback(
        root,
        readback_path,
        expected=expected_readback,
    )
    observed_captured = _capture_retained_inputs(root, observed_readback)
    observed_configuration = _load_configuration(
        root,
        observed_readback.configuration,
    )
    if (
        observed_readback_binding != readback_binding
        or (
            expected_readback is not None
            and observed_readback_binding != expected_readback
        )
        or observed_readback != readback
        or observed_captured != captured
        or observed_configuration != configuration_contract
    ):
        raise EmbeddedArticulationError(
            "Articulation retained inputs changed during preparation"
        )

    publication_root = _fresh_publication_root(output_dir)
    preparation_path = _write_json_create_only(
        publication_root,
        "embedded_articulation_preparation.json",
        preparation,
    )
    preparation_binding = _published_binding(publication_root, preparation_path.name)
    publication = ArticulationPreparationPublication(
        retained_root=str(root),
        readback=readback_binding,
        source=source,
        dependencies=tuple(
            captured[item.relative_path] for item in readback.dependencies
        ),
        configuration=captured[readback.configuration.relative_path],
        inspector_implementation=captured[
            readback.inspector_implementation_artifact.relative_path
        ],
        saved_stage=saved_stage,
        renders=tuple(
            captured[item.relative_path] for item in readback.render_artifacts
        ),
        scene_artifacts=tuple(
            captured[item.relative_path] for item in readback.scene_artifacts
        ),
        retained_closure_digest=_closure_digest(
            readback_binding,
            readback,
            captured,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            saved_stage_dependency_bundle_sha256=(saved_stage_dependency_bundle_sha256),
        ),
        preparation=preparation_binding,
        preparation_digest=canonical_json_digest(preparation),
        evidence_provider=preparation.evidence_provider,
        publisher_implementation_sha256=file_sha256(Path(__file__)),
    )
    publication_path = _write_json_create_only(
        publication_root,
        "articulation_preparation_publication.json",
        publication,
    )
    publication_binding = _published_binding(publication_root, publication_path.name)
    try:
        publication_read = read_contained_artifact(
            publication_root,
            publication_binding.path,
            max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            capture_bytes=True,
        )
        if publication_read.data is None:  # pragma: no cover - helper contract guard
            raise ValueError("publication capture returned no bytes")
        if (
            _binding_from_read(publication_read) != publication_binding
            or ArticulationPreparationPublication.model_validate_json(
                publication_read.data
            )
            != publication
        ):
            raise ValueError("publication exact readback differs")
    except (OSError, ValueError) as exc:  # pragma: no cover - write invariant
        raise EmbeddedArticulationError(
            f"Articulation preparation publication exact readback failed: {exc}"
        ) from exc
    _seal_publication_root(
        publication_root,
        {
            "embedded_articulation_preparation.json",
            "articulation_preparation_publication.json",
        },
    )
    if (
        validate_embedded_articulation_preparation_publication(publication_path)
        != publication
    ):
        raise EmbeddedArticulationError(
            "Articulation preparation publication self-validation differed"
        )
    return publication


def _publication_entry_names(root: Path) -> set[str]:
    try:
        return {item.name for item in root.iterdir()}
    except OSError as exc:
        raise EmbeddedArticulationError(
            "Articulation preparation publication entries are unavailable"
        ) from exc


def validate_embedded_articulation_preparation_publication(
    publication_path: str | Path,
) -> ArticulationPreparationPublication:
    """Revalidate a complete create-only preparation publication."""

    try:
        absolute_path = Path(os.path.abspath(Path(publication_path).expanduser()))
        path = absolute_path.resolve(strict=True)
        if path != absolute_path:
            raise ValueError("publication path traverses a symlink")
        publication_root = path.parent
        publication_read = read_contained_artifact(
            publication_root,
            path.name,
            max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            capture_bytes=True,
        )
        if publication_read.data is None:  # pragma: no cover - helper contract guard
            raise ValueError("publication capture returned no bytes")
        publication = ArticulationPreparationPublication.model_validate_json(
            publication_read.data
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Articulation preparation publication is unsafe or invalid: {exc}"
        ) from exc
    entries = _publication_entry_names(publication_root)
    expected_entries = {
        "embedded_articulation_preparation.json",
        "articulation_preparation_publication.json",
    }
    if entries != expected_entries:
        raise EmbeddedArticulationError(
            "Articulation preparation publication contains mutable or extra entries"
        )
    _validate_sealed_publication_root(publication_root, expected_entries)
    retained_absolute = Path(
        os.path.abspath(Path(publication.retained_root).expanduser())
    )
    try:
        root = retained_absolute.resolve(strict=True)
    except OSError as exc:
        raise EmbeddedArticulationError(
            "Articulation publication retained root does not exist"
        ) from exc
    if root != retained_absolute or not root.is_dir():
        raise EmbeddedArticulationError(
            "Articulation publication retained root is not a symlink-free directory"
        )
    readback_binding, readback = _capture_readback(root, publication.readback.path)
    if str(readback_binding.path) in {
        str(root / item.relative_path) for item in readback.retained_artifacts
    }:
        raise EmbeddedArticulationError(
            "Articulation readback must not alias a retained artifact role"
        )
    captured = _capture_retained_inputs(root, readback)
    configuration_contract = _load_configuration(root, readback.configuration)
    source = captured[readback.source.relative_path]
    saved_stage = captured[readback.saved_stage.relative_path]
    source_dependency_bundle_sha256 = _verify_usd_identity(
        Path(source.path),
        expected_source_sha256=source.sha256,
        expected_dependency_sha256=(
            None
            if isinstance(readback, ArticulationPreparationInspectionReadbackDraft)
            else readback.source_dependency_bundle_sha256
        ),
        expected_retained_paths=(
            Path(source.path),
            *(
                Path(item.path)
                for item in (
                    captured[claim.relative_path] for claim in readback.dependencies
                )
            ),
        ),
        label="source",
        hierarchy_readback=readback if saved_stage == source else None,
    )
    if saved_stage != source:
        saved_stage_dependency_bundle_sha256 = _verify_usd_identity(
            Path(saved_stage.path),
            expected_source_sha256=saved_stage.sha256,
            expected_dependency_sha256=(
                None
                if isinstance(readback, ArticulationPreparationInspectionReadbackDraft)
                else readback.saved_stage_dependency_bundle_sha256
            ),
            expected_retained_paths=(
                Path(saved_stage.path),
                *(
                    Path(item.path)
                    for item in (
                        captured[claim.relative_path] for claim in readback.dependencies
                    )
                ),
            ),
            label="saved stage",
            hierarchy_readback=readback,
        )
    else:
        if not isinstance(
            readback, ArticulationPreparationInspectionReadbackDraft
        ) and (
            readback.saved_stage_dependency_bundle_sha256
            != readback.source_dependency_bundle_sha256
        ):
            raise EmbeddedArticulationError(
                "aliased source and saved stage require one exact dependency identity"
            )
        saved_stage_dependency_bundle_sha256 = source_dependency_bundle_sha256
    preparation = _derive_preparation(
        readback_binding,
        readback,
        captured,
        configuration_contract,
        source_dependency_bundle_sha256=source_dependency_bundle_sha256,
        saved_stage_dependency_bundle_sha256=(saved_stage_dependency_bundle_sha256),
    )
    try:
        preparation_read = read_contained_artifact(
            publication_root,
            publication.preparation.path,
            max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            capture_bytes=True,
        )
        if preparation_read.data is None:  # pragma: no cover - helper contract guard
            raise ValueError("preparation capture returned no bytes")
        persisted_preparation = EmbeddedArticulationPreparation.model_validate_json(
            preparation_read.data
        )
    except (OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Published Articulation preparation is unsafe or invalid: {exc}"
        ) from exc
    if (
        _binding_from_read(preparation_read) != publication.preparation
        or persisted_preparation != preparation
        or canonical_json_digest(preparation) != publication.preparation_digest
        or readback_binding != publication.readback
        or source != publication.source
        or saved_stage != publication.saved_stage
        or captured[readback.configuration.relative_path] != publication.configuration
        or captured[readback.inspector_implementation_artifact.relative_path]
        != publication.inspector_implementation
        or tuple(captured[item.relative_path] for item in readback.dependencies)
        != publication.dependencies
        or tuple(captured[item.relative_path] for item in readback.render_artifacts)
        != publication.renders
        or tuple(captured[item.relative_path] for item in readback.scene_artifacts)
        != publication.scene_artifacts
        or _closure_digest(
            readback_binding,
            readback,
            captured,
            source_dependency_bundle_sha256=source_dependency_bundle_sha256,
            saved_stage_dependency_bundle_sha256=(saved_stage_dependency_bundle_sha256),
        )
        != publication.retained_closure_digest
        or preparation.evidence_provider != publication.evidence_provider
    ):
        raise EmbeddedArticulationError(
            "Articulation preparation publication or retained closure changed"
        )
    try:
        final_readback_binding, final_readback = _capture_readback(
            root, publication.readback.path
        )
        final_captured = _capture_retained_inputs(root, final_readback)
        final_configuration = _load_configuration(
            root,
            final_readback.configuration,
        )
        final_publication_read = read_contained_artifact(
            publication_root,
            path.name,
            max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            capture_bytes=True,
        )
        final_preparation_read = read_contained_artifact(
            publication_root,
            publication.preparation.path,
            max_bytes=_MAX_ARTICULATION_GENERATED_JSON_BYTES,
            capture_bytes=True,
        )
    except (EmbeddedArticulationError, OSError, ValueError) as exc:
        raise EmbeddedArticulationError(
            f"Articulation retained bytes changed during validation: {exc}"
        ) from exc
    if (
        final_readback_binding != readback_binding
        or final_readback != readback
        or final_captured != captured
        or final_configuration != configuration_contract
        or _binding_from_read(final_publication_read)
        != _binding_from_read(publication_read)
        or _binding_from_read(final_preparation_read)
        != _binding_from_read(preparation_read)
        or _publication_entry_names(publication_root) != expected_entries
    ):
        raise EmbeddedArticulationError(
            "Articulation retained or published bytes changed during validation"
        )
    _validate_sealed_publication_root(publication_root, expected_entries)
    return publication


__all__ = [
    "ARTICULATION_PREPARATION_CONFIGURATION_SCHEMA_VERSION",
    "ARTICULATION_PREPARATION_PUBLICATION_SCHEMA_VERSION",
    "ARTICULATION_PREPARATION_READBACK_DRAFT_SCHEMA_VERSION",
    "ARTICULATION_PREPARATION_READBACK_SCHEMA_VERSION",
    "ArticulationPreparationHierarchyPrim",
    "ArticulationPreparationInspectionReadback",
    "ArticulationPreparationInspectionReadbackDraft",
    "ArticulationPreparationInspectorConfiguration",
    "ArticulationPreparationMembershipAuthority",
    "ArticulationPreparationMembershipReadback",
    "ArticulationPreparationPublication",
    "ArticulationPreparationRetainedArtifact",
    "publish_embedded_articulation_preparation",
    "validate_embedded_articulation_preparation_publication",
]
