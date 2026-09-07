# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed release-domain leaves for complete composed SimReady outcomes.

The repository catalog is imported in environments that do not install the
launcher package, so this module owns only public schemas and deterministic
projectors.  The checked-in ``content-workflow-composed-leaf`` launcher owns
execution and imports optional domain runners lazily.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_directory_at,
    open_confined_regular_file,
    open_regular_file_no_follow,
    write_bytes_to_confined,
)
from world_understanding.utils.usd.package import validate_usdz_package_layout

from content_agent_workflows.articulation.asset_leaf_adapter import (
    ArticulationFocusedLeafResult,
)
from content_agent_workflows.common.validation_evidence import (
    VALIDATION_EVIDENCE_SCHEMA_VERSION,
    ValidationEvidence,
)
from content_agent_workflows.physics import (
    PhysicsApplyWorkflowInput,
    PhysicsApplyWorkflowResult,
    parse_physics_component_catalog_entry,
)
from content_agent_workflows.simready import (
    SimReadyConformanceInput,
    SimReadyConformanceReport,
    SimReadyValidationInput,
    SimReadyValidationReport,
)
from content_agent_workflows.texture.asset_leaf_adapter import TextureFocusedLeafResult
from content_agent_workflows.validation import (
    CanonicalVisualEvidencePayload,
)

from .catalog import AssetLeafRuntimeBinding, AssetLeafRuntimeBundle
from .catalog_adapters import (
    CanonicalOvrtxEvidenceLeafInvocation,
    canonical_ovrtx_asset_leaf_runtime_binding,
)
from .models import (
    ASSET_LEAF_RECEIPT_SCHEMA_VERSION,
    ArtifactBinding,
    AssetCompositionRun,
    AssetExecutionGraph,
    AssetLeafProjectionContext,
    AssetLeafProjectionPayload,
    AssetLeafReceipt,
    LeafReceiptArtifactCategory,
)

MATERIAL_ASSIGNMENT_LEAF_ID = "material.assignment.v1"
PHYSICS_INSPECTION_LEAF_ID = "physics.component-inspection.v1"
PHYSICS_APPLY_LEAF_ID = "physics.apply-runtime.v1"
SIMREADY_CONFORMANCE_LEAF_ID = "simready.conformance.v1"
SIMREADY_VALIDATION_LEAF_ID = "simready.validation.v1"
PORTABLE_PACKAGE_LEAF_ID = "asset.portable-package.v1"
COMBINED_RESULT_LEAF_ID = "asset.combined-result.v1"
FINAL_OVRTX_EVIDENCE_LEAF_ID = "asset.final-ovrtx-evidence.v1"
ARTICULATION_PUBLISH_LEAF_ID = "articulation.publish.v1"
TEXTURE_PUBLISH_LEAF_ID = "texture.publish.v1"

RELEASE_COMPOSED_LEAF_IDS = (
    COMBINED_RESULT_LEAF_ID,
    FINAL_OVRTX_EVIDENCE_LEAF_ID,
    MATERIAL_ASSIGNMENT_LEAF_ID,
    PHYSICS_APPLY_LEAF_ID,
    PHYSICS_INSPECTION_LEAF_ID,
    PORTABLE_PACKAGE_LEAF_ID,
    SIMREADY_CONFORMANCE_LEAF_ID,
    SIMREADY_VALIDATION_LEAF_ID,
)

_ENTRYPOINT = "content-workflow-composed-leaf {operation} --invocation"
_TERMINAL_STATUSES = frozenset({"passed", "failed"})


class _FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ComposedLeafInvocation(_FrozenModel):
    """Common confinement identity for one active graph-leaf attempt."""

    attempt_root: str = Field(min_length=1)

    @field_validator("attempt_root")
    @classmethod
    def validate_attempt_root(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("composed leaf attempt_root must be absolute")
        return value


class MaterialAssignmentLeafInvocation(ComposedLeafInvocation):
    """Frozen three-phase Material input owned by one active leaf attempt."""

    source: ArtifactBinding
    repository_root: str = Field(min_length=1)
    materials_yaml: ArtifactBinding
    materials_usd: ArtifactBinding
    materials_usd_dependencies: tuple[ArtifactBinding, ...] = ()
    reference_images: tuple[ArtifactBinding, ...] = ()
    reference_files: tuple[ArtifactBinding, ...] = ()
    output_asset_path: str = Field(min_length=1)
    decision_patch_path: str = Field(min_length=1)
    review_patch_path: str = Field(min_length=1)
    scene_tool_timeout_seconds: float = Field(default=300.0, gt=0.0)
    optimize: bool = False
    root_prim_path: str | None = None
    material_candidate_space: Literal["source", "inspection"] = "source"
    skip_instances: bool = True
    skip_prototypes: bool = False
    skip_invisible: bool = False
    flatten_prototypes: bool | None = None
    enable_deinstance: bool | None = None
    enable_split: bool | None = None
    enable_deduplicate: bool | None = False
    respect_existing_material_bindings: bool = False

    @model_validator(mode="after")
    def validate_material_paths(self) -> MaterialAssignmentLeafInvocation:
        root = Path(self.attempt_root)
        if not Path(self.repository_root).is_absolute():
            raise ValueError("Material repository_root must be absolute")
        for label, value in (
            ("output_asset_path", self.output_asset_path),
            ("decision_patch_path", self.decision_patch_path),
            ("review_patch_path", self.review_patch_path),
        ):
            _require_lexical_descendant(root, Path(value), label=label)
        return self


class PhysicsInspectionLeafInvocation(ComposedLeafInvocation):
    """Exact source identity for provider-free Physics component inspection."""

    source: ArtifactBinding
    component_catalog_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_component_catalog_path(self) -> PhysicsInspectionLeafInvocation:
        _require_lexical_descendant(
            Path(self.attempt_root),
            Path(self.component_catalog_path),
            label="Physics component_catalog_path",
        )
        return self


class PhysicsApplyLeafInvocation(ComposedLeafInvocation):
    """Outer-authored Physics decision plus the exact inspected source."""

    source: ArtifactBinding
    component_catalog: ArtifactBinding
    decision_patch: ArtifactBinding
    params: PhysicsApplyWorkflowInput

    @model_validator(mode="after")
    def validate_physics_paths(self) -> PhysicsApplyLeafInvocation:
        root = Path(self.attempt_root)
        if Path(self.params.usd_path) != Path(self.source.path):
            raise ValueError("Physics workflow input differs from source binding")
        if self.params.inspection_asset_sha256 != self.source.sha256:
            raise ValueError("Physics workflow input must bind the inspected digest")
        if not self.params.resume:
            raise ValueError("Physics composed leaf requires replay-safe resume")
        if self.params.decision_patch_path is None or Path(
            self.params.decision_patch_path
        ) != Path(self.decision_patch.path):
            raise ValueError("Physics workflow input must bind the outer decision")
        _require_lexical_descendant(
            root,
            Path(self.decision_patch.path),
            label="outer Physics decision patch",
        )
        _require_lexical_descendant(
            root, Path(self.params.output_dir), label="output_dir"
        )
        if self.params.output_usd_path is None:
            raise ValueError("Physics composed leaf requires output_usd_path")
        _require_lexical_descendant(
            root,
            Path(self.params.output_usd_path),
            label="output_usd_path",
        )
        return self


class SimReadyConformanceLeafInvocation(ComposedLeafInvocation):
    """Exact source and public SimReady conformance request."""

    source: ArtifactBinding
    params: SimReadyConformanceInput

    @model_validator(mode="after")
    def validate_conformance_paths(self) -> SimReadyConformanceLeafInvocation:
        if Path(self.params.asset_path) != Path(self.source.path):
            raise ValueError("SimReady conformance input differs from source binding")
        if not self.params.resume:
            raise ValueError("SimReady conformance requires replay-safe resume")
        root = Path(self.attempt_root)
        _require_lexical_descendant(
            root,
            Path(self.params.output_dir),
            label="SimReady conformance output_dir",
        )
        if self.params.report_path is None:
            raise ValueError("SimReady composed conformance requires report_path")
        _require_lexical_descendant(
            root,
            Path(self.params.report_path),
            label="SimReady conformance report_path",
        )
        return self


class SimReadyValidationLeafInvocation(ComposedLeafInvocation):
    """Exact source and public SimReady validation request."""

    source: ArtifactBinding
    params: SimReadyValidationInput

    @model_validator(mode="after")
    def validate_validation_paths(self) -> SimReadyValidationLeafInvocation:
        if Path(self.params.asset_path) != Path(self.source.path):
            raise ValueError("SimReady validation input differs from source binding")
        if not self.params.resume:
            raise ValueError("SimReady validation requires replay-safe resume")
        if self.params.report_path is None:
            raise ValueError("SimReady composed validation requires report_path")
        root = Path(self.attempt_root)
        _require_lexical_descendant(
            root,
            Path(self.params.report_path),
            label="SimReady validation report_path",
        )
        for label, value in (
            ("stdout_log_path", self.params.stdout_log_path),
            ("stderr_log_path", self.params.stderr_log_path),
        ):
            if value is not None:
                _require_lexical_descendant(
                    root,
                    Path(value),
                    label=f"SimReady validation {label}",
                )
        return self


class PortablePackageLeafInvocation(ComposedLeafInvocation):
    """Exact staged asset to localize into one canonical USDZ."""

    source: ArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...] = ()
    output_asset_path: str = Field(min_length=1)
    root_layer_name: str = Field(default="asset.usdc", pattern=r"^[^/\\]+\.usd[ac]$")

    @model_validator(mode="after")
    def validate_package_path(self) -> PortablePackageLeafInvocation:
        output = Path(self.output_asset_path)
        _require_lexical_descendant(
            Path(self.attempt_root), output, label="portable package output"
        )
        if output.suffix.lower() != ".usdz":
            raise ValueError("portable package output must use .usdz")
        return self


class CombinedResultLeafInvocation(ComposedLeafInvocation):
    """Exact terminal package, validation, render, and dependency ledger."""

    final_asset: ArtifactBinding
    required_predecessor_receipts: dict[str, ArtifactBinding] = Field(min_length=1)
    output_report_path: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_combined_path(self) -> CombinedResultLeafInvocation:
        _require_lexical_descendant(
            Path(self.attempt_root),
            Path(self.output_report_path),
            label="combined result output",
        )
        if Path(self.final_asset.path).suffix.lower() != ".usdz":
            raise ValueError("combined terminal asset must use .usdz")
        mandatory_predecessors = {
            ARTICULATION_PUBLISH_LEAF_ID,
            MATERIAL_ASSIGNMENT_LEAF_ID,
            PHYSICS_APPLY_LEAF_ID,
            PHYSICS_INSPECTION_LEAF_ID,
            PORTABLE_PACKAGE_LEAF_ID,
            SIMREADY_CONFORMANCE_LEAF_ID,
            SIMREADY_VALIDATION_LEAF_ID,
            TEXTURE_PUBLISH_LEAF_ID,
            FINAL_OVRTX_EVIDENCE_LEAF_ID,
        }
        missing = sorted(
            mandatory_predecessors.difference(self.required_predecessor_receipts)
        )
        if missing:
            raise ValueError(
                "combined terminal is missing mandatory predecessor receipts: "
                f"{missing}"
            )
        if list(self.required_predecessor_receipts) != sorted(
            self.required_predecessor_receipts
        ):
            raise ValueError("combined predecessor receipt IDs must be sorted")
        identities = [
            _identity(item) for item in self.required_predecessor_receipts.values()
        ]
        if len(identities) != len(set(identities)):
            raise ValueError("combined predecessor receipts must be unique")
        return self


class ComposedAssetLeafResult(_FrozenModel):
    """Closed progress/terminal result shared by the release-domain launchers."""

    schema_version: Literal["content-agent-workflows.composed-asset-leaf-result.v1"] = (
        "content-agent-workflows.composed-asset-leaf-result.v1"
    )
    leaf_id: str
    status: Literal["awaiting_decision", "awaiting_review", "passed", "failed"]
    native_status: str = Field(min_length=1, max_length=240)
    invocation: ArtifactBinding
    native_terminal_receipt: ArtifactBinding
    operation_indexes: tuple[ArtifactBinding, ...] = ()
    evidence_indexes: tuple[ArtifactBinding, ...] = ()
    evidence: tuple[ArtifactBinding, ...] = ()
    saved_stage_readbacks: tuple[ArtifactBinding, ...] = ()
    resource_claims: tuple[str, ...] = ()
    resource_release_receipts: tuple[ArtifactBinding, ...] = ()
    output_asset: ArtifactBinding | None = None
    summary: str = Field(min_length=1)
    error: str | None = None

    @model_validator(mode="after")
    def validate_result(self) -> ComposedAssetLeafResult:
        if self.status == "failed" and not self.error:
            raise ValueError("failed composed leaf result requires an error")
        if self.status != "failed" and self.error is not None:
            raise ValueError("nonfailed composed leaf result cannot carry an error")
        if self.status == "passed" and not self.saved_stage_readbacks:
            raise ValueError("passed composed leaf result requires readback evidence")
        if (
            self.status in _TERMINAL_STATUSES
            and self.resource_claims
            and not self.resource_release_receipts
        ):
            raise ValueError("terminal resource claims require release receipts")
        for values in (
            self.operation_indexes,
            self.evidence_indexes,
            self.evidence,
            self.saved_stage_readbacks,
            self.resource_release_receipts,
        ):
            identities = [_identity(item) for item in values]
            if len(identities) != len(set(identities)):
                raise ValueError("composed leaf result bindings must be unique")
        return self


class ComposedLeafFailureReceipt(_FrozenModel):
    """Stable native-exception boundary shared by every composed leaf."""

    schema_version: Literal["content-agent-workflows.composed-leaf-failure.v1"] = (
        "content-agent-workflows.composed-leaf-failure.v1"
    )
    leaf_id: str
    error_type: str = Field(min_length=1)
    error: str = Field(min_length=1)


class MaterialCoordinatorTerminalReceipt(_FrozenModel):
    """Exact terminal Material coordinator receipt projected by the asset leaf."""

    schema_version: Literal["content-agents.material-coordinator-result.v1"] = (
        "content-agents.material-coordinator-result.v1"
    )
    status: Literal["pass", "conditional"]
    output_usd_path: str = Field(min_length=1)
    output_usd_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: ArtifactBinding
    decision_patch: ArtifactBinding
    evidence: tuple[ArtifactBinding, ...] = Field(min_length=1)
    unresolved_issues: tuple[str, ...] = ()


class MaterialApplicationReviewReceipt(_FrozenModel):
    """Exact post-apply OVRTX evidence that awaited outer Material review."""

    schema_version: Literal["content-agents.material-application-receipt.v1"] = (
        "content-agents.material-application-receipt.v1"
    )
    status: Literal["review_required"] = "review_required"
    request: ArtifactBinding
    decision_patch: ArtifactBinding
    policy: ArtifactBinding
    materialized_usd: ArtifactBinding
    receipt_checkpoint_binding: ArtifactBinding
    final_render_bindings: tuple[ArtifactBinding, ...] = Field(min_length=1)
    evidence: tuple[ArtifactBinding, ...] = Field(min_length=1)
    applied_source_prim_paths: tuple[str, ...] = ()
    unresolved_issues: tuple[str, ...] = Field(min_length=1)


class MaterialSessionReleaseReceipt(_FrozenModel):
    """Observable release receipt for the Material leaf's usd-cli session."""

    status: Literal["released", "failed"]
    scene_tool: Literal["usd-cli"] = "usd-cli"
    session_id: str = Field(min_length=1)
    error: str | None = None


class PortablePackageReceipt(_FrozenModel):
    """Dependency-closed identity emitted by the portable package leaf."""

    schema_version: Literal["content-agent-workflows.portable-package-receipt.v1"] = (
        "content-agent-workflows.portable-package-receipt.v1"
    )
    source: ArtifactBinding
    source_dependencies: tuple[ArtifactBinding, ...]
    package: ArtifactBinding
    package_dependencies: tuple[ArtifactBinding, ...]
    root_layer_name: str


class PhysicsComponentInspectionReceipt(_FrozenModel):
    """Typed component catalog bound to the exact inspected source bytes."""

    schema_version: Literal[
        "content-agent-workflows.physics-component-inspection.v1"
    ] = "content-agent-workflows.physics-component-inspection.v1"
    source: ArtifactBinding
    components: tuple[dict[str, Any], ...]

    @field_validator("components")
    @classmethod
    def validate_components(
        cls,
        values: tuple[dict[str, Any], ...],
    ) -> tuple[dict[str, Any], ...]:
        for component in values:
            parse_physics_component_catalog_entry(component)
        return values


class PhysicsApplyLeafReceipt(_FrozenModel):
    """Exact native Physics result and the bytes that runtime validation assessed."""

    schema_version: Literal["content-agent-workflows.physics-apply-leaf-receipt.v1"] = (
        "content-agent-workflows.physics-apply-leaf-receipt.v1"
    )
    source: ArtifactBinding
    component_catalog: ArtifactBinding
    decision_patch: ArtifactBinding
    native_result: ArtifactBinding
    validation_evidence: ArtifactBinding | None = None
    output_asset: ArtifactBinding | None = None


class SimReadyConformanceLeafReceipt(_FrozenModel):
    """Exact conformance report plus its source and resulting staged bytes."""

    schema_version: Literal[
        "content-agent-workflows.simready-conformance-leaf-receipt.v1"
    ] = "content-agent-workflows.simready-conformance-leaf-receipt.v1"
    source: ArtifactBinding
    native_report: ArtifactBinding
    output_asset: ArtifactBinding | None = None


class CombinedAssetResultReceipt(_FrozenModel):
    """Canonical terminal index for one complete graph result."""

    schema_version: Literal["content-agent-workflows.combined-asset-result.v1"] = (
        "content-agent-workflows.combined-asset-result.v1"
    )
    final_asset: ArtifactBinding
    package_receipt: ArtifactBinding
    simready_validation: ArtifactBinding
    canonical_ovrtx_evidence: ArtifactBinding
    required_predecessor_receipts: dict[str, ArtifactBinding]


def _read_artifact(
    path: str | Path,
    *,
    retain_bytes: bool = False,
) -> tuple[ArtifactBinding, bytes]:
    candidate = Path(os.path.abspath(Path(path).expanduser()))
    digest = hashlib.sha256()
    size_bytes = 0
    chunks: list[bytes] = []
    try:
        with open_regular_file_no_follow(candidate) as (stream, before):
            descriptor = stream.fileno()
            while chunk := os.read(descriptor, 1024 * 1024):
                digest.update(chunk)
                size_bytes += len(chunk)
                if retain_bytes:
                    chunks.append(chunk)
            after = os.fstat(descriptor)
            stable_fields = (
                "st_dev",
                "st_ino",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
            if any(
                getattr(before, field) != getattr(after, field)
                for field in stable_fields
            ):
                raise ValueError(
                    f"composed leaf artifact changed while read: {candidate}"
                )
            if size_bytes != after.st_size:
                raise ValueError(
                    f"composed leaf artifact read was incomplete: {candidate}"
                )
            # Reopen the lexical locator through the same no-follow traversal
            # while the read descriptor is still pinned.  This proves the path
            # still names the exact inode and generation that supplied the
            # digest without closing the descriptor and resolving a mutable
            # pathname afterward.
            with open_regular_file_no_follow(candidate) as (_reopened, current):
                if any(
                    getattr(after, field) != getattr(current, field)
                    for field in stable_fields
                ):
                    raise ValueError(
                        f"composed leaf artifact path changed while read: {candidate}"
                    )
    except (ArtifactPathError, OSError) as exc:
        raise ValueError(
            f"composed leaf artifact cannot be opened safely: {candidate}"
        ) from exc
    return (
        ArtifactBinding(
            path=str(candidate),
            sha256=digest.hexdigest(),
            size_bytes=size_bytes,
        ),
        b"".join(chunks),
    )


def artifact_binding(path: str | Path) -> ArtifactBinding:
    """Bind one regular, non-symlink file for a checked-in leaf launcher."""

    return _read_artifact(path)[0]


def read_artifact_binding_and_bytes(
    path: str | Path,
) -> tuple[ArtifactBinding, bytes]:
    """Bind and retain one artifact through the same descriptor-pinned read."""

    return _read_artifact(path, retain_bytes=True)


def _canonical_json_bytes(model: BaseModel) -> bytes:
    return (
        json.dumps(
            model.model_dump(mode="json"),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
        )
        + "\n"
    ).encode("utf-8")


def write_canonical_json(path: str | Path, model: BaseModel) -> ArtifactBinding:
    """Write one immutable canonical JSON artifact without replacing bytes."""

    destination = Path(os.path.abspath(Path(path).expanduser()))
    payload = _canonical_json_bytes(model)
    try:
        with open_confined_directory(
            destination.parent,
            create=True,
            mode=0o700,
        ) as parent_descriptor:
            created = write_bytes_to_confined(
                parent_descriptor,
                destination.name,
                payload,
                overwrite=False,
                file_mode=0o600,
            )
    except (ArtifactPathError, OSError) as exc:
        raise ValueError(
            f"canonical composed leaf destination is unsafe: {destination}"
        ) from exc
    if not created:
        raise FileExistsError(destination)
    observed = artifact_binding(destination)
    expected = ArtifactBinding(
        path=str(destination),
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )
    if observed != expected:
        raise ValueError(
            f"canonical composed leaf destination changed after write: {destination}"
        )
    return observed


def write_or_verify_canonical_json(
    path: str | Path,
    model: BaseModel,
) -> ArtifactBinding:
    """Create canonical JSON once, or prove an exact converged replay."""

    destination = Path(os.path.abspath(Path(path).expanduser()))
    try:
        return write_canonical_json(destination, model)
    except FileExistsError:
        pass
    observed = artifact_binding(destination)
    expected = _canonical_json_bytes(model)
    if observed.sha256 != hashlib.sha256(
        expected
    ).hexdigest() or observed.size_bytes != len(expected):
        raise ValueError(f"canonical composed leaf artifact changed: {destination}")
    return observed


def _identity(binding: ArtifactBinding) -> tuple[str, str, int]:
    return binding.path, binding.sha256, binding.size_bytes


def _require_lexical_descendant(root: Path, candidate: Path, *, label: str) -> None:
    if not root.is_absolute() or not candidate.is_absolute():
        raise ValueError(f"{label} must be absolute")
    canonical_root = Path(os.path.abspath(root))
    canonical_candidate = Path(os.path.abspath(candidate))
    if root != canonical_root or candidate != canonical_candidate:
        raise ValueError(f"{label} must use a canonical absolute path")
    try:
        relative = canonical_candidate.relative_to(canonical_root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain below attempt_root") from exc
    if not relative.parts:
        raise ValueError(f"{label} must name a descendant of attempt_root")


def prepare_confined_output_path(
    root: str | Path,
    candidate: str | Path,
    *,
    label: str,
    directory: bool = False,
) -> Path:
    """Create and verify an output's directory chain beneath an attempt root.

    Invocation validation is intentionally side-effect free.  Launchers call
    this immediately before handing path-based outputs to a native workflow so
    every previously missing component is created relative to a held attempt
    descriptor and every existing component is opened with no-follow semantics.
    """

    root_path = Path(root)
    candidate_path = Path(candidate)
    _require_lexical_descendant(root_path, candidate_path, label=label)
    relative = candidate_path.relative_to(root_path)
    directory_relative = relative.parent
    try:
        with open_confined_directory(root_path) as root_descriptor:
            if directory_relative.parts:
                with open_confined_directory_at(
                    root_descriptor,
                    directory_relative.as_posix(),
                    create=True,
                    mode=0o700,
                ):
                    pass
            if directory:
                try:
                    with open_confined_directory_at(
                        root_descriptor,
                        relative.as_posix(),
                    ):
                        pass
                except FileNotFoundError:
                    # The native workflow owns creation of its final output
                    # directory. Its now-confined parent is created above.
                    pass
            else:
                try:
                    with open_confined_regular_file(
                        root_descriptor,
                        relative.as_posix(),
                    ) as (_stream, metadata):
                        if metadata.st_nlink != 1:
                            raise ValueError(
                                f"{label} must not be a hard-linked output"
                            )
                except FileNotFoundError:
                    pass
    except (ArtifactPathError, OSError) as exc:
        raise ValueError(f"{label} contains an unsafe directory component") from exc
    return candidate_path


def _verify_binding(
    binding: ArtifactBinding,
    *,
    label: str,
    required_root: Path | None = None,
) -> Path:
    candidate = Path(binding.path)
    if not candidate.is_absolute():
        raise ValueError(f"{label} path must be absolute")
    if required_root is not None:
        _require_lexical_descendant(required_root, candidate, label=label)
    observed = artifact_binding(candidate)
    if observed != binding:
        raise ValueError(f"{label} identity changed: {candidate}")
    return candidate


def _load_bound_json(binding: ArtifactBinding, *, label: str) -> dict[str, Any]:
    if not Path(binding.path).is_absolute():
        raise ValueError(f"{label} path must be absolute")
    try:
        observed, payload_bytes = _read_artifact(binding.path, retain_bytes=True)
        if observed != binding:
            raise ValueError(f"{label} identity changed: {binding.path}")
        payload = json.loads(payload_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _load_receipt_result[ModelT: BaseModel](
    receipt: AssetLeafReceipt,
    model: type[ModelT],
    *,
    label: str,
) -> ModelT:
    if receipt.result is None:
        raise ValueError(f"{label} omitted its exact graph result")
    return model.model_validate(_load_bound_json(receipt.result, label=label))


def _verify_common_result(
    result: ComposedAssetLeafResult,
    context: AssetLeafProjectionContext,
) -> Path:
    if result.leaf_id != context.leaf_id:
        raise ValueError("composed leaf result belongs to another leaf")
    context.require_invocation_artifact(result.invocation)
    if result.status not in _TERMINAL_STATUSES:
        raise ValueError("composed leaf progress cannot be projected as terminal")
    attempt_root = Path(context.invocation_artifact.path).parent
    _verify_binding(
        result.native_terminal_receipt,
        label="native terminal receipt",
        required_root=attempt_root,
    )
    for label, binding in (
        *(("operation index", item) for item in result.operation_indexes),
        *(("evidence index", item) for item in result.evidence_indexes),
        *(("evidence", item) for item in result.evidence),
        *(("saved-stage readback", item) for item in result.saved_stage_readbacks),
        *(("resource release", item) for item in result.resource_release_receipts),
    ):
        # Composition state separately confines every projected path to the
        # active attempt; rehash here so stale bytes fail before that transition.
        _verify_binding(binding, label=label)
    if result.output_asset is not None:
        _verify_binding(result.output_asset, label="composed leaf output asset")
        if result.output_asset not in result.saved_stage_readbacks:
            raise ValueError(
                "composed leaf output is absent from saved-stage readbacks"
            )
    return attempt_root


def resolve_combined_predecessor_evidence(
    invocation: CombinedResultLeafInvocation,
) -> tuple[ArtifactBinding, ArtifactBinding, ArtifactBinding]:
    """Resolve mandatory predecessor receipts to exact package/validation/render."""

    _verify_binding(invocation.final_asset, label="combined final asset")
    receipts: dict[str, AssetLeafReceipt] = {}
    expected_run_root = _leaf_receipt_run_root(
        Path(invocation.attempt_root) / "leaf_receipt.json",
        leaf_id=COMBINED_RESULT_LEAF_ID,
        attempt=None,
    )
    run = AssetCompositionRun.model_validate(
        _load_bound_json(
            artifact_binding(expected_run_root / "asset_run.json"),
            label="active asset run state",
        )
    )
    if run.selected_mode != "agentic" or run.execution_graph is None:
        raise ValueError("combined result requires a frozen agentic graph")
    graph = AssetExecutionGraph.model_validate(
        _load_bound_json(run.execution_graph, label="frozen execution graph")
    )
    combined_state = run.leaf_states.get(COMBINED_RESULT_LEAF_ID)
    running_context = (
        combined_state is not None
        and combined_state.status == "running"
        and run.current_leaf_id == COMBINED_RESULT_LEAF_ID
        and run.coordinator.next_action == "execute_leaf"
    )
    completed_context = (
        combined_state is not None
        and combined_state.status == "passed"
        and combined_state.receipt is not None
    )
    if not running_context and not completed_context:
        raise ValueError("combined result is not an admissible graph attempt")
    assert combined_state is not None
    combined_index = graph.selected_leaf_ids.index(COMBINED_RESULT_LEAF_ID) + 1
    expected_attempt_root = (
        expected_run_root
        / "leaves"
        / f"{combined_index:03d}-{COMBINED_RESULT_LEAF_ID}"
        / "attempts"
        / f"{combined_state.attempt_count:02d}"
    )
    if Path(invocation.attempt_root).resolve(strict=True) != expected_attempt_root:
        raise ValueError("combined invocation is outside its active graph attempt")
    if set(invocation.required_predecessor_receipts) != set(combined_state.depends_on):
        raise ValueError("combined invocation differs from frozen graph dependencies")
    graph_nodes = {node.leaf_id: node for node in graph.nodes}
    run_identity: tuple[str, str, str, str] | None = None
    for leaf_id, binding in invocation.required_predecessor_receipts.items():
        state = run.leaf_states.get(leaf_id)
        node = graph_nodes.get(leaf_id)
        if (
            state is None
            or node is None
            or state.status != "passed"
            or state.receipt != binding
        ):
            raise ValueError(
                f"combined predecessor is not the graph-state receipt: {leaf_id}"
            )
        try:
            receipt = AssetLeafReceipt.model_validate(
                _load_bound_json(binding, label=f"{leaf_id} predecessor receipt")
            )
        except ValueError as exc:
            raise ValueError(
                f"combined predecessor receipt is invalid: {leaf_id}"
            ) from exc
        if (
            receipt.schema_version != ASSET_LEAF_RECEIPT_SCHEMA_VERSION
            or receipt.leaf_id != leaf_id
            or receipt.requirement != "required"
            or receipt.native_disposition != "passed"
            or receipt.graph_digest != graph.graph_digest
            or receipt.leaf_catalog_digest != graph.leaf_catalog_digest
            or receipt.sole_coordinator_identity_digest
            != graph.sole_coordinator_identity_digest
            or receipt.descriptor_digest != node.descriptor_digest
            or receipt.depends_on != node.depends_on
        ):
            raise ValueError(
                f"combined predecessor receipt is not a required pass: {leaf_id}"
            )
        expected_dependency_receipts: dict[str, ArtifactBinding] = {}
        for dependency in node.depends_on:
            dependency_state = run.leaf_states.get(dependency)
            if dependency_state is None or dependency_state.receipt is None:
                raise ValueError(
                    f"combined predecessor has unresolved dependency: {leaf_id}"
                )
            expected_dependency_receipts[dependency] = dependency_state.receipt
        if receipt.dependency_receipts != expected_dependency_receipts:
            raise ValueError(
                f"combined predecessor dependency custody changed: {leaf_id}"
            )
        receipt_run_root = _leaf_receipt_run_root(
            Path(binding.path),
            leaf_id=leaf_id,
            attempt=receipt.attempt,
        )
        if receipt_run_root != expected_run_root:
            raise ValueError(f"combined predecessor belongs to another run: {leaf_id}")
        identity = (
            receipt.run_id,
            receipt.graph_digest,
            receipt.leaf_catalog_digest,
            receipt.sole_coordinator_identity_digest,
        )
        if run_identity is None:
            run_identity = identity
        elif identity != run_identity:
            raise ValueError("combined predecessor receipts disagree on run identity")
        receipts[leaf_id] = receipt

    package_graph_receipt = receipts[PORTABLE_PACKAGE_LEAF_ID]

    articulation_result = _load_receipt_result(
        receipts[ARTICULATION_PUBLISH_LEAF_ID],
        ArticulationFocusedLeafResult,
        label="Articulation publish result",
    )
    if (
        articulation_result.leaf_id != ARTICULATION_PUBLISH_LEAF_ID
        or ArtifactBinding.model_validate(
            articulation_result.invocation.model_dump(mode="json")
        )
        != receipts[ARTICULATION_PUBLISH_LEAF_ID].invocation
        or articulation_result.native_disposition != "passed"
        or articulation_result.output is None
    ):
        raise ValueError("Articulation predecessor omitted its exact passed output")
    articulation_output = ArtifactBinding.model_validate(
        articulation_result.output.model_dump(mode="json")
    )
    _verify_binding(articulation_output, label="Articulation published output")

    material_receipt = receipts[MATERIAL_ASSIGNMENT_LEAF_ID]
    material_invocation = MaterialAssignmentLeafInvocation.model_validate(
        _load_bound_json(material_receipt.invocation, label="Material invocation")
    )
    material_result = _load_receipt_result(
        material_receipt,
        ComposedAssetLeafResult,
        label="Material result",
    )
    if (
        material_invocation.source != articulation_output
        or material_result.leaf_id != MATERIAL_ASSIGNMENT_LEAF_ID
        or material_result.invocation != material_receipt.invocation
        or material_result.status != "passed"
        or material_result.output_asset is None
    ):
        raise ValueError("Material predecessor does not consume Articulation output")
    _verify_binding(
        material_result.output_asset,
        label="Material published output",
    )

    texture_result = _load_receipt_result(
        receipts[TEXTURE_PUBLISH_LEAF_ID],
        TextureFocusedLeafResult,
        label="Texture publish result",
    )
    texture_source = ArtifactBinding.model_validate(
        texture_result.source.model_dump(mode="json")
    )
    texture_output = ArtifactBinding.model_validate(
        texture_result.output.model_dump(mode="json")
    )
    if (
        texture_result.leaf_id != TEXTURE_PUBLISH_LEAF_ID
        or ArtifactBinding.model_validate(
            texture_result.invocation.model_dump(mode="json")
        )
        != receipts[TEXTURE_PUBLISH_LEAF_ID].invocation
        or texture_result.native_disposition != "passed"
        or texture_source != material_result.output_asset
    ):
        raise ValueError("Texture predecessor does not consume Material output")
    _verify_binding(texture_output, label="Texture published output")

    inspection_receipt = receipts[PHYSICS_INSPECTION_LEAF_ID]
    inspection_invocation = PhysicsInspectionLeafInvocation.model_validate(
        _load_bound_json(
            inspection_receipt.invocation,
            label="Physics inspection invocation",
        )
    )
    inspection_native = inspection_receipt.native_terminal_receipt
    if inspection_native is None:
        raise ValueError("Physics inspection omitted its component catalog")
    inspection_catalog = PhysicsComponentInspectionReceipt.model_validate(
        _load_bound_json(inspection_native, label="Physics component catalog")
    )
    if (
        inspection_invocation.source != texture_output
        or inspection_catalog.source != texture_output
    ):
        raise ValueError("Physics inspection does not consume Texture output")

    physics_receipt = receipts[PHYSICS_APPLY_LEAF_ID]
    physics_invocation = PhysicsApplyLeafInvocation.model_validate(
        _load_bound_json(physics_receipt.invocation, label="Physics apply invocation")
    )
    physics_result = _load_receipt_result(
        physics_receipt,
        ComposedAssetLeafResult,
        label="Physics apply result",
    )
    if (
        physics_invocation.source != texture_output
        or physics_invocation.component_catalog != inspection_native
        or physics_result.leaf_id != PHYSICS_APPLY_LEAF_ID
        or physics_result.invocation != physics_receipt.invocation
        or physics_result.status != "passed"
        or physics_result.output_asset is None
    ):
        raise ValueError("Physics apply does not consume exact inspection custody")
    _verify_binding(physics_result.output_asset, label="Physics published output")

    conformance_receipt = receipts[SIMREADY_CONFORMANCE_LEAF_ID]
    conformance_invocation = SimReadyConformanceLeafInvocation.model_validate(
        _load_bound_json(
            conformance_receipt.invocation,
            label="SimReady conformance invocation",
        )
    )
    conformance_result = _load_receipt_result(
        conformance_receipt,
        ComposedAssetLeafResult,
        label="SimReady conformance result",
    )
    if (
        conformance_invocation.source != physics_result.output_asset
        or conformance_result.leaf_id != SIMREADY_CONFORMANCE_LEAF_ID
        or conformance_result.invocation != conformance_receipt.invocation
        or conformance_result.status != "passed"
        or conformance_result.output_asset is None
    ):
        raise ValueError("SimReady conformance does not consume Physics output")
    _verify_binding(
        conformance_result.output_asset,
        label="SimReady conformed output",
    )

    package_native = package_graph_receipt.native_terminal_receipt
    if package_native is None:
        raise ValueError("portable-package predecessor omitted its native receipt")
    package_receipt = PortablePackageReceipt.model_validate(
        _load_bound_json(package_native, label="portable-package native receipt")
    )
    if (
        PortablePackageLeafInvocation.model_validate(
            _load_bound_json(
                package_graph_receipt.invocation,
                label="portable-package invocation",
            )
        ).source
        != conformance_result.output_asset
        or package_receipt.source != conformance_result.output_asset
        or package_receipt.package != invocation.final_asset
        or package_graph_receipt.saved_stage_readbacks[-1:] != [invocation.final_asset]
    ):
        raise ValueError("portable-package predecessor binds another final asset")

    validation_graph_receipt = receipts[SIMREADY_VALIDATION_LEAF_ID]
    validation_native = validation_graph_receipt.native_terminal_receipt
    if validation_native is None:
        raise ValueError("SimReady predecessor omitted its native receipt")
    validation = SimReadyValidationReport.model_validate(
        _load_bound_json(validation_native, label="SimReady predecessor report")
    )
    if (
        SimReadyValidationLeafInvocation.model_validate(
            _load_bound_json(
                validation_graph_receipt.invocation,
                label="SimReady validation invocation",
            )
        ).source
        != invocation.final_asset
        or not validation.passed
        or Path(validation.asset_path) != Path(invocation.final_asset.path)
        or validation.asset_sha256 != invocation.final_asset.sha256
    ):
        raise ValueError("SimReady predecessor does not pass the final package bytes")

    ovrtx_graph_receipt = receipts[FINAL_OVRTX_EVIDENCE_LEAF_ID]
    ovrtx_invocation = CanonicalOvrtxEvidenceLeafInvocation.model_validate(
        _load_bound_json(
            ovrtx_graph_receipt.invocation,
            label="canonical OVRTX predecessor invocation",
        )
    )
    if Path(ovrtx_invocation.post_mutation_usd) != Path(invocation.final_asset.path):
        raise ValueError("canonical OVRTX predecessor rendered another asset path")
    payload_candidates: list[
        tuple[ArtifactBinding, CanonicalVisualEvidencePayload]
    ] = []
    for index, binding in enumerate(ovrtx_graph_receipt.saved_stage_readbacks):
        try:
            payload = CanonicalVisualEvidencePayload.model_validate(
                _load_bound_json(
                    binding,
                    label=f"canonical OVRTX saved-stage readback {index + 1}",
                )
            )
        except ValueError:
            continue
        payload_candidates.append((binding, payload))
    if len(payload_candidates) != 1:
        raise ValueError("canonical OVRTX predecessor lacks one exact payload")
    ovrtx_payload_binding, ovrtx_payload = payload_candidates[0]
    rendered_asset = ArtifactBinding.model_validate(
        ovrtx_payload.post_mutation_output.model_dump(mode="json")
    )
    if (
        rendered_asset != invocation.final_asset
        or ovrtx_graph_receipt.native_status != "pass"
        or ovrtx_graph_receipt.projector_id != "asset.projector.final-ovrtx-evidence.v1"
    ):
        raise ValueError(
            "canonical OVRTX predecessor does not bind final package bytes"
        )
    return package_native, validation_native, ovrtx_payload_binding


def _leaf_receipt_run_root(
    path: Path,
    *,
    leaf_id: str,
    attempt: int | None,
) -> Path:
    """Recover and validate the canonical graph-leaf receipt directory shape."""

    resolved = path.resolve(strict=False)
    attempt_dir = resolved.parent
    attempts_dir = attempt_dir.parent
    leaf_dir = attempts_dir.parent
    leaves_dir = leaf_dir.parent
    if (
        resolved.name != "leaf_receipt.json"
        or attempts_dir.name != "attempts"
        or leaves_dir.name != "leaves"
        or not leaf_dir.name.endswith(f"-{leaf_id}")
        or len(leaf_dir.name.split("-", 1)[0]) != 3
        or not leaf_dir.name.split("-", 1)[0].isdigit()
        or not attempt_dir.name.isdigit()
        or (attempt is not None and attempt_dir.name != f"{attempt:02d}")
    ):
        raise ValueError(f"noncanonical graph leaf receipt path: {path}")
    return leaves_dir.parent


def _validate_domain_result(
    invocation: BaseModel,
    result: ComposedAssetLeafResult,
) -> None:
    if isinstance(invocation, MaterialAssignmentLeafInvocation):
        for label, binding in (
            ("Material source", invocation.source),
            ("Material manifest", invocation.materials_yaml),
            ("Material library", invocation.materials_usd),
            *(
                ("Material library dependency", item)
                for item in invocation.materials_usd_dependencies
            ),
            *(
                ("Material reference image", item)
                for item in invocation.reference_images
            ),
            *(("Material reference file", item) for item in invocation.reference_files),
        ):
            _verify_binding(binding, label=label)
        if _validate_exception_failure(result):
            return
        if result.status == "passed":
            attempt_root = Path(invocation.attempt_root).resolve(strict=True)
            if result.output_asset is None or Path(result.output_asset.path) != Path(
                invocation.output_asset_path
            ):
                raise ValueError("Material result output differs from its invocation")
            expected_terminal_path = attempt_root / "native" / "coordinator_result.json"
            if Path(result.native_terminal_receipt.path) != expected_terminal_path:
                raise ValueError("Material native result path differs from invocation")
            material_native = MaterialCoordinatorTerminalReceipt.model_validate(
                _load_bound_json(
                    result.native_terminal_receipt,
                    label="Material coordinator result",
                )
            )
            if (
                material_native.status != "pass"
                or result.native_status != "pass"
                or material_native.output_usd_path != result.output_asset.path
                or material_native.output_usd_sha256 != result.output_asset.sha256
                or material_native.unresolved_issues
            ):
                raise ValueError("Material native result is not an exact pass")
            for label, binding in (
                ("Material coordinator request", material_native.request),
                ("Material decision patch", material_native.decision_patch),
                *(
                    ("Material native evidence", item)
                    for item in material_native.evidence
                ),
            ):
                _verify_binding(binding, label=label, required_root=attempt_root)
            required_native_evidence = {
                result.native_terminal_receipt,
                material_native.request,
                material_native.decision_patch,
                *material_native.evidence,
            }
            if not required_native_evidence.issubset(result.evidence):
                raise ValueError("Material result omits native evidence bindings")

            application_path = (
                attempt_root / "native" / "raw" / "material_application_receipt.json"
            )
            application_bindings = tuple(
                item
                for item in result.saved_stage_readbacks
                if Path(item.path) == application_path
            )
            if len(application_bindings) != 1:
                raise ValueError("Material result omits its exact application receipt")
            application = MaterialApplicationReviewReceipt.model_validate(
                _load_bound_json(
                    application_bindings[0],
                    label="Material application receipt",
                )
            )
            expected_staged_output = (
                attempt_root / "native" / "output" / "materialized.usda"
            )
            if (
                application.request != material_native.request
                or application.decision_patch != material_native.decision_patch
                or Path(application.materialized_usd.path) != expected_staged_output
                or application.materialized_usd not in material_native.evidence
            ):
                raise ValueError(
                    "Material staged application custody differs from the "
                    "coordinator result"
                )
            application_evidence = (
                application.request,
                application.decision_patch,
                application.policy,
                application.materialized_usd,
                application.receipt_checkpoint_binding,
                *application.final_render_bindings,
                *application.evidence,
            )
            for binding in application_evidence:
                _verify_binding(
                    binding,
                    label="Material application evidence",
                    required_root=attempt_root,
                )
            required_result_evidence = {
                *application.final_render_bindings,
                *application.evidence,
            }
            if not required_result_evidence.issubset(result.evidence):
                raise ValueError("Material result omits post-apply OVRTX evidence")
            if result.saved_stage_readbacks != (
                result.native_terminal_receipt,
                application_bindings[0],
                application.materialized_usd,
                result.output_asset,
            ):
                raise ValueError("Material saved-stage readbacks are not exact")

            release_path = (
                attempt_root / "native" / "raw" / "material_session_release.json"
            )
            if (
                len(result.resource_release_receipts) != 1
                or Path(result.resource_release_receipts[0].path) != release_path
            ):
                raise ValueError("Material result omits its exact release receipt")
            release = MaterialSessionReleaseReceipt.model_validate(
                _load_bound_json(
                    result.resource_release_receipts[0],
                    label="Material session release receipt",
                )
            )
            if (
                release.status != "released"
                or release.error is not None
                or result.resource_claims != (f"usd-cli:{release.session_id}",)
            ):
                raise ValueError("Material usd-cli session was not cleanly released")
            if result.resource_release_receipts[0] not in result.evidence:
                raise ValueError("Material result omits release evidence")
        return
    if isinstance(invocation, PhysicsInspectionLeafInvocation):
        _verify_binding(invocation.source, label="Physics inspection source")
        if _validate_exception_failure(result):
            return
        inspection_receipt = PhysicsComponentInspectionReceipt.model_validate(
            _load_bound_json(
                result.native_terminal_receipt,
                label="Physics component catalog",
            )
        )
        if inspection_receipt.source != invocation.source or Path(
            result.native_terminal_receipt.path
        ) != Path(invocation.component_catalog_path):
            raise ValueError("Physics component inspection receipt is invalid")
        return
    if isinstance(invocation, PhysicsApplyLeafInvocation):
        for label, binding in (
            ("Physics source", invocation.source),
            ("Physics component catalog", invocation.component_catalog),
            ("Physics decision patch", invocation.decision_patch),
        ):
            _verify_binding(binding, label=label)
        if _validate_exception_failure(result):
            return
        apply_receipt = PhysicsApplyLeafReceipt.model_validate(
            _load_bound_json(
                result.native_terminal_receipt,
                label="Physics apply leaf receipt",
            )
        )
        if (
            apply_receipt.source != invocation.source
            or apply_receipt.component_catalog != invocation.component_catalog
            or apply_receipt.decision_patch != invocation.decision_patch
            or apply_receipt.output_asset != result.output_asset
        ):
            raise ValueError("Physics leaf receipt differs from its invocation")
        physics_native = PhysicsApplyWorkflowResult.model_validate(
            _load_bound_json(
                apply_receipt.native_result,
                label="Physics apply result",
            )
        )
        if result.status == "passed" and (
            physics_native.success is not True
            or physics_native.asset != invocation.source.path
            or physics_native.output_dir != invocation.params.output_dir.as_posix()
            or physics_native.validation_status != "pass"
            or result.output_asset is None
            or physics_native.physics_usd_path != result.output_asset.path
        ):
            raise ValueError("Physics native result is not an exact pass")
        if result.status == "passed":
            validation_binding = apply_receipt.validation_evidence
            if (
                validation_binding is None
                or physics_native.validation_evidence_path is None
                or Path(validation_binding.path)
                != Path(physics_native.validation_evidence_path)
                or validation_binding not in result.evidence
            ):
                raise ValueError(
                    "Physics pass omits its exact native validation evidence"
                )
            _verify_binding(
                validation_binding,
                label="Physics validation evidence",
                required_root=Path(invocation.attempt_root).resolve(strict=True),
            )
            validation_evidence = ValidationEvidence.model_validate(
                _load_bound_json(
                    validation_binding,
                    label="Physics validation evidence",
                )
            )
            assert result.output_asset is not None
            if (
                validation_evidence.schema_version != VALIDATION_EVIDENCE_SCHEMA_VERSION
                or validation_evidence.workflow != "physics_authoring"
                or Path(validation_evidence.asset) != Path(result.output_asset.path)
                or validation_evidence.metadata.get("asset_sha256")
                != result.output_asset.sha256
                or validation_evidence.sim_ready_status != "pass"
                or validation_evidence.failures
                or validation_evidence.warnings
                or validation_evidence.unresolved_issues
            ):
                raise ValueError(
                    "Physics validation evidence does not bind the exact clean output"
                )
            native_patch_path = physics_native.decision_patch_path
            if native_patch_path is None:
                raise ValueError("Physics native result omitted its decision patch")
            native_patch = _load_bound_json(
                artifact_binding(native_patch_path),
                label="native Physics decision patch",
            )
            outer_patch = _load_bound_json(
                invocation.decision_patch,
                label="outer Physics decision patch",
            )
            if native_patch != outer_patch:
                raise ValueError("Physics native decisions differ from the outer patch")
        return
    if isinstance(invocation, SimReadyConformanceLeafInvocation):
        _verify_binding(invocation.source, label="SimReady conformance source")
        if _validate_exception_failure(result):
            return
        conformance_receipt = SimReadyConformanceLeafReceipt.model_validate(
            _load_bound_json(
                result.native_terminal_receipt,
                label="SimReady conformance leaf receipt",
            )
        )
        if (
            conformance_receipt.source != invocation.source
            or conformance_receipt.output_asset != result.output_asset
        ):
            raise ValueError("SimReady conformance leaf receipt changed")
        conformance_native = SimReadyConformanceReport.model_validate(
            _load_bound_json(
                conformance_receipt.native_report,
                label="SimReady conformance report",
            )
        )
        if result.status == "passed" and (
            conformance_native.passed is not True
            or conformance_native.input_usd_path != invocation.source.path
            or result.output_asset is None
            or conformance_native.output_usd_path != result.output_asset.path
        ):
            raise ValueError("SimReady conformance result is not an exact pass")
        return
    if isinstance(invocation, SimReadyValidationLeafInvocation):
        _verify_binding(invocation.source, label="SimReady validation source")
        if _validate_exception_failure(result):
            return
        report_path = invocation.params.report_path
        if report_path is None or Path(result.native_terminal_receipt.path) != Path(
            report_path
        ):
            raise ValueError("SimReady validation report path differs from invocation")
        validation_native = _load_bound_json(
            result.native_terminal_receipt, label="SimReady validation report"
        )
        if result.status == "passed" and (
            validation_native.get("passed") is not True
            or validation_native.get("asset_path") != invocation.source.path
            or validation_native.get("asset_sha256") != invocation.source.sha256
        ):
            raise ValueError("SimReady validation result is not an exact pass")
        return
    if isinstance(invocation, PortablePackageLeafInvocation):
        _verify_binding(invocation.source, label="portable package source")
        for dependency in invocation.source_dependencies:
            _verify_binding(dependency, label="portable package source dependency")
        if _validate_exception_failure(result):
            return
        package_leaf_receipt = PortablePackageReceipt.model_validate(
            _load_bound_json(
                result.native_terminal_receipt, label="portable package receipt"
            )
        )
        package_member_order = validate_usdz_package_layout(
            Path(package_leaf_receipt.package.path)
        )
        actual_root_layer_name = package_member_order[0]
        if (
            package_leaf_receipt.source != invocation.source
            or tuple(package_leaf_receipt.source_dependencies)
            != invocation.source_dependencies
            or result.output_asset != package_leaf_receipt.package
            or Path(package_leaf_receipt.package.path)
            != Path(invocation.output_asset_path)
            or package_leaf_receipt.package_dependencies
            or package_leaf_receipt.root_layer_name != actual_root_layer_name
            or (
                Path(invocation.source.path).suffix.lower() != ".usdz"
                and package_leaf_receipt.root_layer_name != invocation.root_layer_name
            )
        ):
            raise ValueError("portable package receipt identity is invalid")
        return
    if not isinstance(invocation, CombinedResultLeafInvocation):
        raise TypeError(f"unsupported composed leaf invocation: {type(invocation)}")
    _verify_binding(invocation.final_asset, label="combined final asset")
    for leaf_id, binding in invocation.required_predecessor_receipts.items():
        _verify_binding(binding, label=f"{leaf_id} predecessor receipt")
    if _validate_exception_failure(result):
        return
    if Path(result.native_terminal_receipt.path) != Path(invocation.output_report_path):
        raise ValueError("combined result receipt path differs from invocation")
    package_receipt, simready_validation, canonical_ovrtx_evidence = (
        resolve_combined_predecessor_evidence(invocation)
    )
    combined_receipt = CombinedAssetResultReceipt.model_validate(
        _load_bound_json(
            result.native_terminal_receipt, label="combined result receipt"
        )
    )
    if combined_receipt != CombinedAssetResultReceipt(
        final_asset=invocation.final_asset,
        package_receipt=package_receipt,
        simready_validation=simready_validation,
        canonical_ovrtx_evidence=canonical_ovrtx_evidence,
        required_predecessor_receipts=invocation.required_predecessor_receipts,
    ):
        raise ValueError("combined result differs from its exact invocation")


def _validate_exception_failure(result: ComposedAssetLeafResult) -> bool:
    """Validate and recognize one exact checked-in launcher exception result."""

    if result.status != "failed" or result.native_status != "exception":
        return False
    failure = ComposedLeafFailureReceipt.model_validate(
        _load_bound_json(
            result.native_terminal_receipt,
            label="composed leaf failure receipt",
        )
    )
    if (
        failure.leaf_id != result.leaf_id
        or result.error != f"{failure.error_type}: {failure.error}"
        or result.output_asset is not None
        or result.evidence != (result.native_terminal_receipt,)
        or result.saved_stage_readbacks != (result.native_terminal_receipt,)
    ):
        raise ValueError("composed leaf exception result is not exact")
    return True


def _composed_leaf_projector(
    invocation: BaseModel,
    native_result: BaseModel,
    context: AssetLeafProjectionContext,
) -> AssetLeafProjectionPayload:
    """Project only an exact terminal result from the selected domain schema."""

    result = cast(ComposedAssetLeafResult, native_result)
    _verify_common_result(result, context)
    _validate_domain_result(invocation, result)
    disposition: Literal["passed", "failed"] = (
        "passed" if result.status == "passed" else "failed"
    )
    return AssetLeafProjectionPayload(
        native_disposition=disposition,
        native_status=result.native_status,
        native_terminal_receipt=result.native_terminal_receipt,
        operation_indexes=result.operation_indexes,
        evidence_indexes=result.evidence_indexes,
        evidence=result.evidence,
        saved_stage_readbacks=result.saved_stage_readbacks,
        resource_claims=result.resource_claims,
        resource_release_receipts=result.resource_release_receipts,
        summary=result.summary,
        error=result.error,
    )


def release_composed_asset_leaf_runtime_bundle() -> AssetLeafRuntimeBundle:
    """Expose complete Material, Physics, SimReady, and package coverage."""

    definitions: tuple[
        tuple[
            str,
            str,
            type[BaseModel],
            tuple[str, ...],
            tuple[LeafReceiptArtifactCategory, ...],
        ],
        ...,
    ] = (
        (
            MATERIAL_ASSIGNMENT_LEAF_ID,
            "material",
            MaterialAssignmentLeafInvocation,
            (),
            ("evidence", "saved_stage_readback", "resource_release"),
        ),
        (
            PHYSICS_INSPECTION_LEAF_ID,
            "physics-inspect",
            PhysicsInspectionLeafInvocation,
            (),
            ("evidence", "saved_stage_readback"),
        ),
        (
            PHYSICS_APPLY_LEAF_ID,
            "physics-apply",
            PhysicsApplyLeafInvocation,
            (PHYSICS_INSPECTION_LEAF_ID,),
            ("evidence", "saved_stage_readback"),
        ),
        (
            SIMREADY_CONFORMANCE_LEAF_ID,
            "simready-conform",
            SimReadyConformanceLeafInvocation,
            (),
            ("evidence", "saved_stage_readback"),
        ),
        (
            SIMREADY_VALIDATION_LEAF_ID,
            "simready-validate",
            SimReadyValidationLeafInvocation,
            (PORTABLE_PACKAGE_LEAF_ID, SIMREADY_CONFORMANCE_LEAF_ID),
            ("evidence", "saved_stage_readback"),
        ),
        (
            PORTABLE_PACKAGE_LEAF_ID,
            "package",
            PortablePackageLeafInvocation,
            (SIMREADY_CONFORMANCE_LEAF_ID,),
            ("evidence", "saved_stage_readback"),
        ),
        (
            COMBINED_RESULT_LEAF_ID,
            "combined",
            CombinedResultLeafInvocation,
            (
                ARTICULATION_PUBLISH_LEAF_ID,
                MATERIAL_ASSIGNMENT_LEAF_ID,
                PHYSICS_APPLY_LEAF_ID,
                PHYSICS_INSPECTION_LEAF_ID,
                PORTABLE_PACKAGE_LEAF_ID,
                SIMREADY_CONFORMANCE_LEAF_ID,
                SIMREADY_VALIDATION_LEAF_ID,
                TEXTURE_PUBLISH_LEAF_ID,
                FINAL_OVRTX_EVIDENCE_LEAF_ID,
            ),
            ("evidence", "saved_stage_readback"),
        ),
    )
    return AssetLeafRuntimeBundle.create(
        bundle_id="release-composed",
        bindings=(
            canonical_ovrtx_asset_leaf_runtime_binding(
                leaf_id=FINAL_OVRTX_EVIDENCE_LEAF_ID,
                projector_id="asset.projector.final-ovrtx-evidence.v1",
            ),
            *(
                AssetLeafRuntimeBinding.create(
                    leaf_id=leaf_id,
                    entrypoint=_ENTRYPOINT.format(operation=operation),
                    invocation_model=invocation_model,
                    result_model=ComposedAssetLeafResult,
                    projector_id=f"asset.projector.{leaf_id}",
                    projector=_composed_leaf_projector,
                    required_dependencies=dependencies,
                    required_artifact_categories=required_categories,
                )
                for (
                    leaf_id,
                    operation,
                    invocation_model,
                    dependencies,
                    required_categories,
                ) in definitions
            ),
        ),
    )


__all__ = [
    "ARTICULATION_PUBLISH_LEAF_ID",
    "COMBINED_RESULT_LEAF_ID",
    "FINAL_OVRTX_EVIDENCE_LEAF_ID",
    "MATERIAL_ASSIGNMENT_LEAF_ID",
    "PHYSICS_APPLY_LEAF_ID",
    "PHYSICS_INSPECTION_LEAF_ID",
    "PORTABLE_PACKAGE_LEAF_ID",
    "RELEASE_COMPOSED_LEAF_IDS",
    "SIMREADY_CONFORMANCE_LEAF_ID",
    "SIMREADY_VALIDATION_LEAF_ID",
    "TEXTURE_PUBLISH_LEAF_ID",
    "CombinedAssetResultReceipt",
    "CombinedResultLeafInvocation",
    "ComposedAssetLeafResult",
    "ComposedLeafInvocation",
    "ComposedLeafFailureReceipt",
    "MaterialApplicationReviewReceipt",
    "MaterialAssignmentLeafInvocation",
    "MaterialCoordinatorTerminalReceipt",
    "MaterialSessionReleaseReceipt",
    "PhysicsApplyLeafInvocation",
    "PhysicsApplyLeafReceipt",
    "PhysicsComponentInspectionReceipt",
    "PhysicsInspectionLeafInvocation",
    "PortablePackageLeafInvocation",
    "PortablePackageReceipt",
    "SimReadyConformanceLeafInvocation",
    "SimReadyConformanceLeafReceipt",
    "SimReadyValidationLeafInvocation",
    "artifact_binding",
    "prepare_confined_output_path",
    "read_artifact_binding_and_bytes",
    "release_composed_asset_leaf_runtime_bundle",
    "resolve_combined_predecessor_evidence",
    "write_canonical_json",
    "write_or_verify_canonical_json",
]
