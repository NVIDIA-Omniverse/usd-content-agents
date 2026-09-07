# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resumable render-and-reference Validation Agent workflow.

One invocation moves through four phases, in order.

**Identity.** ``_workflow_identity`` binds the run to the request digest, the
resolved policy digest, the backend digest, the path-and-content identity of
every source and reference artifact, and the executor's template versions.
Resume recomputes this and fails closed on any drift, so a changed prompt,
asset, reference, policy, backend, or template version can never be mistaken
for a continuation of an earlier run.

**Planning.** The executor proposes a plan; ``_bind_plan`` binds it to that
identity and to the run's artifact paths, ``_plan_digest`` freezes it, and
``_records_from_plan`` derives the ordered work items. The checkpoint holds
``workflow_identity``, ``plan_digest``, and ``ordered_work_item_ids`` as
immutable for the life of the run.

**Execution.** Work items run in plan order. Each is claimed atomically through
``_mark_running`` so two runners cannot execute the same check, executes inside
a descriptor-pinned attempt directory, and is accepted together with the
identities of the evidence it produced. Cancellation is checked between items
and cannot be overwritten by a late success. An accepted result whose evidence
has since gone missing or stale is reopened on resume, along with everything
downstream of it.

**Finalization.** ``store.finalize()`` publishes the request, plan, result,
evidence, and summary while excluding concurrent checkpoint mutation.
``validation_result.json`` is the terminal bundle's commit marker: it is
removed before the bundle is rewritten and written last, so its presence always
implies a complete bundle at the checkpoint revision it names.

On POSIX every artifact operation is descriptor-relative and identity-checked
(``O_NOFOLLOW``, ``dir_fd``, and inode comparison) to resist symlink and ABA
swaps. Attempt isolation goes through ``/proc/self/fd``, which is why execution
requires Linux, a Linux container, or WSL2.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, Protocol
from urllib.parse import ParseResult, unquote, urlparse

from world_understanding.utils.credentials import (
    InlineSecretError,
    ensure_no_inline_secrets,
)
from world_understanding.utils.usd.asset_paths import collect_layer_authored_asset_paths
from world_understanding.validation import (
    ValidationIssue,
    ValidationPlan,
    ValidationRequest,
    ValidationResult,
    ValidationTemplateContext,
    ValidationTemplateResult,
    aggregate_validation_verdict,
)
from world_understanding.validation.cli import scaffold_policy_from_request
from world_understanding.validation.scaffold_runner import (
    ScaffoldValidationStepExecutor,
)

from content_agent_workflows.common.artifacts import (
    _directory_chain_matches,
    _open_directory_no_symlinks,
    artifact_set_digest,
    atomic_write_json,
    atomic_write_json_at,
    file_sha256,
    load_json,
)
from content_agent_workflows.common.domain_execution import (
    domain_execution_context_from_metadata,
)

from .checkpoint import (
    FileValidationCheckpointStore,
    ValidationCheckpointError,
    ValidationCheckpointStore,
)
from .finalizer import (
    ValidationWorkflowPaths,
    finalize_validation_workflow,
    finalized_validation_artifacts_are_valid,
    write_validation_planning_artifacts,
)
from .models import (
    VALIDATION_EXTERNAL_RUNTIME_DEPENDENCY_PATHS,
    VALIDATION_WORK_ITEM_SCHEMA_VERSION,
    ValidationAcceptedTemplateResult,
    ValidationArtifactIdentity,
    ValidationWorkflowCheckpoint,
    ValidationWorkflowIdentity,
    ValidationWorkflowRun,
    ValidationWorkflowStatus,
    ValidationWorkItemRecord,
    ValidationWorkItemState,
    validation_external_dependency_digest,
    validation_workflow_identity_digest,
)

VISUAL_TEMPLATE_ORDER = ("render_valid", "look_right")
VALIDATION_TEMPLATE_CAPABILITIES: Final[Mapping[str, tuple[str, ...]]] = {
    "render_valid": ("renderer_or_existing_render_evidence",),
    "look_right": ("optional_external_critique", "visual_evidence"),
    "physics_sane": ("usd_schema_runtime",),
    "physical_behavior": ("existing_simulator_or_runtime_evidence",),
}
_CANONICAL_VISUAL_EVIDENCE_MODES = {"canonical_usd", "canonical-usd"}
_RUN_OWNER_FILE_NAME = ".validation_workflow_owner.json"
_RUNTIME_MDL_IDENTITY_PREFIX: Final = "mdl://runtime/"
_PUBLIC_RUNTIME_MDL_MODULES: Final = frozenset(
    path.removeprefix(_RUNTIME_MDL_IDENTITY_PREFIX)
    for path in VALIDATION_EXTERNAL_RUNTIME_DEPENDENCY_PATHS
)

ProgressCallback = Callable[[ValidationWorkflowCheckpoint], None]
ExternalCancellationCheck = Callable[[], bool]
ArtifactRole = Literal["source", "reference", "evidence"]


class ValidationWorkflowError(RuntimeError):
    """Raised when a resumable Validation workflow cannot proceed safely."""


class ValidationWorkflowIdentityMismatch(ValidationWorkflowError):
    """Raised when a checkpoint belongs to different immutable inputs."""


class _ValidationArtifactFingerprintError(ValidationWorkflowError):
    """Raised when an immutable input's dependency closure cannot be hashed."""


class ValidationStepExecutor(Protocol):
    """Planner and one-template execution boundary used by the workflow."""

    @property
    def template_versions(self) -> Mapping[str, str]:
        """Return stable implementation versions keyed by template name."""

    def plan(
        self,
        request: ValidationRequest,
        *,
        working_dir: Path,
    ) -> ValidationPlan:
        """Build a stable Validation Agent plan."""

    def run(
        self,
        template_name: str,
        context: ValidationTemplateContext,
    ) -> ValidationTemplateResult:
        """Execute exactly one planned template."""


class ValidationCancellationToken:
    """Thread-safe cooperative cancellation signal."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()


def _canonical_digest(payload: Any) -> str:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_path(path: str | Path, *, base_dir: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _best_effort_resolved_path(path: str | Path, *, base_dir: Path) -> str:
    try:
        return str(_resolve_path(path, base_dir=base_dir))
    except (OSError, RuntimeError, ValueError):
        return str(path)


def _artifact_identity(
    path: str | Path,
    *,
    role: ArtifactRole,
    base_dir: Path,
) -> ValidationArtifactIdentity:
    resolved = _resolve_path(path, base_dir=base_dir)
    if resolved.is_file():
        return ValidationArtifactIdentity(
            role=role,
            path=str(resolved),
            kind="file",
            sha256=file_sha256(resolved),
        )
    if resolved.is_dir():
        if any(entry.is_symlink() for entry in resolved.rglob("*")):
            raise ValueError(f"Artifact directory cannot contain symlinks: {resolved}")
        return ValidationArtifactIdentity(
            role=role,
            path=str(resolved),
            kind="directory",
            sha256=artifact_set_digest([resolved]),
        )
    return ValidationArtifactIdentity(
        role=role,
        path=str(resolved),
        kind="missing",
    )


def _artifact_identities(
    paths: Sequence[str | Path],
    *,
    role: ArtifactRole,
    base_dir: Path,
) -> tuple[ValidationArtifactIdentity, ...]:
    identities: list[ValidationArtifactIdentity] = []
    seen: set[str] = set()
    for path in paths:
        identity = _artifact_identity(path, role=role, base_dir=base_dir)
        candidates: tuple[ValidationArtifactIdentity, ...] = (identity,)
        if (
            role == "source"
            and identity.kind == "file"
            and Path(identity.path).suffix.lower() in {".usd", ".usda", ".usdc"}
        ):
            dependencies, runtime_modules = _usd_dependency_closure(Path(identity.path))
            candidates = tuple(
                _artifact_identity(
                    dependency,
                    role=role,
                    base_dir=base_dir,
                )
                for dependency in dependencies
            ) + tuple(
                _external_runtime_dependency_identity(module)
                for module in runtime_modules
            )
        for candidate in candidates:
            if candidate.path in seen:
                continue
            identities.append(candidate)
            seen.add(candidate.path)
    return tuple(identities)


def _usd_dependency_paths(root: Path) -> tuple[Path, ...]:
    dependencies, _runtime_modules = _usd_dependency_closure(root)
    return dependencies


def _external_runtime_dependency_identity(
    module: str,
) -> ValidationArtifactIdentity:
    path = f"{_RUNTIME_MDL_IDENTITY_PREFIX}{module}"
    return ValidationArtifactIdentity(
        role="source",
        path=path,
        kind="external",
        sha256=validation_external_dependency_digest(path),
    )


def _usd_dependency_closure(root: Path) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    try:
        from pxr import Ar, Sdf, UsdUtils
    except ImportError as exc:
        raise ValidationWorkflowError(
            "OpenUSD Python bindings are required to fingerprint USD inputs."
        ) from exc

    try:
        layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(root))
    except Exception as exc:
        raise _ValidationArtifactFingerprintError(
            f"Could not fingerprint USD input dependencies: {root}"
        ) from exc
    unresolved_paths = tuple(sorted(str(value) for value in unresolved))

    dependencies: set[Path] = {root.resolve()}
    runtime_modules: set[str] = set()
    for layer in layers:
        real_path = str(getattr(layer, "realPath", "") or "")
        identifier = str(getattr(layer, "identifier", "") or "")
        try:
            authored_paths, composition_paths = collect_layer_authored_asset_paths(
                layer,
                sdf=Sdf,
            )
        except (AttributeError, ValueError) as exc:
            raise _ValidationArtifactFingerprintError(
                "Could not classify authored USD dependencies: "
                f"{identifier or real_path}"
            ) from exc
        runtime_modules.update(
            path
            for path in authored_paths - composition_paths
            if path in unresolved_paths and path in _PUBLIC_RUNTIME_MDL_MODULES
        )
        dependency = _usd_local_dependency_path(
            real_path or identifier,
            root=root,
            package_utils=Ar,
        )
        if dependency is not None:
            dependencies.add(dependency)
            continue
        raise _ValidationArtifactFingerprintError(
            f"USD input layer dependency is not a local file: {identifier or real_path}"
        )
    for asset in assets:
        resolved_path = str(getattr(asset, "resolvedPath", "") or "")
        authored_path = str(getattr(asset, "path", "") or str(asset))
        if authored_path in _PUBLIC_RUNTIME_MDL_MODULES:
            runtime_modules.add(authored_path)
            continue
        dependency = _usd_local_dependency_path(
            resolved_path or authored_path,
            root=root,
            package_utils=Ar,
        )
        if dependency is not None:
            dependencies.add(dependency)
            continue
        raise _ValidationArtifactFingerprintError(
            f"USD input dependency is not a local file: {authored_path}"
        )
    unsupported_unresolved = tuple(
        path for path in unresolved_paths if path not in runtime_modules
    )
    if unsupported_unresolved:
        raise _ValidationArtifactFingerprintError(
            "USD input dependency closure is unresolved: "
            + ", ".join(unsupported_unresolved)
        )

    ordered = [root.resolve()]
    ordered.extend(
        sorted(dependencies - {root.resolve()}, key=lambda path: path.as_posix())
    )
    return tuple(ordered), tuple(sorted(runtime_modules))


def _usd_local_dependency_path(
    value: str,
    *,
    root: Path,
    package_utils: Any,
) -> Path | None:
    candidate_value = value
    if package_utils.IsPackageRelativePath(candidate_value):
        candidate_value, _ = package_utils.SplitPackageRelativePathOuter(
            candidate_value
        )
    candidate = Path(candidate_value)
    if not candidate.is_absolute():
        candidate = root.parent / candidate
    if not candidate.is_file():
        return None
    return candidate.resolve()


def _best_effort_artifact_identities(
    paths: Sequence[str | Path],
    *,
    role: ArtifactRole,
    base_dir: Path,
) -> tuple[ValidationArtifactIdentity, ...]:
    identities: list[ValidationArtifactIdentity] = []
    seen: set[str] = set()
    for path in paths:
        try:
            candidates = _artifact_identities(
                (path,),
                role=role,
                base_dir=base_dir,
            )
        except (OSError, RuntimeError, ValueError, ValidationWorkflowError):
            candidates = (
                ValidationArtifactIdentity(
                    role=role,
                    path=_best_effort_resolved_path(path, base_dir=base_dir),
                    kind="missing",
                ),
            )
        for candidate in candidates:
            if candidate.path in seen:
                continue
            identities.append(candidate)
            seen.add(candidate.path)
    return tuple(identities)


def _reference_paths(policy: Mapping[str, Any]) -> tuple[str, ...]:
    value = policy.get("reference_image_paths")
    if value is None:
        return ()
    if isinstance(value, str | Path):
        return (str(value),)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(str(path) for path in value)
    raise ValidationWorkflowError(
        "policy.reference_image_paths must be a path or sequence of paths"
    )


def _selected_templates(request: ValidationRequest) -> tuple[str, ...]:
    requested = request.requested_templates or VISUAL_TEMPLATE_ORDER
    unknown = sorted(set(requested) - set(VISUAL_TEMPLATE_ORDER))
    if unknown:
        raise ValidationWorkflowError(
            "Wave 2 supports only render_valid and look_right; unsupported "
            f"templates: {', '.join(unknown)}"
        )
    selected = set(requested)
    if "look_right" in selected:
        selected.add("render_valid")
    return tuple(name for name in VISUAL_TEMPLATE_ORDER if name in selected)


def _effective_request(
    request: ValidationRequest,
    *,
    output_dir: Path,
) -> ValidationRequest:
    isolated_request = request.model_copy(deep=True)
    requested_templates = _selected_templates(isolated_request)
    policy = dict(isolated_request.policy)
    evidence_mode = policy.get("visual_evidence_mode")
    if evidence_mode is None:
        policy["visual_evidence_mode"] = "canonical_usd"
    elif str(evidence_mode).strip().lower() not in _CANONICAL_VISUAL_EVIDENCE_MODES:
        raise ValidationWorkflowError(
            "Wave 2 requires canonical USD visual evidence so look_right can "
            "consume only the current accepted render_valid evidence."
        )
    project = isolated_request.project.model_copy(
        update={"working_dir": str(output_dir)}
    )
    return isolated_request.model_copy(
        update={
            "project": project,
            "requested_templates": requested_templates,
            "policy": policy,
        }
    )


def _workflow_identity(
    request: ValidationRequest,
    *,
    config_base_dir: Path,
    executor: ValidationStepExecutor,
) -> ValidationWorkflowIdentity:
    resolved_policy = scaffold_policy_from_request(
        request,
        base_dir=config_base_dir,
    )
    sources = _artifact_identities(
        request.inputs,
        role="source",
        base_dir=config_base_dir,
    )
    references = (
        _artifact_identities(
            _reference_paths(resolved_policy),
            role="reference",
            base_dir=config_base_dir,
        )
        if "look_right" in request.requested_templates
        else ()
    )
    template_versions: dict[str, str] = {}
    for template_name in request.requested_templates:
        version = executor.template_versions.get(template_name)
        if not version:
            raise ValidationWorkflowError(
                f"Executor did not provide a version for {template_name}"
            )
        template_versions[template_name] = version

    request_digest = _canonical_digest(request.model_dump(mode="json"))
    policy_digest = _canonical_digest(resolved_policy)
    backend_digest = _canonical_digest(
        {
            "render": request.render.model_dump(mode="json"),
            "render_backend": resolved_policy.get("render_backend"),
            "look_right_vlm": resolved_policy.get("look_right_vlm"),
            "look_right_llm_judge": resolved_policy.get("look_right_llm_judge"),
        }
    )
    return ValidationWorkflowIdentity(
        request_digest=request_digest,
        policy_digest=policy_digest,
        backend_digest=backend_digest,
        source_artifacts=sources,
        reference_artifacts=references,
        template_versions=template_versions,
        identity_digest=validation_workflow_identity_digest(
            schema_version="content-agent-workflows.validation-identity.v1",
            request_digest=request_digest,
            policy_digest=policy_digest,
            backend_digest=backend_digest,
            source_artifacts=sources,
            reference_artifacts=references,
            template_versions=template_versions,
        ),
    )


def _validate_output_location(
    output_dir: Path,
    identity: ValidationWorkflowIdentity,
) -> None:
    for artifact in (
        *identity.source_artifacts,
        *identity.reference_artifacts,
    ):
        if artifact.kind != "directory":
            continue
        input_dir = Path(artifact.path)
        if output_dir == input_dir or input_dir in output_dir.parents:
            raise ValidationWorkflowError(
                f"Validation output_dir cannot be inside a {artifact.role} "
                "directory because workflow artifacts would change its identity: "
                f"{input_dir}"
            )


def _validate_output_artifact_locations(paths: ValidationWorkflowPaths) -> None:
    candidates = {
        **{
            name: Path(path)
            for name, path in paths.artifact_paths().items()
            if name != "validation_checkpoint"
        },
        "attempts": paths.attempts,
        "validation_run_owner": paths.output_dir / _RUN_OWNER_FILE_NAME,
    }
    if (
        paths.checkpoint == paths.output_dir
        or paths.output_dir in paths.checkpoint.parents
    ):
        candidates["validation_checkpoint"] = paths.checkpoint
        candidates["validation_checkpoint_lock"] = paths.checkpoint.with_suffix(
            f"{paths.checkpoint.suffix}.lock"
        )
    for name, candidate in candidates.items():
        try:
            relative = candidate.relative_to(paths.output_dir)
        except ValueError as exc:
            raise ValidationWorkflowError(
                f"Canonical {name} workflow artifact is outside output_dir: {candidate}"
            ) from exc
        current = paths.output_dir
        for part in relative.parts:
            current /= part
            if current.is_symlink():
                raise ValidationWorkflowError(
                    f"Canonical {name} workflow artifact cannot use a symlink: "
                    f"{current}"
                )
        expected_directory = name == "attempts"
        matches_expected_kind = (
            candidate.is_dir() if expected_directory else candidate.is_file()
        )
        if candidate.exists() and not matches_expected_kind:
            expected_kind = "directory" if expected_directory else "file"
            raise ValidationWorkflowError(
                f"Canonical {name} workflow artifact must be a {expected_kind}: "
                f"{candidate}"
            )
        resolved = candidate.resolve()
        if resolved != paths.output_dir and paths.output_dir not in resolved.parents:
            raise ValidationWorkflowError(
                f"Canonical {name} workflow artifact resolves outside output_dir: "
                f"{resolved}"
            )


def _validate_attempt_artifact_location(
    path: Path,
    *,
    paths: ValidationWorkflowPaths,
) -> None:
    try:
        relative = path.relative_to(paths.attempts)
    except ValueError as exc:
        raise ValidationWorkflowError(
            f"Validation attempt artifact is outside the attempts directory: {path}"
        ) from exc
    current = paths.attempts
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValidationWorkflowError(
                f"Validation attempt artifact cannot use a symlink: {current}"
            )
    resolved = path.resolve()
    if resolved != paths.attempts and paths.attempts not in resolved.parents:
        raise ValidationWorkflowError(
            "Validation attempt artifact resolves outside the attempts directory: "
            f"{resolved}"
        )


@contextmanager
def _pinned_directory_fd(
    path: Path,
    *,
    expected_identity: tuple[int, int],
    description: str,
    cleanup_names_on_error: Sequence[str] = (),
) -> Iterator[int]:
    """Keep directory-relative operations bound to one checked inode."""

    if os.name != "posix":  # pragma: win32 cover
        raise ValidationWorkflowError(
            f"Validation {description} pinning requires POSIX."
        )
    try:
        directory_fd, directory_chain = _open_directory_no_symlinks(path)
    except OSError as exc:
        raise ValidationWorkflowError(
            f"Validation {description} could not be opened safely: {path}"
        ) from exc
    try:
        fd_stat = os.fstat(directory_fd)
        try:
            path_stat = path.stat()
        except OSError as exc:
            raise ValidationWorkflowError(
                f"Validation {description} identity changed: {path}"
            ) from exc
        if (
            (fd_stat.st_dev, fd_stat.st_ino) != expected_identity
            or (path_stat.st_dev, path_stat.st_ino) != expected_identity
            or not _directory_chain_matches(path, directory_chain)
        ):
            raise ValidationWorkflowError(
                f"Validation {description} identity changed: {path}"
            )
        try:
            yield directory_fd
            try:
                final_path_stat = path.stat()
            except OSError as exc:
                raise ValidationWorkflowError(
                    f"Validation {description} identity changed: {path}"
                ) from exc
            if (
                final_path_stat.st_dev,
                final_path_stat.st_ino,
            ) != expected_identity or not _directory_chain_matches(
                path, directory_chain
            ):
                raise ValidationWorkflowError(
                    f"Validation {description} identity changed: {path}"
                )
        except BaseException:
            for cleanup_name in cleanup_names_on_error:
                with suppress(OSError):
                    os.unlink(cleanup_name, dir_fd=directory_fd)
            raise
    finally:
        os.close(directory_fd)


def _invalidate_terminal_bundle(
    paths: ValidationWorkflowPaths,
    *,
    output_dir_identity: tuple[int, int],
) -> None:
    """Remove a prior terminal bundle before resumed work can execute."""

    terminal_artifacts = (
        paths.result,
        paths.evidence,
        paths.final_summary,
    )
    if os.name == "posix":
        with _pinned_directory_fd(
            paths.output_dir,
            expected_identity=output_dir_identity,
            description="output directory",
        ) as output_dir_fd:
            for artifact in terminal_artifacts:
                try:
                    os.unlink(artifact.name, dir_fd=output_dir_fd)
                except FileNotFoundError:
                    pass
        return
    for artifact in terminal_artifacts:  # pragma: win32 cover
        try:
            artifact.unlink()
        except FileNotFoundError:
            pass


@contextmanager
def _pinned_attempt_working_directory(
    path: Path,
    *,
    paths: ValidationWorkflowPaths,
    expected_identity: tuple[int, int],
) -> Iterator[Path]:
    """Keep executor writes bound to the checked attempt directory inode."""

    _validate_attempt_artifact_location(path, paths=paths)
    if os.name != "posix":  # pragma: win32 cover
        yield path
        return
    with _pinned_directory_fd(
        path,
        expected_identity=expected_identity,
        description="attempt directory",
    ) as directory_fd:
        descriptor_path = Path(f"/proc/self/fd/{directory_fd}")
        if not descriptor_path.is_dir():
            raise ValidationWorkflowError(
                "Validation attempt execution requires a descriptor-backed "
                "working directory on this platform."
            )
        yield descriptor_path


def _validate_source_artifacts(
    identity: ValidationWorkflowIdentity,
    *,
    resume_checkpoint_exists: bool,
) -> None:
    missing = tuple(
        artifact.path
        for artifact in identity.source_artifacts
        if artifact.kind == "missing"
    )
    if not missing:
        return
    message = "Validation source input artifacts do not exist: " + ", ".join(missing)
    if resume_checkpoint_exists:
        raise ValidationWorkflowIdentityMismatch(message)
    raise ValidationWorkflowError(message)


def _validate_checkpoint_location(
    checkpoint_path: Path,
    identity: ValidationWorkflowIdentity,
) -> None:
    for artifact in (
        *identity.source_artifacts,
        *identity.reference_artifacts,
    ):
        if artifact.kind != "directory":
            continue
        input_dir = Path(artifact.path)
        if checkpoint_path == input_dir or input_dir in checkpoint_path.parents:
            raise ValidationWorkflowError(
                f"Validation checkpoint_store.path cannot be inside a {artifact.role} "
                "directory because checkpoint writes would change its identity: "
                f"{input_dir}"
            )


def _validate_checkpoint_artifact_collision(
    paths: ValidationWorkflowPaths,
) -> None:
    checkpoint_lock = paths.checkpoint.with_suffix(f"{paths.checkpoint.suffix}.lock")
    try:
        if checkpoint_lock.is_file() and checkpoint_lock.stat().st_nlink > 1:
            raise ValidationWorkflowError(
                "Validation checkpoint_store lock path cannot be a hard link "
                f"because lock identity must be single-link: {checkpoint_lock}"
            )
    except OSError as exc:
        raise ValidationWorkflowError(
            "Validation checkpoint_store lock path could not be inspected safely: "
            f"{checkpoint_lock}"
        ) from exc
    if (
        checkpoint_lock == paths.output_dir
        or checkpoint_lock in paths.output_dir.parents
    ):
        raise ValidationWorkflowError(
            "Validation checkpoint_store lock path cannot contain the canonical "
            f"output_dir workflow artifact: {paths.output_dir}"
        )
    if paths.checkpoint == paths.attempts or paths.attempts in paths.checkpoint.parents:
        raise ValidationWorkflowError(
            "Validation checkpoint_store.path cannot alias or be inside the "
            f"canonical attempts workflow artifact directory: {paths.attempts}"
        )
    all_canonical_artifacts = {
        name: Path(path).expanduser().resolve()
        for name, path in paths.artifact_paths().items()
    }
    all_canonical_artifacts["validation_run_owner"] = (
        paths.output_dir / _RUN_OWNER_FILE_NAME
    )
    all_canonical_artifacts["output_dir"] = paths.output_dir
    for name, artifact_path in all_canonical_artifacts.items():
        aliases_existing_artifact = False
        try:
            if checkpoint_lock.exists() and artifact_path.exists():
                aliases_existing_artifact = checkpoint_lock.samefile(artifact_path)
                if artifact_path.is_dir() and not aliases_existing_artifact:
                    aliases_existing_artifact = any(
                        member != checkpoint_lock
                        and member.is_file()
                        and checkpoint_lock.samefile(member)
                        for member in artifact_path.rglob("*")
                    )
        except OSError:
            pass
        if checkpoint_lock == artifact_path or aliases_existing_artifact:
            raise ValidationWorkflowError(
                "Validation checkpoint_store lock path cannot alias the canonical "
                f"{name} workflow artifact: {artifact_path}"
            )
    canonical_artifacts = {
        name: Path(path).expanduser().resolve()
        for name, path in paths.artifact_paths().items()
        if name != "validation_checkpoint"
    }
    canonical_artifacts["validation_run_owner"] = (
        paths.output_dir / _RUN_OWNER_FILE_NAME
    )
    canonical_artifacts["output_dir"] = paths.output_dir
    for name, artifact_path in canonical_artifacts.items():
        aliases_artifact = paths.checkpoint == artifact_path
        descends_from_file_artifact = (
            name != "output_dir" and artifact_path in paths.checkpoint.parents
        )
        contains_output_dir = (
            name == "output_dir" and paths.checkpoint in artifact_path.parents
        )
        if contains_output_dir:
            raise ValidationWorkflowError(
                "Validation checkpoint_store.path cannot contain the canonical "
                f"output_dir workflow artifact: {artifact_path}"
            )
        if aliases_artifact or descends_from_file_artifact:
            raise ValidationWorkflowError(
                "Validation checkpoint_store.path cannot alias or be inside the "
                "canonical "
                f"{name} workflow artifact: {artifact_path}"
            )


def _validate_input_artifact_collisions(
    paths: ValidationWorkflowPaths,
    identity: ValidationWorkflowIdentity,
) -> None:
    canonical_artifacts = {
        name: Path(path).expanduser().resolve()
        for name, path in paths.artifact_paths().items()
    }
    canonical_artifacts["validation_checkpoint_lock"] = paths.checkpoint.with_suffix(
        f"{paths.checkpoint.suffix}.lock"
    )
    canonical_artifacts["validation_run_owner"] = (
        paths.output_dir / _RUN_OWNER_FILE_NAME
    )
    canonical_artifacts["output_dir"] = paths.output_dir
    for artifact in (
        *identity.source_artifacts,
        *identity.reference_artifacts,
    ):
        artifact_path = Path(artifact.path)
        if artifact_path == paths.attempts or paths.attempts in artifact_path.parents:
            raise ValidationWorkflowError(
                f"Validation {artifact.role} input artifact cannot alias or be "
                "inside the canonical attempts workflow artifact directory: "
                f"{artifact_path}"
            )
        for name, output_path in canonical_artifacts.items():
            aliases_existing_artifact = False
            try:
                if artifact_path.exists() and output_path.exists():
                    aliases_existing_artifact = artifact_path.samefile(output_path)
                    if artifact.kind == "directory" and not aliases_existing_artifact:
                        aliases_existing_artifact = any(
                            member.is_file() and member.samefile(output_path)
                            for member in artifact_path.rglob("*")
                        )
            except OSError:
                pass
            if artifact_path == output_path or aliases_existing_artifact:
                raise ValidationWorkflowError(
                    f"Validation {artifact.role} input artifact cannot alias the "
                    f"canonical {name} workflow artifact: {output_path}"
                )


def _claim_output_directory(
    paths: ValidationWorkflowPaths,
) -> None:
    owner_path = paths.output_dir / _RUN_OWNER_FILE_NAME
    expected = {
        "schema_version": "content-agent-workflows.validation-run-owner.v1",
        "checkpoint_path": str(paths.checkpoint),
    }
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        owner_fd = os.open(
            owner_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
    except FileExistsError:
        owner_fd = -1
        try:
            owner_fd = os.open(
                owner_path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_NONBLOCK", 0),
            )
            opened_stat = os.fstat(owner_fd)
            if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                raise ValidationWorkflowError(
                    "Validation output_dir is owned by a different checkpoint "
                    f"store or has an invalid owner marker: {owner_path}"
                )
            with os.fdopen(owner_fd, mode="r", encoding="utf-8") as stream:
                owner_fd = -1
                actual = json.load(stream)
                final_stat = os.fstat(stream.fileno())
            path_stat = owner_path.lstat()
            opened_identity = (opened_stat.st_dev, opened_stat.st_ino)
            if (
                final_stat.st_nlink != 1
                or not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_nlink != 1
                or (path_stat.st_dev, path_stat.st_ino) != opened_identity
                or actual != expected
            ):
                raise ValidationWorkflowError(
                    "Validation output_dir is owned by a different checkpoint "
                    f"store or has an invalid owner marker: {owner_path}"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            if isinstance(exc, ValidationWorkflowError):
                raise
            raise ValidationWorkflowError(
                f"Validation output_dir owner marker is invalid: {owner_path}"
            ) from exc
        finally:
            if owner_fd >= 0:
                os.close(owner_fd)
        return
    except OSError as exc:
        raise ValidationWorkflowError(
            f"Validation output_dir owner marker could not be created: {owner_path}"
        ) from exc
    try:
        with os.fdopen(owner_fd, mode="w", encoding="utf-8") as stream:
            owner_fd = -1
            json.dump(expected, stream, indent=2, sort_keys=True, ensure_ascii=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if owner_fd >= 0:
            os.close(owner_fd)


def _work_item_identity(
    workflow_identity: ValidationWorkflowIdentity,
    *,
    template_name: str,
    depends_on: tuple[str, ...],
) -> str:
    return _canonical_digest(
        {
            "workflow_identity": workflow_identity.identity_digest,
            "template_name": template_name,
            "template_version": workflow_identity.template_versions[template_name],
            "depends_on": depends_on,
        }
    )


def _bind_plan(
    plan: ValidationPlan,
    *,
    identity: ValidationWorkflowIdentity,
    artifact_paths: Mapping[str, str],
) -> ValidationPlan:
    plan = plan.model_copy(deep=True)
    expected_names = tuple(identity.template_versions)
    actual_names = tuple(step.template_name for step in plan.steps)
    if actual_names != expected_names:
        raise ValidationWorkflowError(
            "Validation planner did not preserve the required ordered work "
            f"items: expected {expected_names}, got {actual_names}"
        )

    bound_steps = []
    for step in plan.steps:
        work_item_id = f"validation:{step.template_name}"
        depends_on = (
            ("validation:render_valid",)
            if step.template_name == "look_right"
            and any(item.template_name == "render_valid" for item in plan.steps)
            else ()
        )
        work_item_identity = _work_item_identity(
            identity,
            template_name=step.template_name,
            depends_on=depends_on,
        )
        metadata = dict(step.metadata)
        metadata["agentic_work_item"] = {
            "schema_version": VALIDATION_WORK_ITEM_SCHEMA_VERSION,
            "work_item_id": work_item_id,
            "identity_digest": work_item_identity,
            "template_version": identity.template_versions[step.template_name],
            "depends_on": list(depends_on),
            "request_digest": identity.request_digest,
            "source_artifacts": [
                artifact.model_dump(mode="json")
                for artifact in identity.source_artifacts
            ],
            "reference_artifacts": [
                artifact.model_dump(mode="json")
                for artifact in identity.reference_artifacts
            ],
            "policy_digest": identity.policy_digest,
            "backend_digest": identity.backend_digest,
        }
        try:
            required_capabilities = VALIDATION_TEMPLATE_CAPABILITIES[step.template_name]
        except KeyError as exc:
            raise ValidationWorkflowError(
                "Validation template has no declared capability mapping: "
                f"{step.template_name}"
            ) from exc
        bound_steps.append(
            step.model_copy(
                update={
                    "required_capabilities": required_capabilities,
                    "metadata": metadata,
                }
            )
        )

    metadata = dict(plan.metadata)
    metadata["agentic_workflow"] = {
        "schema_version": "content-agent-workflows.validation-plan.v1",
        "identity": identity.model_dump(mode="json"),
    }
    return plan.model_copy(
        update={
            "steps": tuple(bound_steps),
            "artifact_paths": dict(artifact_paths),
            "metadata": metadata,
        }
    )


def _plan_digest(plan: ValidationPlan) -> str:
    return _canonical_digest(plan.model_dump(mode="json"))


def _records_from_plan(plan: ValidationPlan) -> tuple[ValidationWorkItemRecord, ...]:
    records: list[ValidationWorkItemRecord] = []
    for step in plan.steps:
        work_item = step.metadata.get("agentic_work_item")
        if not isinstance(work_item, Mapping):
            raise ValidationWorkflowError(
                f"Plan step {step.template_name} has no bound work item metadata"
            )
        records.append(
            ValidationWorkItemRecord(
                work_item_id=str(work_item["work_item_id"]),
                template_name=step.template_name,
                identity_digest=str(work_item["identity_digest"]),
            )
        )
    return tuple(records)


def _accepted_result_is_valid(
    accepted: ValidationAcceptedTemplateResult,
    *,
    paths: ValidationWorkflowPaths,
    expected_template_name: str,
    expected_attempt: int,
    forbidden_artifacts: Sequence[ValidationArtifactIdentity],
) -> bool:
    try:
        if (
            accepted.result.template_name != expected_template_name
            or accepted.attempt != expected_attempt
        ):
            return False
        candidate_result_path = Path(accepted.result_path).expanduser()
        if not candidate_result_path.is_absolute():
            return False
        result_path = Path(os.path.abspath(candidate_result_path))
        expected_result_path = (
            paths.attempts
            / expected_template_name
            / f"attempt-{expected_attempt:04d}"
            / "template_result.json"
        )
        _validate_attempt_artifact_location(result_path, paths=paths)
        if result_path != expected_result_path or not result_path.is_file():
            return False
        if file_sha256(result_path) != accepted.result_sha256:
            return False
        persisted_result = ValidationTemplateResult.model_validate(
            load_json(result_path)
        )
        if persisted_result != accepted.result:
            return False
        declared_evidence, missing_evidence = _evidence_artifacts(
            persisted_result,
            base_dir=result_path.parent,
        )
        has_terminal_missing_evidence = persisted_result.status == "failed" and any(
            issue.code == "validation.accepted_evidence_missing"
            for issue in persisted_result.issues
        )
        if missing_evidence and not has_terminal_missing_evidence:
            return False
        declared_evidence, workflow_owned_evidence = _workflow_owned_evidence_artifacts(
            declared_evidence,
            paths=paths,
            result_path=result_path,
        )
        has_terminal_workflow_owned_evidence = (
            persisted_result.status == "failed"
            and any(
                issue.code == "validation.workflow_artifact_evidence_collision"
                for issue in persisted_result.issues
            )
        )
        if workflow_owned_evidence and not has_terminal_workflow_owned_evidence:
            return False
        if _normalized_artifact_identities(
            declared_evidence
        ) != _normalized_artifact_identities(accepted.evidence_artifacts):
            return False
        for artifact in accepted.evidence_artifacts:
            current = _artifact_identity(
                artifact.path,
                role="evidence",
                base_dir=result_path.parent,
            )
            if current != artifact:
                return False
        if (
            accepted.result.template_name == "render_valid"
            and _input_artifact_collision_paths(
                accepted.evidence_artifacts,
                forbidden_artifacts=forbidden_artifacts,
            )
        ):
            return False
        if accepted.result.template_name == "render_valid" and accepted.result.passed:
            render_origin = accepted.result.metadata.get("render_evidence_origin")
            render_paths = (
                _reported_render_image_paths(accepted.result)
                if render_origin == "qualified_precomputed"
                else _runtime_render_image_paths(accepted.result)
            )
            expected_render_paths = {
                str(_resolve_path(path, base_dir=result_path.parent))
                for path in render_paths
            }
            accepted_evidence_paths = {
                artifact.path
                for artifact in accepted.evidence_artifacts
                if artifact.kind == "file"
            }
            if not expected_render_paths or not expected_render_paths.issubset(
                accepted_evidence_paths
            ):
                return False
    except (OSError, RuntimeError, ValueError, ValidationWorkflowError):
        return False
    return True


def _final_integrity_issues(
    checkpoint: ValidationWorkflowCheckpoint,
    *,
    paths: ValidationWorkflowPaths,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    reference_before: tuple[ValidationArtifactIdentity, ...],
    reference_after: tuple[ValidationArtifactIdentity, ...],
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    if source_before != source_after:
        issues.append(
            ValidationIssue(
                code="validation.source_asset_modified",
                severity="fail",
                message=(
                    "A source asset changed before final validation artifacts "
                    "were published."
                ),
                details={
                    "source_before": [
                        identity.model_dump(mode="json") for identity in source_before
                    ],
                    "source_after": [
                        identity.model_dump(mode="json") for identity in source_after
                    ],
                },
            )
        )
    validates_reference = any(
        record.template_name == "look_right" for record in checkpoint.records
    )
    if validates_reference and reference_before != reference_after:
        issues.append(
            ValidationIssue(
                code="validation.reference_evidence_stale",
                severity="fail",
                message=(
                    "Reference evidence changed before final validation artifacts "
                    "were published."
                ),
                template_name="look_right",
                details={
                    "reference_before": [
                        identity.model_dump(mode="json")
                        for identity in reference_before
                    ],
                    "reference_after": [
                        identity.model_dump(mode="json") for identity in reference_after
                    ],
                },
            )
        )
    for record in checkpoint.records:
        accepted = record.accepted_result
        if accepted is None or _accepted_result_is_valid(
            accepted,
            paths=paths,
            expected_template_name=record.template_name,
            expected_attempt=record.attempts,
            forbidden_artifacts=(
                *checkpoint.workflow_identity.source_artifacts,
                *checkpoint.workflow_identity.reference_artifacts,
            ),
        ):
            continue
        issues.append(
            ValidationIssue(
                code="validation.accepted_evidence_stale",
                severity="fail",
                message=(
                    f"Accepted {record.template_name} result or evidence changed "
                    "before final validation artifacts were published."
                ),
                template_name=record.template_name,
                details={
                    "work_item_id": record.work_item_id,
                    "result_path": accepted.result_path,
                },
            )
        )
    return tuple(issues)


def _prepare_resume_checkpoint(
    store: ValidationCheckpointStore,
    *,
    paths: ValidationWorkflowPaths,
    identity: ValidationWorkflowIdentity,
    plan_digest: str,
    expected_records: tuple[ValidationWorkItemRecord, ...],
    invalidate_terminal_bundle: Callable[[], None],
) -> ValidationWorkflowCheckpoint:
    expected_ids = tuple(record.work_item_id for record in expected_records)

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        if current.workflow_identity != identity:
            raise ValidationWorkflowIdentityMismatch(
                "Validation checkpoint identity changed. Prompt, source, "
                "reference, policy, backend, or template version differs."
            )
        if current.plan_digest != plan_digest:
            raise ValidationWorkflowIdentityMismatch(
                "Validation checkpoint plan digest differs from the current plan."
            )
        if current.ordered_work_item_ids != expected_ids:
            raise ValidationWorkflowIdentityMismatch(
                "Validation checkpoint work items differ from the current plan."
            )
        active_records = tuple(
            record
            for record in current.records
            if record.state == ValidationWorkItemState.RUNNING
            and record.accepted_result is None
        )
        if active_records:
            active_names = ", ".join(record.template_name for record in active_records)
            raise ValidationCheckpointError(
                "Cannot resume while another runner has active validation work: "
                f"{active_names}. Persist cancellation before recovering an "
                "orphaned claim."
            )
        expected_by_id = {record.work_item_id: record for record in expected_records}
        invalidate_downstream = False
        records: list[ValidationWorkItemRecord] = []
        for record in current.records:
            expected = expected_by_id[record.work_item_id]
            if (
                record.template_name != expected.template_name
                or record.identity_digest != expected.identity_digest
            ):
                raise ValidationWorkflowIdentityMismatch(
                    f"Validation work item identity changed: {record.work_item_id}"
                )
            accepted = record.accepted_result
            if (
                invalidate_downstream
                or accepted is None
                or not _accepted_result_is_valid(
                    accepted,
                    paths=paths,
                    expected_template_name=record.template_name,
                    expected_attempt=record.attempts,
                    forbidden_artifacts=(
                        *current.workflow_identity.source_artifacts,
                        *current.workflow_identity.reference_artifacts,
                    ),
                )
            ):
                invalidate_downstream = True
                records.append(
                    record.model_copy(
                        update={
                            "state": ValidationWorkItemState.PENDING,
                            "accepted_result": None,
                            "last_error": (
                                "Accepted result or evidence was missing or stale; "
                                "this work item must run again."
                                if accepted is not None
                                else None
                            ),
                            "started_at": None,
                            "finished_at": None,
                        }
                    )
                )
                continue
            records.append(
                record.model_copy(
                    update={
                        "state": ValidationWorkItemState.COMPLETED,
                        "last_error": None,
                    }
                )
            )
        resumed = current.model_copy(
            update={
                "records": tuple(records),
                "cancellation_requested": False,
                "cancellation_reason": None,
            }
        )
        if resumed == current:
            # A resume that changes nothing must not advance the revision.
            # `update()` only skips the bump when the mutation hands back the
            # very object it was given, and a published terminal bundle still
            # describes this revision faithfully.
            return current
        # Every resume that does advance the revision strands the published
        # bundle one revision behind. `validation_result.json` is the terminal
        # commit marker, so drop the bundle while still holding the checkpoint
        # lock and before the new revision reaches disk; otherwise a later
        # failure leaves a stale success advertised for a revision that no
        # longer exists.
        invalidate_terminal_bundle()
        return resumed

    return store.update(mutation)


def _checkpoint_record(
    checkpoint: ValidationWorkflowCheckpoint,
    work_item_id: str,
) -> ValidationWorkItemRecord:
    return next(
        record for record in checkpoint.records if record.work_item_id == work_item_id
    )


def _replace_record(
    checkpoint: ValidationWorkflowCheckpoint,
    updated: ValidationWorkItemRecord,
    *,
    cancellation_requested: bool | None = None,
    cancellation_reason: str | None = None,
) -> ValidationWorkflowCheckpoint:
    update: dict[str, Any] = {
        "records": tuple(
            updated if record.work_item_id == updated.work_item_id else record
            for record in checkpoint.records
        )
    }
    if cancellation_requested is not None:
        update["cancellation_requested"] = cancellation_requested
        update["cancellation_reason"] = cancellation_reason
    return checkpoint.model_copy(update=update)


def _report(
    checkpoint: ValidationWorkflowCheckpoint,
    callback: ProgressCallback | None,
) -> None:
    if callback is not None:
        callback(checkpoint)


def _cancellation_requested(
    token: ValidationCancellationToken,
    external_check: ExternalCancellationCheck | None,
) -> bool:
    if token.is_cancelled():
        return True
    if external_check is not None and external_check():
        token.cancel()
        return True
    return False


def _mark_running(
    store: ValidationCheckpointStore,
    work_item_id: str,
) -> tuple[ValidationWorkflowCheckpoint, bool]:
    claimed = False

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        nonlocal claimed
        if current.cancellation_requested:
            return current
        record = _checkpoint_record(current, work_item_id)
        if record.accepted_result is not None:
            return current
        if record.state != ValidationWorkItemState.PENDING:
            raise ValidationCheckpointError(
                f"Cannot start {work_item_id} from state {record.state.value}"
            )
        claimed = True
        updated = record.model_copy(
            update={
                "state": ValidationWorkItemState.RUNNING,
                "attempts": record.attempts + 1,
                "accepted_result": None,
                "last_error": None,
                "started_at": datetime.now(UTC),
                "finished_at": None,
            }
        )
        return _replace_record(current, updated)

    return store.update(mutation), claimed


def _release_unstarted_claim(
    store: ValidationCheckpointStore,
    *,
    record: ValidationWorkItemRecord,
) -> ValidationWorkflowCheckpoint:
    """Release a claim when progress reporting aborts before execution."""

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        latest = _checkpoint_record(current, record.work_item_id)
        if (
            latest.attempts != record.attempts
            or latest.state != ValidationWorkItemState.RUNNING
            or latest.accepted_result is not None
        ):
            return current
        return _replace_record(
            current,
            latest.model_copy(
                update={
                    "state": ValidationWorkItemState.PENDING,
                    "accepted_result": None,
                    "last_error": (
                        "Progress reporting aborted before template execution."
                    ),
                    "started_at": None,
                    "finished_at": None,
                }
            ),
        )

    return store.update(mutation)


def _cancel_remaining(
    store: ValidationCheckpointStore,
    *,
    reason: str,
    recover_orphaned_claims: bool = False,
) -> ValidationWorkflowCheckpoint:
    ensure_no_inline_secrets(
        {"reason": reason},
        context="validation workflow cancellation reason",
    )

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        if all(record.accepted_result is not None for record in current.records):
            return current
        now = datetime.now(UTC)
        records = tuple(
            record
            if record.accepted_result is not None
            else record.model_copy(
                update={
                    "last_error": reason,
                    "finished_at": None,
                }
            )
            if (
                record.state == ValidationWorkItemState.RUNNING
                and not recover_orphaned_claims
            )
            else record.model_copy(
                update={
                    "state": ValidationWorkItemState.CANCELLED,
                    "last_error": reason,
                    "finished_at": now,
                }
            )
            for record in current.records
        )
        return current.model_copy(
            update={
                "records": records,
                "cancellation_requested": True,
                "cancellation_reason": reason,
            }
        )

    return store.update(mutation)


def request_validation_cancellation(
    output_dir: str | Path,
    *,
    reason: str = "Validation workflow cancellation requested.",
    checkpoint_store: ValidationCheckpointStore | None = None,
) -> ValidationWorkflowCheckpoint:
    """Persist cancellation while retaining every live attempt fence."""

    paths = ValidationWorkflowPaths.from_output_dir(output_dir)
    store = checkpoint_store or FileValidationCheckpointStore(paths.checkpoint)
    return _cancel_remaining(
        store,
        reason=reason,
    )


def recover_orphaned_validation_claims(
    output_dir: str | Path,
    *,
    reason: str = "Recover explicitly orphaned validation claims.",
    checkpoint_store: ValidationCheckpointStore | None = None,
) -> ValidationWorkflowCheckpoint:
    """Release cancelled RUNNING claims after their executors are confirmed gone."""

    paths = ValidationWorkflowPaths.from_output_dir(output_dir)
    store = checkpoint_store or FileValidationCheckpointStore(paths.checkpoint)
    return _cancel_remaining(
        store,
        reason=reason,
        recover_orphaned_claims=True,
    )


def _acknowledge_cancelled_attempt(
    store: ValidationCheckpointStore,
    *,
    record: ValidationWorkItemRecord,
    reason: str,
) -> ValidationWorkflowCheckpoint:
    """Release one live attempt fence after its owning runner stops."""

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        latest = _checkpoint_record(current, record.work_item_id)
        if latest.accepted_result is not None:
            return current
        if latest.attempts != record.attempts:
            raise ValidationCheckpointError(
                f"Cannot release stale attempt of {record.work_item_id}"
            )
        if latest.state != ValidationWorkItemState.RUNNING:
            return current
        updated = latest.model_copy(
            update={
                "state": ValidationWorkItemState.CANCELLED,
                "accepted_result": None,
                "last_error": reason,
                "finished_at": datetime.now(UTC),
            }
        )
        return _replace_record(
            current,
            updated,
            cancellation_requested=True,
            cancellation_reason=current.cancellation_reason or reason,
        )

    return store.update(mutation)


def _template_execution_error(
    template_name: str,
    exc: Exception,
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.template_execution_error",
        severity="fail",
        message=(
            f"{template_name} could not complete because its executor raised "
            f"{type(exc).__name__}."
        ),
        template_name=template_name,
        details={"exception_type": type(exc).__name__},
    )
    return ValidationTemplateResult(
        template_name=template_name,
        status="error",
        issues=(issue,),
        metrics={"issue_count": 1},
        metadata={"executor_error": type(exc).__name__},
    )


def _template_result_name_mismatch(
    expected_template_name: str,
    actual_template_name: str,
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.template_result_mismatch",
        severity="fail",
        message=(
            f"{expected_template_name} returned a result for "
            f"{actual_template_name}; the mismatched result was rejected."
        ),
        template_name=expected_template_name,
        details={
            "expected_template_name": expected_template_name,
            "actual_template_name": actual_template_name,
        },
    )
    return ValidationTemplateResult(
        template_name=expected_template_name,
        status="error",
        issues=(issue,),
        metrics={"issue_count": 1},
        metadata={"executor_result_rejected": True},
    )


def _inline_credential_result(
    template_name: str,
    exc: InlineSecretError,
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.inline_credential_rejected",
        severity="fail",
        message=(
            f"{template_name} returned inline credential material; the unsafe "
            "result was rejected before persistence."
        ),
        template_name=template_name,
        details={"paths": list(exc.paths)},
    )
    return ValidationTemplateResult(
        template_name=template_name,
        status="failed",
        issues=(issue,),
        metrics={"issue_count": 1},
        metadata={"unsafe_result_rejected": True},
    )


def _source_modified_result(
    template_name: str,
    before: tuple[ValidationArtifactIdentity, ...],
    after: tuple[ValidationArtifactIdentity, ...],
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.source_asset_modified",
        severity="fail",
        message=(
            "Validation changed or removed a source asset. The template result "
            "was rejected."
        ),
        template_name=template_name,
        details={
            "source_before": [identity.model_dump(mode="json") for identity in before],
            "source_after": [identity.model_dump(mode="json") for identity in after],
        },
    )
    return ValidationTemplateResult(
        template_name=template_name,
        status="failed",
        issues=(issue,),
        metrics={"issue_count": 1},
        metadata={"source_asset_unchanged": False},
    )


def _missing_reference_result(
    identity: ValidationWorkflowIdentity,
    request: ValidationRequest,
) -> ValidationTemplateResult | None:
    references = identity.reference_artifacts
    missing = tuple(artifact for artifact in references if artifact.kind == "missing")
    if references and not missing:
        return None
    gate_policy = request.policy.get("gate_policy")
    dependency_gate = (
        isinstance(gate_policy, Mapping)
        and str(gate_policy.get("dependency_unavailable", "")).strip().lower()
        == "block"
    )
    required = bool(request.policy.get("reference_evidence_required"))
    severity: Literal["warn", "fail"] = (
        "fail" if dependency_gate or required else "warn"
    )
    issue = ValidationIssue(
        code="visual.reference_evidence_missing",
        severity=severity,
        message=(
            (
                "No reference image evidence was configured; look_right did "
                "not invoke the VLM judge."
            )
            if not references
            else (
                "One or more configured reference images are missing; "
                "look_right did not invoke the VLM judge."
            )
        ),
        template_name="look_right",
        details={"paths": [artifact.path for artifact in missing]},
    )
    return ValidationTemplateResult(
        template_name="look_right",
        status="failed" if severity == "fail" else "skipped",
        issues=(issue,),
        metrics={"issue_count": 1, "vlm_invoked": False},
        metadata={"reference_evidence_available": False},
    )


def _look_right_integrity_failure(
    *,
    code: str,
    message: str,
    metadata: Mapping[str, Any],
    details: Mapping[str, Any] | None = None,
    result: ValidationTemplateResult | None = None,
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code=code,
        severity="fail",
        message=message,
        template_name="look_right",
        details=dict(details or {}),
    )
    issues = (*result.issues, issue) if result is not None else (issue,)
    metrics = dict(result.metrics) if result is not None else {}
    metrics["issue_count"] = len(issues)
    metrics.setdefault("vlm_invoked", False)
    result_metadata = dict(result.metadata) if result is not None else {}
    result_metadata.update(metadata)
    return ValidationTemplateResult(
        template_name="look_right",
        status="failed",
        issues=issues,
        metrics=metrics,
        evidence=dict(result.evidence) if result is not None else {},
        evidence_items=result.evidence_items if result is not None else (),
        metadata=result_metadata,
    )


def _stale_render_handoff_result(
    result: ValidationTemplateResult | None = None,
) -> ValidationTemplateResult:
    return _look_right_integrity_failure(
        code="validation.render_evidence_stale",
        message=(
            "The accepted render_valid evidence is missing or stale; the "
            "look_right result was rejected."
        ),
        metadata={"render_handoff_valid": False},
        result=result,
    )


def _skipped_render_handoff_result(
    render_result: ValidationTemplateResult,
) -> ValidationTemplateResult:
    return ValidationTemplateResult(
        template_name="look_right",
        status="skipped",
        metrics={"issue_count": 0, "vlm_invoked": False},
        metadata={
            "render_handoff_valid": False,
            "skip_reason": "render_valid_skipped",
            "render_valid_status": render_result.status,
        },
    )


def _nonpassing_render_handoff_result(
    render_result: ValidationTemplateResult,
) -> ValidationTemplateResult:
    return ValidationTemplateResult(
        template_name="look_right",
        status="skipped",
        metrics={"issue_count": 0, "vlm_invoked": False},
        metadata={
            "render_handoff_valid": False,
            "skip_reason": f"render_valid_{render_result.status}",
            "render_valid_status": render_result.status,
        },
    )


def _stale_reference_result(
    expected: tuple[ValidationArtifactIdentity, ...],
    current: tuple[ValidationArtifactIdentity, ...],
    *,
    result: ValidationTemplateResult | None = None,
) -> ValidationTemplateResult:
    return _look_right_integrity_failure(
        code="validation.reference_evidence_stale",
        message=(
            "Reference image evidence changed after the workflow identity was "
            "created; the look_right result was rejected."
        ),
        metadata={"reference_evidence_unchanged": False},
        details={
            "reference_before": [
                artifact.model_dump(mode="json") for artifact in expected
            ],
            "reference_after": [
                artifact.model_dump(mode="json") for artifact in current
            ],
        },
        result=result,
    )


def _runtime_render_image_paths(
    result: ValidationTemplateResult,
) -> tuple[str, ...]:
    runtime_render = result.metadata.get("runtime_render")
    if not isinstance(runtime_render, Mapping):
        return ()
    value = runtime_render.get("image_paths")
    if isinstance(value, str | Path):
        return (str(value),)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(str(path) for path in value if isinstance(path, str | Path))
    return ()


def _reported_render_image_paths(
    result: ValidationTemplateResult,
) -> tuple[str, ...]:
    value = result.evidence.get("image_paths")
    if isinstance(value, str | Path):
        return (str(value),)
    if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
        return tuple(str(path) for path in value if isinstance(path, str | Path))
    return ()


def _qualified_render_evidence(
    request: ValidationRequest,
    *,
    identity: ValidationWorkflowIdentity,
    base_dir: Path,
) -> dict[str, tuple[ValidationArtifactIdentity, dict[str, Any]]]:
    raw_records = request.policy.get("qualified_render_evidence")
    if not isinstance(raw_records, Sequence) or isinstance(
        raw_records, str | bytes | bytearray
    ):
        return {}
    # The v1 qualification record binds only a source digest, not a distinct
    # top-level source path. Fail closed for multi-source requests because one
    # record cannot prove current render coverage for every requested source.
    if len(request.inputs) != 1:
        return {}
    # The workflow identity intentionally contains the complete USD dependency
    # closure so mutations to a sublayer or referenced asset invalidate the run.
    # Qualified render authority is narrower: a receipt must bind one of the
    # top-level inputs the operator actually requested, never merely a member of
    # that dependency closure.
    top_level_source_paths = {
        str(_resolve_path(source_path, base_dir=base_dir))
        for source_path in request.inputs
    }
    source_digests = {
        artifact.sha256
        for artifact in identity.source_artifacts
        if artifact.path in top_level_source_paths
        if artifact.kind != "missing" and artifact.sha256 is not None
    }
    if not source_digests:
        return {}
    qualified: dict[str, tuple[ValidationArtifactIdentity, dict[str, Any]]] = {}
    required_fields = {
        "path",
        "role",
        "sha256",
        "source_sha256",
        "qualification",
        "ovrtx_render_metadata",
    }
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != required_fields:
            continue
        path_value = raw_record.get("path")
        digest = raw_record.get("sha256")
        qualification = raw_record.get("qualification")
        render_metadata = raw_record.get("ovrtx_render_metadata")
        if (
            not isinstance(path_value, str | Path)
            or raw_record.get("role") != "qualified_render"
            or not isinstance(digest, str)
            or raw_record.get("source_sha256") not in source_digests
            or not isinstance(qualification, str)
            or not qualification.strip()
            or not _valid_qualified_ovrtx_metadata(render_metadata)
        ):
            continue
        artifact = _artifact_identity(
            path_value,
            role="evidence",
            base_dir=base_dir,
        )
        if artifact.kind != "file" or artifact.sha256 != digest:
            continue
        qualified[artifact.path] = (artifact, dict(raw_record))
    return qualified


def _valid_qualified_ovrtx_metadata(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    views = value.get("views")
    identities = value.get("renderer_identities")
    return bool(
        value.get("backend") in {"ovrtx", "remote"}
        and value.get("renderer") == "ovrtx"
        and value.get("scene_tool") == "usd-cli"
        and isinstance(value.get("scene_tool_source_revision"), str)
        and value["scene_tool_source_revision"]
        and value.get("workflow") == "validation-canonical-visual-evidence"
        and isinstance(views, Sequence)
        and not isinstance(views, str | bytes | bytearray)
        and bool(views)
        and all(isinstance(item, str) and item for item in views)
        and isinstance(value.get("image_width"), int)
        and not isinstance(value.get("image_width"), bool)
        and value["image_width"] > 0
        and isinstance(value.get("image_height"), int)
        and not isinstance(value.get("image_height"), bool)
        and value["image_height"] > 0
        and isinstance(identities, Sequence)
        and not isinstance(identities, str | bytes | bytearray)
        and bool(identities)
        and all(
            (isinstance(item, Mapping) and item.get("engine") == "ovrtx")
            or (value.get("backend") == "ovrtx" and item is None)
            for item in identities
        )
    )


def _is_artifact_path_key(key_hint: str) -> bool:
    if _is_scene_graph_path_key(key_hint):
        return False
    return key_hint in {"images", "path", "paths", "uri", "uris", "output_dir"} or (
        key_hint.endswith(("_path", "_paths", "_uri", "_uris", "_output_dir"))
    )


_ARTIFACT_PATH_ROLE_PARTS = {
    "artifact",
    "artifacts",
    "capture",
    "captures",
    "diff",
    "diffs",
    "evidence",
    "file",
    "files",
    "heatmap",
    "heatmaps",
    "image",
    "images",
    "manifest",
    "manifests",
    "map",
    "maps",
    "mask",
    "masks",
    "overlay",
    "overlays",
    "packet",
    "packets",
    "preview",
    "previews",
    "render",
    "renders",
    "report",
    "reports",
    "sheet",
    "sheets",
    "snapshot",
    "snapshots",
    "texture",
    "textures",
    "thumbnail",
    "thumbnails",
    "video",
    "videos",
}

_SCENE_GRAPH_PATH_PARTS = {
    "ancestor",
    "ancestors",
    "attribute",
    "authoring",
    "body",
    "camera",
    "child",
    "children",
    "collider",
    "collision",
    "collection",
    "collections",
    "component",
    "geometry",
    "instance",
    "joint",
    "light",
    "link",
    "mass",
    "material",
    "mesh",
    "node",
    "nodes",
    "object",
    "parent",
    "physics",
    "prim",
    "primvar",
    "primvars",
    "property",
    "prototype",
    "related",
    "relationship",
    "root",
    "scene",
    "scope",
    "selection",
    "shader",
    "shaders",
    "sibling",
    "siblings",
    "skeleton",
    "subset",
    "subsets",
    "target",
    "xform",
}


def _semantic_path_role(key_hint: str) -> Literal["artifact", "scene"] | None:
    for part in reversed(key_hint.split("_")):
        if part in _ARTIFACT_PATH_ROLE_PARTS:
            return "artifact"
        if part in _SCENE_GRAPH_PATH_PARTS:
            return "scene"
    return None


def _is_scene_graph_path_key(key_hint: str) -> bool:
    if not key_hint.endswith(("_path", "_paths")):
        return False
    return _semantic_path_role(key_hint) == "scene"


def _is_explicit_typed_evidence_artifact_path_key(key_hint: str) -> bool:
    explicit_names = {
        "artifact_path",
        "artifact_paths",
        "evidence_path",
        "evidence_paths",
        "file_path",
        "file_paths",
        "image_path",
        "image_paths",
        "images",
        "manifest_path",
        "manifest_paths",
        "output_dir",
        "path",
        "paths",
        "report_path",
        "report_paths",
        "uri",
        "uris",
        "visual_grounding_packet",
    }
    explicit_suffixes = (
        "_artifact_path",
        "_artifact_paths",
        "_evidence_path",
        "_evidence_paths",
        "_file_path",
        "_file_paths",
        "_image_path",
        "_image_paths",
        "_manifest_path",
        "_manifest_paths",
        "_output_dir",
        "_report_path",
        "_report_paths",
        "_uri",
        "_uris",
    )
    return key_hint in explicit_names or key_hint.endswith(explicit_suffixes)


def _is_typed_evidence_artifact_path_key(key_hint: str) -> bool:
    if _is_scene_graph_path_key(key_hint):
        return False
    return _is_explicit_typed_evidence_artifact_path_key(key_hint) or key_hint.endswith(
        ("_path", "_paths")
    )


ArtifactPathKeyHint = (
    str
    | tuple[
        Literal[
            "context",
            "context_source",
            "ignore",
            "map",
            "pathlike",
            "record",
            "scene",
            "scene_value",
            "source",
            "value",
        ],
        str,
    ]
)

_DESCRIPTIVE_METADATA_KEYS = {
    "alt",
    "alt_text",
    "caption",
    "captions",
    "command",
    "commands",
    "description",
    "descriptions",
    "label",
    "labels",
    "message",
    "messages",
    "name",
    "names",
    "note",
    "notes",
    "reason",
    "reasons",
    "summary",
    "text",
    "title",
    "titles",
}

_NON_PATH_METADATA_KEYS = {
    "author",
    "backend",
    "camera",
    "category",
    "channels",
    "checksum",
    "checksums",
    "code",
    "confidence",
    "color_space",
    "coordinate_system",
    "content_type",
    "created_at",
    "creator",
    "dimensions",
    "digest",
    "digests",
    "dtype",
    "duration",
    "engine",
    "encoding",
    "format",
    "framework",
    "fps",
    "frame_rate",
    "hash",
    "hashes",
    "height",
    "id",
    "ids",
    "index",
    "indices",
    "kind",
    "md5",
    "media_type",
    "mime_type",
    "mode",
    "model",
    "original",
    "phase",
    "preset",
    "producer",
    "profile",
    "provider",
    "quality",
    "render_output_dir",
    "render_time",
    "revision",
    "role",
    "score",
    "severity",
    "sha",
    "sha1",
    "sha256",
    "sha512",
    "size",
    "status",
    "source",
    "schema_version",
    "subject",
    "tag",
    "tags",
    "type",
    "unit",
    "units",
    "updated_at",
    "verdict",
    "version",
    "view",
    "views",
    "width",
}

_NON_PATH_METADATA_SUFFIXES = tuple(
    f"_{key}" for key in sorted(_NON_PATH_METADATA_KEYS | {"ids", "indices"})
)

_ARTIFACT_FILE_SUFFIXES = {
    ".bmp",
    ".csv",
    ".exr",
    ".fbx",
    ".gif",
    ".glb",
    ".gltf",
    ".hdr",
    ".htm",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".log",
    ".md",
    ".mtl",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".obj",
    ".pdf",
    ".ply",
    ".png",
    ".stl",
    ".tif",
    ".tiff",
    ".txt",
    ".usd",
    ".usda",
    ".usdc",
    ".usdz",
    ".webm",
    ".webp",
    ".xml",
    ".yaml",
    ".yml",
    ".zip",
}

_AUTHORITATIVE_ARTIFACT_CONTEXT_KEYS = {
    "artifact",
    "artifacts",
    "derivative",
    "derivatives",
}


def _nested_artifact_path_context_hint(
    key_hint: str,
) -> tuple[Literal["context"], str]:
    return ("context", key_hint)


def _nested_artifact_path_record_hint(
    key_hint: str,
) -> tuple[Literal["record"], str]:
    return ("record", key_hint)


def _nested_artifact_path_map_hint(
    key_hint: str,
) -> tuple[Literal["map"], str]:
    return ("map", key_hint)


def _artifact_path_container_key(key_hint: ArtifactPathKeyHint) -> str:
    return key_hint[1] if isinstance(key_hint, tuple) else key_hint


def _nested_artifact_path_value_hint(
    key_hint: str,
) -> tuple[Literal["value"], str]:
    return ("value", key_hint)


def _ignored_artifact_path_hint(
    key_hint: str,
) -> tuple[Literal["ignore"], str]:
    return ("ignore", key_hint)


def _scene_artifact_path_context_hint(
    key_hint: str,
) -> tuple[Literal["scene"], str]:
    return ("scene", key_hint)


def _is_artifact_path_context_key(key_hint: str) -> bool:
    return key_hint in {
        "context",
        "contexts",
        "detail",
        "details",
        "diagnostic",
        "diagnostics",
        "error",
        "errors",
        "issue",
        "issues",
        "meta",
        "metadata",
        "render_response",
        "response",
        "responses",
        "warning",
        "warnings",
    } or key_hint.endswith(
        (
            "_context",
            "_contexts",
            "_detail",
            "_details",
            "_diagnostic",
            "_diagnostics",
            "_error",
            "_errors",
            "_issue",
            "_issues",
            "_response",
            "_responses",
            "_warning",
            "_warnings",
        )
    )


def _has_known_artifact_file_suffix(path: Path) -> bool:
    return path.suffix.lower() in _ARTIFACT_FILE_SUFFIXES


def _has_artifact_file_suffix(path: Path) -> bool:
    return not any(character.isspace() for character in path.name) and (
        _has_known_artifact_file_suffix(path)
    )


def _exists_for_artifact_classification(path: Path) -> bool:
    try:
        return path.exists()
    except (OSError, ValueError):
        return False


def _parse_artifact_uri(value: str) -> ParseResult | None:
    try:
        return urlparse(value)
    except ValueError:
        return None


def _is_windows_drive_artifact_path(
    value: str,
    *,
    parsed: ParseResult,
) -> bool:
    return (
        len(parsed.scheme) == 1
        and len(value) > 2
        and value[1] == ":"
        and value[2] in {"/", "\\"}
    )


def _path_is_declared_artifact(
    path: Path,
    *,
    base_dir: Path,
    suffix_predicate: Callable[[Path], bool],
    allow_external_absolute_suffix: bool,
) -> bool:
    resolved_base = base_dir.resolve()
    if path.is_absolute():
        try:
            resolved_path = path.resolve()
        except (OSError, ValueError):
            return True
        if resolved_path.is_relative_to(
            resolved_base
        ) or _exists_for_artifact_classification(resolved_path):
            return True
        return allow_external_absolute_suffix and suffix_predicate(path)
    try:
        resolved_path = (resolved_base / path).resolve()
    except (OSError, ValueError):
        return True
    if _exists_for_artifact_classification(resolved_path):
        return True
    if not resolved_path.is_relative_to(resolved_base):
        return allow_external_absolute_suffix and suffix_predicate(path)
    return suffix_predicate(path)


def _looks_like_local_artifact_path(
    value: Any,
    *,
    base_dir: Path,
) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = _parse_artifact_uri(value)
    if parsed is None:
        return True
    if parsed.scheme:
        if _is_windows_drive_artifact_path(value, parsed=parsed):
            return True
        return parsed.scheme == "file"
    return _path_is_declared_artifact(
        Path(value),
        base_dir=base_dir,
        suffix_predicate=_has_artifact_file_suffix,
        allow_external_absolute_suffix=True,
    )


def _looks_like_pathlike_artifact_path(
    value: str,
    *,
    base_dir: Path,
) -> bool:
    if _looks_like_local_artifact_path(value, base_dir=base_dir):
        return True
    parsed = _parse_artifact_uri(value)
    if parsed is None:
        return False
    if parsed.scheme:
        return True
    path = Path(value)
    return (
        path.is_absolute()
        or _has_known_artifact_file_suffix(path)
        or any(separator in value for separator in {"/", "\\"})
        or value.startswith(("~", "."))
    )


def _looks_like_source_artifact_path(
    value: str,
    *,
    base_dir: Path,
) -> bool:
    if _looks_like_local_artifact_path(value, base_dir=base_dir):
        return True
    parsed = _parse_artifact_uri(value)
    if parsed is None:
        return True
    if parsed.scheme:
        return True
    path = Path(value)
    if _has_known_artifact_file_suffix(path):
        return True
    return (
        path.is_absolute()
        or any(separator in value for separator in {"/", "\\"})
        or value.startswith(("~", "."))
        or not any(character.isspace() for character in value)
        and bool(path.suffix)
    )


def _looks_like_scene_key_artifact_path(
    value: Any,
    *,
    base_dir: Path,
) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if value.startswith("~"):
        return True
    parsed = _parse_artifact_uri(value)
    if parsed is None:
        return True
    if parsed.scheme:
        return True
    local = value
    local_path = Path(local)
    if local_path.is_absolute():
        try:
            resolved_path = local_path.resolve()
            resolved_base = base_dir.resolve()
        except (OSError, ValueError):
            return True
        if not resolved_path.is_relative_to(resolved_base):
            return resolved_path != Path(
                resolved_path.anchor
            ) and _exists_for_artifact_classification(resolved_path)
    return _path_is_declared_artifact(
        local_path,
        base_dir=base_dir,
        suffix_predicate=_has_artifact_file_suffix,
        allow_external_absolute_suffix=False,
    )


def _pathlike_artifact_path_hint(
    key_hint: str,
) -> tuple[Literal["pathlike"], str]:
    return ("pathlike", key_hint)


def _source_artifact_path_hint(
    key_hint: str,
) -> tuple[Literal["source"], str]:
    return ("source", key_hint)


def _context_source_artifact_path_hint(
    key_hint: str,
) -> tuple[Literal["context_source"], str]:
    return ("context_source", key_hint)


def _scene_value_artifact_path_hint(
    key_hint: str,
) -> tuple[Literal["scene_value"], str]:
    return ("scene_value", key_hint)


def _is_artifact_path_value(
    key_hint: ArtifactPathKeyHint,
    *,
    base_dir: Path,
    value: str,
    path_key_predicate: Callable[[str], bool],
) -> bool:
    if isinstance(key_hint, str):
        return path_key_predicate(key_hint)
    hint_kind = key_hint[0]
    return (
        hint_kind == "value"
        or (
            hint_kind == "pathlike"
            and _looks_like_pathlike_artifact_path(
                value,
                base_dir=base_dir,
            )
        )
        or (
            hint_kind == "source"
            and _looks_like_source_artifact_path(
                value,
                base_dir=base_dir,
            )
        )
        or (
            hint_kind == "context_source"
            and _looks_like_pathlike_artifact_path(
                value,
                base_dir=base_dir,
            )
        )
        or (
            hint_kind == "scene_value"
            and _looks_like_scene_key_artifact_path(value, base_dir=base_dir)
        )
    )


def _nested_artifact_path_key_hint(
    *,
    parent_key_hint: ArtifactPathKeyHint,
    child_key_hint: str,
    child_value: Any,
    path_key_predicate: Callable[[str], bool],
) -> ArtifactPathKeyHint:
    parent_container_key = _artifact_path_container_key(parent_key_hint)
    child_is_container = isinstance(child_value, Mapping) or (
        isinstance(child_value, Sequence)
        and not isinstance(child_value, str | bytes | bytearray)
    )
    if isinstance(parent_key_hint, tuple):
        parent_mode = parent_key_hint[0]
        if parent_mode == "ignore":
            return parent_key_hint
        if (
            parent_mode == "context"
            and parent_container_key in {"render_response", "runtime_render"}
            and child_key_hint == "usd_path"
        ):
            return _ignored_artifact_path_hint(child_key_hint)
        if parent_mode == "scene":
            if (
                _is_non_path_metadata_key(child_key_hint)
                or child_key_hint in _DESCRIPTIVE_METADATA_KEYS
            ):
                return _ignored_artifact_path_hint(child_key_hint)
            if _is_scene_graph_path_key(child_key_hint):
                if child_is_container:
                    return _scene_artifact_path_context_hint(child_key_hint)
                return _scene_value_artifact_path_hint(child_key_hint)
            if child_key_hint in {"path", "paths", "uri", "uris"}:
                if child_is_container:
                    return _scene_artifact_path_context_hint(child_key_hint)
                return _scene_value_artifact_path_hint(child_key_hint)
            if path_key_predicate(child_key_hint) or _is_named_artifact_path_value_key(
                child_key_hint
            ):
                if isinstance(child_value, Mapping):
                    return _artifact_path_mapping_hint(
                        child_key_hint,
                        child_value,
                    )
                return _nested_artifact_path_value_hint(child_key_hint)
            if child_is_container:
                return parent_key_hint
            return _ignored_artifact_path_hint(child_key_hint)
        if parent_mode in {"pathlike", "scene_value"}:
            return _ignored_artifact_path_hint(child_key_hint)
        if parent_mode in {"map", "record", "value"} and child_key_hint == "source":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    child_key_hint,
                    child_value,
                )
            return _source_artifact_path_hint(child_key_hint)
        if (
            parent_mode == "record"
            and path_key_predicate(parent_container_key)
            and _is_artifact_path_context_key(child_key_hint)
            and child_is_container
        ):
            return _nested_artifact_path_value_hint(parent_container_key)
        if parent_mode == "context" and child_key_hint == "source":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    child_key_hint,
                    child_value,
                )
            return _context_source_artifact_path_hint(child_key_hint)
        if (
            parent_mode in {"map", "record", "value"}
            and _is_artifact_path_context_key(child_key_hint)
            and not path_key_predicate(parent_container_key)
        ):
            if child_is_container:
                return _nested_artifact_path_context_hint(parent_container_key)
            return _ignored_artifact_path_hint(child_key_hint)
        parent_is_explicit_path_container = (
            parent_mode == "value"
            and parent_container_key != "images"
            and path_key_predicate(parent_container_key)
        )
        if parent_is_explicit_path_container and not isinstance(
            child_value,
            Mapping,
        ):
            if child_key_hint == "render_output_dir":
                return _nested_artifact_path_value_hint(child_key_hint)
            if _is_scene_graph_path_key(child_key_hint):
                if child_is_container:
                    return _scene_artifact_path_context_hint(child_key_hint)
                return _scene_value_artifact_path_hint(child_key_hint)
            if (
                _is_non_path_metadata_key(child_key_hint)
                or child_key_hint in _DESCRIPTIVE_METADATA_KEYS
            ):
                return _ignored_artifact_path_hint(child_key_hint)
            if path_key_predicate(child_key_hint) or _is_named_artifact_path_value_key(
                child_key_hint
            ):
                return _nested_artifact_path_value_hint(child_key_hint)
            return _nested_artifact_path_value_hint(parent_container_key)
        if _is_non_path_metadata_key(child_key_hint):
            return _ignored_artifact_path_hint(child_key_hint)
        if child_key_hint in _DESCRIPTIVE_METADATA_KEYS:
            return _ignored_artifact_path_hint(child_key_hint)
        if _is_scene_graph_path_key(child_key_hint):
            if child_is_container:
                return _scene_artifact_path_context_hint(child_key_hint)
            if parent_mode in {"context", "map", "record"}:
                return _scene_value_artifact_path_hint(child_key_hint)
            return _ignored_artifact_path_hint(child_key_hint)
        if child_key_hint in {"path", "paths", "uri", "uris"} or path_key_predicate(
            child_key_hint
        ):
            if isinstance(child_value, Mapping):
                mapping_hint = _artifact_path_mapping_hint(
                    child_key_hint,
                    child_value,
                )
                if (
                    child_key_hint == "artifact_paths"
                    and mapping_hint[0] == "map"
                    and not _has_plural_artifact_path_record_marker(
                        child_value,
                        path_key_predicate=path_key_predicate,
                    )
                ):
                    return _nested_artifact_path_value_hint(child_key_hint)
                return mapping_hint
            return _nested_artifact_path_value_hint(child_key_hint)
        if (
            parent_mode == "record"
            and not isinstance(child_value, Mapping)
            and _is_named_artifact_path_value_key(child_key_hint)
        ):
            return _nested_artifact_path_value_hint(child_key_hint)
        if parent_mode == "value":
            if isinstance(child_value, Mapping):
                if path_key_predicate(
                    parent_container_key
                ) and _is_artifact_path_context_key(child_key_hint):
                    return parent_key_hint
                if _is_artifact_path_context_key(
                    child_key_hint
                ) and not path_key_predicate(parent_container_key):
                    return _nested_artifact_path_context_hint(parent_container_key)
                return _artifact_path_mapping_hint(
                    parent_container_key,
                    child_value,
                )
            if parent_container_key == "images":
                if isinstance(child_value, str | Sequence):
                    return _nested_artifact_path_value_hint(parent_container_key)
                return _ignored_artifact_path_hint(child_key_hint)
            if path_key_predicate(parent_container_key) and isinstance(
                child_value, str | Sequence
            ):
                return _nested_artifact_path_value_hint(parent_container_key)
            return child_key_hint
        if parent_mode == "map":
            if isinstance(child_value, Mapping):
                if _is_artifact_path_context_key(
                    child_key_hint
                ) and not path_key_predicate(parent_container_key):
                    return _nested_artifact_path_context_hint(parent_container_key)
                return _artifact_path_mapping_hint(
                    parent_container_key,
                    child_value,
                )
            if child_is_container:
                return parent_key_hint
            if isinstance(child_value, str | Sequence):
                return _nested_artifact_path_value_hint(parent_container_key)
            return child_key_hint
        if parent_mode == "record":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    child_key_hint,
                    child_value,
                )
            if isinstance(child_value, Sequence) and not isinstance(
                child_value, str | bytes | bytearray
            ):
                return _nested_artifact_path_value_hint(parent_container_key)
            if isinstance(child_value, str):
                return _nested_artifact_path_value_hint(parent_container_key)
            return child_key_hint
        if parent_mode == "context":
            if (
                _is_non_path_metadata_key(child_key_hint)
                or child_key_hint in _DESCRIPTIVE_METADATA_KEYS
            ):
                return _ignored_artifact_path_hint(child_key_hint)
            if _is_scene_graph_path_key(child_key_hint):
                if child_is_container:
                    return _scene_artifact_path_context_hint(child_key_hint)
                return _scene_value_artifact_path_hint(child_key_hint)
            if _is_named_artifact_path_container_key(
                child_key_hint,
                path_key_predicate=path_key_predicate,
            ):
                if isinstance(child_value, Mapping):
                    return _artifact_path_mapping_hint(
                        child_key_hint,
                        child_value,
                    )
                if _is_exact_named_artifact_path_value_key(child_key_hint):
                    return _nested_artifact_path_value_hint(child_key_hint)
                return _pathlike_artifact_path_hint(child_key_hint)
            if child_is_container:
                return parent_key_hint
            if parent_container_key in _AUTHORITATIVE_ARTIFACT_CONTEXT_KEYS:
                return _pathlike_artifact_path_hint(child_key_hint)
            return child_key_hint
    if (
        _is_non_path_metadata_key(child_key_hint)
        or child_key_hint in _DESCRIPTIVE_METADATA_KEYS
    ):
        return _ignored_artifact_path_hint(child_key_hint)
    if _is_artifact_path_context_key(child_key_hint) and child_is_container:
        return _nested_artifact_path_context_hint(child_key_hint)
    if _is_scene_graph_path_key(child_key_hint):
        if child_is_container:
            return _scene_artifact_path_context_hint(child_key_hint)
        return _scene_value_artifact_path_hint(child_key_hint)
    if path_key_predicate(child_key_hint):
        if isinstance(child_value, Mapping):
            if child_key_hint == "images":
                return _nested_artifact_path_map_hint(child_key_hint)
            mapping_hint = _artifact_path_mapping_hint(
                child_key_hint,
                child_value,
            )
            if (
                child_key_hint == "artifact_paths"
                and mapping_hint[0] == "map"
                and not _has_plural_artifact_path_record_marker(
                    child_value,
                    path_key_predicate=path_key_predicate,
                )
            ):
                return _nested_artifact_path_value_hint(child_key_hint)
            return mapping_hint
        return _nested_artifact_path_value_hint(child_key_hint)
    if _is_named_artifact_path_container_key(
        child_key_hint,
        path_key_predicate=path_key_predicate,
    ):
        if isinstance(child_value, Mapping):
            if child_key_hint == "images":
                return _nested_artifact_path_map_hint(child_key_hint)
            return _artifact_path_mapping_hint(
                child_key_hint,
                child_value,
            )
        if _is_exact_named_artifact_path_value_key(child_key_hint):
            return _nested_artifact_path_value_hint(child_key_hint)
        return _pathlike_artifact_path_hint(child_key_hint)
    return child_key_hint


def _sequence_artifact_path_key_hint(
    *,
    parent_key_hint: ArtifactPathKeyHint,
    child_value: Any,
    path_key_predicate: Callable[[str], bool],
) -> ArtifactPathKeyHint:
    if isinstance(parent_key_hint, tuple):
        parent_mode = parent_key_hint[0]
        if parent_mode == "ignore":
            return parent_key_hint
        if parent_mode == "scene":
            if isinstance(child_value, Mapping) or (
                isinstance(child_value, Sequence)
                and not isinstance(child_value, str | bytes | bytearray)
            ):
                return parent_key_hint
            return _scene_value_artifact_path_hint(
                _artifact_path_container_key(parent_key_hint)
            )
        if parent_mode == "pathlike":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    _artifact_path_container_key(parent_key_hint),
                    child_value,
                )
            return parent_key_hint
        if parent_mode == "scene_value":
            if isinstance(child_value, Mapping):
                return _ignored_artifact_path_hint(
                    _artifact_path_container_key(parent_key_hint)
                )
            return parent_key_hint
        if parent_mode == "value":
            if isinstance(child_value, Mapping):
                container_key = _artifact_path_container_key(parent_key_hint)
                if container_key != "images" and path_key_predicate(container_key):
                    return parent_key_hint
                return _artifact_path_mapping_hint(
                    container_key,
                    child_value,
                )
            return parent_key_hint
        if parent_mode in {"context_source", "source"}:
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    _artifact_path_container_key(parent_key_hint),
                    child_value,
                )
            return parent_key_hint
        if parent_mode == "map":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    _artifact_path_container_key(parent_key_hint),
                    child_value,
                )
            if isinstance(child_value, Sequence) and not isinstance(
                child_value, str | bytes | bytearray
            ):
                return parent_key_hint
            return _nested_artifact_path_value_hint(
                _artifact_path_container_key(parent_key_hint)
            )
        if parent_mode == "record":
            if isinstance(child_value, Mapping):
                return _artifact_path_mapping_hint(
                    _artifact_path_container_key(parent_key_hint),
                    child_value,
                )
            if isinstance(child_value, Sequence) and not isinstance(
                child_value, str | bytes | bytearray
            ):
                return parent_key_hint
            return _nested_artifact_path_value_hint(
                _artifact_path_container_key(parent_key_hint)
            )
        if parent_mode == "context":
            if isinstance(child_value, Mapping) or (
                isinstance(child_value, Sequence)
                and not isinstance(child_value, str | bytes | bytearray)
            ):
                return parent_key_hint
            return _ignored_artifact_path_hint(
                _artifact_path_container_key(parent_key_hint)
            )
        return _ignored_artifact_path_hint(
            _artifact_path_container_key(parent_key_hint)
        )
    if path_key_predicate(parent_key_hint):
        if isinstance(child_value, Mapping):
            return _artifact_path_mapping_hint(
                parent_key_hint,
                child_value,
            )
        return _nested_artifact_path_value_hint(parent_key_hint)
    return parent_key_hint


def _artifact_path_mapping_hint(
    container_key: str,
    value: Mapping[Any, Any],
) -> ArtifactPathKeyHint:
    if _has_singular_artifact_path_record_marker(value):
        return _nested_artifact_path_record_hint(container_key)
    return _nested_artifact_path_map_hint(container_key)


def _has_singular_artifact_path_record_marker(
    value: Mapping[Any, Any],
) -> bool:
    return any(str(key).lower() in {"path", "uri"} for key in value)


def _has_plural_artifact_path_record_marker(
    value: Mapping[Any, Any],
    *,
    path_key_predicate: Callable[[str], bool],
) -> bool:
    # An exact plural name is a marker on its own; an ambiguous ``*_paths`` or
    # ``*_uris`` suffix must also satisfy the predicate. Both predicates in use
    # already match the exact names, so this precedence is behaviourally
    # neutral today - test_plural_artifact_marker_precedence_is_behaviourally
    # _neutral pins that invariant.
    return any(
        (
            key_hint in {"images", "paths", "uris"}
            or (key_hint.endswith(("_paths", "_uris")) and path_key_predicate(key_hint))
        )
        for key in value
        for key_hint in (str(key).lower(),)
    )


def _is_named_artifact_path_container_key(
    key_hint: str,
    *,
    path_key_predicate: Callable[[str], bool],
) -> bool:
    if path_key_predicate(key_hint):
        return True
    return _is_named_artifact_path_value_key(key_hint) or key_hint.endswith(
        (
            "_artifacts",
            "_captures",
            "_evidence",
            "_files",
            "_images",
            "_previews",
            "_renders",
            "_reports",
            "_snapshots",
            "_textures",
            "_thumbnails",
            "_videos",
        )
    )


_EXACT_NAMED_ARTIFACT_PATH_VALUE_KEYS = {
    "artifact",
    "artifacts",
    "capture",
    "captures",
    "contact_sheet",
    "contact_sheets",
    "diff",
    "diffs",
    "evidence",
    "file",
    "files",
    "image",
    "images",
    "manifest",
    "manifests",
    "map",
    "maps",
    "preview",
    "previews",
    "render",
    "renders",
    "report",
    "reports",
    "snapshot",
    "snapshots",
    "texture",
    "textures",
    "thumbnail",
    "thumbnails",
    "video",
    "videos",
}


def _is_exact_named_artifact_path_value_key(key_hint: str) -> bool:
    return key_hint in _EXACT_NAMED_ARTIFACT_PATH_VALUE_KEYS


def _is_named_artifact_path_value_key(key_hint: str) -> bool:
    return (
        _is_exact_named_artifact_path_value_key(key_hint)
        or key_hint.rsplit("_", maxsplit=1)[-1] in _ARTIFACT_PATH_ROLE_PARTS
    )


def _is_non_path_metadata_key(key_hint: str) -> bool:
    candidate_keys = {key_hint}
    if key_hint.endswith("ies"):
        candidate_keys.add(f"{key_hint[:-3]}y")
    if key_hint.endswith("es"):
        candidate_keys.add(key_hint[:-2])
    if key_hint.endswith("s"):
        candidate_keys.add(key_hint[:-1])
    non_path_suffixes = _NON_PATH_METADATA_SUFFIXES + (
        "_alt",
        "_alt_text",
        "_caption",
        "_captions",
        "_command",
        "_commands",
        "_count",
        "_description",
        "_descriptions",
        "_fps",
        "_frame_rate",
        "_kind",
        "_label",
        "_labels",
        "_message",
        "_messages",
        "_name",
        "_names",
        "_note",
        "_notes",
        "_reason",
        "_reasons",
        "_status",
        "_summary",
        "_summaries",
        "_system",
        "_text",
        "_title",
        "_titles",
        "_type",
        "_version",
    )
    return any(
        candidate in _NON_PATH_METADATA_KEYS
        or candidate in _DESCRIPTIVE_METADATA_KEYS
        or candidate.endswith(non_path_suffixes)
        for candidate in candidate_keys
    )


def _canonicalize_declared_artifact_paths(
    value: Any,
    *,
    base_dir: Path,
    key_hint: ArtifactPathKeyHint = "",
    path_key_predicate: Callable[[str], bool] = _is_artifact_path_key,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _canonicalize_declared_artifact_paths(
                child,
                base_dir=base_dir,
                key_hint=_nested_artifact_path_key_hint(
                    parent_key_hint=key_hint,
                    child_key_hint=str(key).lower(),
                    child_value=child,
                    path_key_predicate=path_key_predicate,
                ),
                path_key_predicate=path_key_predicate,
            )
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _canonicalize_declared_artifact_paths(
                child,
                base_dir=base_dir,
                key_hint=_sequence_artifact_path_key_hint(
                    parent_key_hint=key_hint,
                    child_value=child,
                    path_key_predicate=path_key_predicate,
                ),
                path_key_predicate=path_key_predicate,
            )
            for child in value
        ]
    if not isinstance(value, str) or not _is_artifact_path_value(
        key_hint,
        base_dir=base_dir,
        value=value,
        path_key_predicate=path_key_predicate,
    ):
        return value
    if not value.strip():
        return value
    parsed = _parse_artifact_uri(value)
    if parsed is None:
        return value
    is_windows_drive_path = _is_windows_drive_artifact_path(value, parsed=parsed)
    if parsed.scheme and parsed.scheme != "file" and not is_windows_drive_path:
        return value
    if parsed.scheme == "file" and parsed.netloc.lower() not in {"", "localhost"}:
        return value
    if (
        isinstance(key_hint, tuple)
        and key_hint[0] == "scene_value"
        and (parsed.scheme == "file" or value.startswith("~"))
    ):
        return value
    local = unquote(parsed.path) if parsed.scheme == "file" else value
    if not local.strip():
        return value
    return _best_effort_resolved_path(local, base_dir=base_dir)


def _canonicalize_evidence_item_path(
    path: str | None,
    *,
    base_dir: Path,
) -> str | None:
    if path is None or not path.strip():
        return path
    parsed = _parse_artifact_uri(path)
    if parsed is None:
        return path
    is_windows_drive_path = _is_windows_drive_artifact_path(path, parsed=parsed)
    if parsed.scheme and parsed.scheme != "file" and not is_windows_drive_path:
        return path
    if parsed.scheme == "file" and parsed.netloc.lower() not in {"", "localhost"}:
        return path
    local = unquote(parsed.path) if parsed.scheme == "file" else path
    if not local.strip():
        return path
    return _best_effort_resolved_path(local, base_dir=base_dir)


def _canonicalize_operational_metadata_paths(
    value: Any,
    *,
    base_dir: Path,
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: (
                _best_effort_resolved_path(child, base_dir=base_dir)
                if str(key).lower() == "working_dir"
                and isinstance(child, str)
                and child.strip()
                else _canonicalize_operational_metadata_paths(
                    child,
                    base_dir=base_dir,
                )
            )
            for key, child in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [
            _canonicalize_operational_metadata_paths(
                child,
                base_dir=base_dir,
            )
            for child in value
        ]
    return value


def _canonicalize_result_artifact_paths(
    result: ValidationTemplateResult,
    *,
    base_dir: Path,
) -> ValidationTemplateResult:
    evidence_items = tuple(
        item.model_copy(
            update={
                "path": _canonicalize_evidence_item_path(
                    item.path,
                    base_dir=base_dir,
                ),
                "metadata": _canonicalize_declared_artifact_paths(
                    item.metadata,
                    base_dir=base_dir,
                    path_key_predicate=_is_typed_evidence_artifact_path_key,
                ),
            }
        )
        for item in result.evidence_items
    )
    return result.model_copy(
        update={
            "evidence": _canonicalize_declared_artifact_paths(
                result.evidence,
                base_dir=base_dir,
                path_key_predicate=_is_typed_evidence_artifact_path_key,
            ),
            "evidence_items": evidence_items,
            "metadata": _canonicalize_operational_metadata_paths(
                _canonicalize_declared_artifact_paths(
                    result.metadata,
                    base_dir=base_dir,
                    path_key_predicate=_is_typed_evidence_artifact_path_key,
                ),
                base_dir=base_dir,
            ),
        }
    )


def _input_artifact_collision_paths(
    artifacts: Sequence[ValidationArtifactIdentity],
    *,
    forbidden_artifacts: Sequence[ValidationArtifactIdentity],
) -> tuple[str, ...]:
    forbidden_files = {
        Path(artifact.path)
        for artifact in forbidden_artifacts
        if artifact.kind == "file"
    }
    forbidden_directories = tuple(
        Path(artifact.path)
        for artifact in forbidden_artifacts
        if artifact.kind == "directory"
    )
    forbidden_file_inodes: set[tuple[int, int]] = set()
    forbidden_file_candidates = list(forbidden_files)
    for forbidden_directory in forbidden_directories:
        try:
            forbidden_file_candidates.extend(
                member for member in forbidden_directory.rglob("*") if member.is_file()
            )
        except OSError:
            continue
    for forbidden_file in forbidden_file_candidates:
        try:
            file_stat = forbidden_file.stat()
        except OSError:
            continue
        forbidden_file_inodes.add((file_stat.st_dev, file_stat.st_ino))
    collisions: list[str] = []
    for artifact in artifacts:
        artifact_path = Path(artifact.path)
        aliases_input = artifact_path in forbidden_files or any(
            directory == artifact_path or directory in artifact_path.parents
            for directory in forbidden_directories
        )
        try:
            artifact_stat = artifact_path.stat()
            aliases_input = (
                aliases_input
                or (
                    artifact_stat.st_dev,
                    artifact_stat.st_ino,
                )
                in forbidden_file_inodes
            )
        except OSError:
            pass
        if aliases_input:
            collisions.append(artifact.path)
    return tuple(collisions)


def _validate_render_evidence_result(
    result: ValidationTemplateResult,
    *,
    base_dir: Path,
    qualified_evidence_base_dir: Path | None = None,
    request: ValidationRequest,
    identity: ValidationWorkflowIdentity,
    forbidden_artifacts: Sequence[ValidationArtifactIdentity],
) -> ValidationTemplateResult:
    if result.status != "passed":
        return result
    runtime_image_paths = _runtime_render_image_paths(result)
    qualified = _qualified_render_evidence(
        request,
        identity=identity,
        base_dir=qualified_evidence_base_dir or base_dir,
    )
    reported_image_paths = _reported_render_image_paths(result)
    qualified_image_paths = tuple(
        path
        for path in reported_image_paths
        if _best_effort_resolved_path(path, base_dir=base_dir) in qualified
    )
    using_qualified_evidence = bool(
        not runtime_image_paths
        and qualified_image_paths
        and len(qualified_image_paths) == len(reported_image_paths)
    )
    image_paths = (
        qualified_image_paths if using_qualified_evidence else runtime_image_paths
    )
    identity_paths: tuple[str | Path, ...] = tuple(
        (
            unquote(parsed.path)
            if parsed is not None
            and parsed.scheme == "file"
            and parsed.netloc.lower() in {"", "localhost"}
            and parsed.path
            else path
        )
        for path in image_paths
        for parsed in (_parse_artifact_uri(str(path)),)
    )
    image_artifacts = _best_effort_artifact_identities(
        identity_paths,
        role="evidence",
        base_dir=base_dir,
    )
    collisions = _input_artifact_collision_paths(
        image_artifacts,
        forbidden_artifacts=forbidden_artifacts,
    )
    if (
        image_artifacts
        and all(artifact.kind == "file" for artifact in image_artifacts)
        and not collisions
    ):
        metadata = dict(result.metadata)
        qualified_records = [
            qualified[artifact.path][1]
            for artifact in image_artifacts
            if artifact.path in qualified
        ]
        metadata.update(
            {
                "render_evidence_accepted": True,
                "render_evidence_origin": (
                    "qualified_precomputed"
                    if using_qualified_evidence
                    else "runtime_render"
                ),
                "qualified_render_evidence": qualified_records,
            }
        )
        if using_qualified_evidence:
            qualified_backends = {
                render_metadata.get("backend")
                for record in qualified_records
                for render_metadata in (record.get("ovrtx_render_metadata"),)
                if isinstance(render_metadata, Mapping)
                and isinstance(render_metadata.get("backend"), str)
            }
            metadata["runtime_render"] = {
                "status": "completed",
                "backend": (
                    next(iter(qualified_backends))
                    if len(qualified_backends) == 1
                    else None
                ),
                "image_paths": [artifact.path for artifact in image_artifacts],
                "render_response": None,
                "render_output_dir": None,
                "issues": [],
                "metadata": {
                    "render_evidence_origin": "qualified_precomputed",
                },
            }
        return result.model_copy(update={"metadata": metadata})

    issue_code = (
        "validation.render_evidence_input_collision"
        if collisions
        else "validation.render_evidence_missing"
    )
    issue = ValidationIssue(
        code=issue_code,
        severity="fail",
        message=(
            (
                "render_valid reported a source or reference input as runtime "
                "render evidence without an exact qualified-render binding; "
                "the result was rejected."
            )
            if collisions
            else (
                "render_valid reported success without current local runtime "
                "render images; the result was rejected."
            )
        ),
        template_name="render_valid",
        details={
            "claimed_image_paths": list(image_paths),
            "input_collision_paths": list(collisions),
            "image_artifacts": [
                artifact.model_dump(mode="json") for artifact in image_artifacts
            ],
            "qualified_render_paths": sorted(qualified),
        },
    )
    issues = (*result.issues, issue)
    metrics = dict(result.metrics)
    metrics["issue_count"] = len(issues)
    metadata = dict(result.metadata)
    metadata["render_evidence_accepted"] = False
    return result.model_copy(
        update={
            "status": "failed",
            "issues": issues,
            "metrics": metrics,
            "metadata": metadata,
        }
    )


def _candidate_artifact_paths(
    value: Any,
    *,
    base_dir: Path,
    key_hint: ArtifactPathKeyHint = "",
    path_key_predicate: Callable[[str], bool] = _is_artifact_path_key,
) -> tuple[str, ...]:
    paths: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            paths.extend(
                _candidate_artifact_paths(
                    child,
                    base_dir=base_dir,
                    key_hint=_nested_artifact_path_key_hint(
                        parent_key_hint=key_hint,
                        child_key_hint=str(key).lower(),
                        child_value=child,
                        path_key_predicate=path_key_predicate,
                    ),
                    path_key_predicate=path_key_predicate,
                )
            )
    elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        for child in value:
            paths.extend(
                _candidate_artifact_paths(
                    child,
                    base_dir=base_dir,
                    key_hint=_sequence_artifact_path_key_hint(
                        parent_key_hint=key_hint,
                        child_value=child,
                        path_key_predicate=path_key_predicate,
                    ),
                    path_key_predicate=path_key_predicate,
                )
            )
    elif isinstance(value, str):
        if _is_artifact_path_value(
            key_hint,
            base_dir=base_dir,
            value=value,
            path_key_predicate=path_key_predicate,
        ):
            paths.append(value)
    return tuple(paths)


def _evidence_artifacts(
    result: ValidationTemplateResult,
    *,
    base_dir: Path,
) -> tuple[tuple[ValidationArtifactIdentity, ...], tuple[str, ...]]:
    # ``qualified_render_evidence`` is an already validated typed provenance
    # record. Its renderer names, workflow slug, qualification text, and source
    # digest are metadata, not path candidates. The actual image remains
    # discoverable through ``result.evidence.image_paths`` and is still bound
    # below by its exact file identity.
    artifact_metadata = dict(result.metadata)
    artifact_metadata.pop("qualified_render_evidence", None)
    candidates = (
        *(item.path for item in result.evidence_items if item.path is not None),
        *(
            candidate
            for item in result.evidence_items
            for candidate in _candidate_artifact_paths(
                item.metadata,
                base_dir=base_dir,
                path_key_predicate=_is_typed_evidence_artifact_path_key,
            )
        ),
        *_candidate_artifact_paths(
            result.evidence,
            base_dir=base_dir,
            path_key_predicate=_is_typed_evidence_artifact_path_key,
        ),
        *_candidate_artifact_paths(
            artifact_metadata,
            base_dir=base_dir,
            path_key_predicate=_is_typed_evidence_artifact_path_key,
        ),
    )
    identities: list[ValidationArtifactIdentity] = []
    missing: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if not candidate.strip():
            if candidate not in seen:
                missing.append(candidate)
                seen.add(candidate)
            continue
        parsed = _parse_artifact_uri(candidate)
        if parsed is None:
            if candidate not in seen:
                missing.append(candidate)
                seen.add(candidate)
            continue
        is_windows_drive_path = _is_windows_drive_artifact_path(
            candidate,
            parsed=parsed,
        )
        if parsed.scheme and parsed.scheme != "file" and not is_windows_drive_path:
            if candidate not in seen:
                missing.append(candidate)
                seen.add(candidate)
            continue
        if parsed.scheme == "file" and parsed.netloc.lower() not in {"", "localhost"}:
            if candidate not in seen:
                missing.append(candidate)
                seen.add(candidate)
            continue
        local = unquote(parsed.path) if parsed.scheme == "file" else candidate
        if not local.strip():
            if candidate not in seen:
                missing.append(candidate)
                seen.add(candidate)
            continue
        try:
            identity = _artifact_identity(
                local,
                role="evidence",
                base_dir=base_dir,
            )
        except (OSError, RuntimeError, ValueError):
            identity = ValidationArtifactIdentity(
                role="evidence",
                path=_best_effort_resolved_path(local, base_dir=base_dir),
                kind="missing",
            )
        if identity.path in seen:
            continue
        if identity.kind == "missing":
            missing.append(identity.path)
            seen.add(identity.path)
            continue
        identities.append(identity)
        seen.add(identity.path)
    return tuple(identities), tuple(missing)


def _normalized_artifact_identities(
    artifacts: Sequence[ValidationArtifactIdentity],
) -> tuple[ValidationArtifactIdentity, ...]:
    return tuple(
        sorted(
            artifacts,
            key=lambda artifact: (
                artifact.path,
                artifact.kind,
                artifact.role,
                artifact.sha256 or "",
            ),
        )
    )


def _missing_declared_evidence_result(
    result: ValidationTemplateResult,
    missing_paths: Sequence[str],
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.accepted_evidence_missing",
        severity="fail",
        message=(
            f"{result.template_name} declared evidence that could not be "
            "content-validated when its result was accepted."
        ),
        template_name=result.template_name,
        details={"missing_evidence_paths": list(missing_paths)},
    )
    issues = (*result.issues, issue)
    metrics = dict(result.metrics)
    metrics["issue_count"] = len(issues)
    metadata = dict(result.metadata)
    metadata["declared_evidence_accepted"] = False
    return result.model_copy(
        update={
            "status": "failed",
            "issues": issues,
            "metrics": metrics,
            "metadata": metadata,
        }
    )


def _workflow_owned_evidence_artifacts(
    evidence: Sequence[ValidationArtifactIdentity],
    *,
    paths: ValidationWorkflowPaths,
    result_path: Path,
) -> tuple[
    tuple[ValidationArtifactIdentity, ...],
    tuple[ValidationArtifactIdentity, ...],
]:
    workflow_owned = {
        *(
            Path(path).expanduser().resolve()
            for path in paths.artifact_paths().values()
        ),
        paths.checkpoint.with_suffix(f"{paths.checkpoint.suffix}.lock"),
        paths.output_dir,
        paths.attempts,
        paths.output_dir / _RUN_OWNER_FILE_NAME,
        result_path.resolve(),
    }
    workflow_owned_files = {
        *(Path(path) for path in paths.artifact_paths().values()),
        paths.checkpoint.with_suffix(f"{paths.checkpoint.suffix}.lock"),
        paths.output_dir / _RUN_OWNER_FILE_NAME,
        result_path,
    }
    if paths.attempts.is_dir():
        workflow_owned_files.update(paths.attempts.rglob("template_result.json"))
    workflow_owned_file_inodes: set[tuple[int, int]] = set()
    for owned_file in workflow_owned_files:
        try:
            if owned_file.is_file():
                stat = owned_file.stat()
                workflow_owned_file_inodes.add((stat.st_dev, stat.st_ino))
        except OSError:
            continue
    accepted: list[ValidationArtifactIdentity] = []
    collisions: list[ValidationArtifactIdentity] = []
    result_path = result_path.resolve()
    for artifact in evidence:
        artifact_path = Path(artifact.path).expanduser().resolve()
        is_attempt_result = (
            artifact_path.name == "template_result.json"
            and paths.attempts in artifact_path.parents
        )
        contains_workflow_owned_artifact = artifact.kind == "directory" and any(
            artifact_path in owned_path.parents for owned_path in workflow_owned
        )
        aliases_workflow_owned_file = False
        try:
            if artifact.kind == "file" and artifact_path.is_file():
                stat = artifact_path.stat()
                aliases_workflow_owned_file = (
                    stat.st_dev,
                    stat.st_ino,
                ) in workflow_owned_file_inodes
            elif artifact.kind == "directory":
                aliases_workflow_owned_file = any(
                    (stat.st_dev, stat.st_ino) in workflow_owned_file_inodes
                    for member in artifact_path.rglob("*")
                    if member.is_file()
                    for stat in (member.stat(),)
                )
        except OSError:
            pass
        if (
            artifact_path in workflow_owned
            or is_attempt_result
            or contains_workflow_owned_artifact
            or aliases_workflow_owned_file
        ):
            collisions.append(artifact)
        else:
            accepted.append(artifact)
    return tuple(accepted), tuple(collisions)


def _workflow_owned_evidence_result(
    result: ValidationTemplateResult,
    collisions: Sequence[ValidationArtifactIdentity],
) -> ValidationTemplateResult:
    issue = ValidationIssue(
        code="validation.workflow_artifact_evidence_collision",
        severity="fail",
        message=(
            f"{result.template_name} declared a mutable workflow-owned artifact "
            "as validation evidence."
        ),
        template_name=result.template_name,
        details={
            "workflow_artifact_paths": [artifact.path for artifact in collisions],
        },
    )
    issues = (*result.issues, issue)
    metrics = dict(result.metrics)
    metrics["issue_count"] = len(issues)
    metadata = dict(result.metadata)
    metadata["declared_evidence_accepted"] = False
    return result.model_copy(
        update={
            "status": "failed",
            "issues": issues,
            "metrics": metrics,
            "metadata": metadata,
        }
    )


def _commit_template_result(
    store: ValidationCheckpointStore,
    *,
    record: ValidationWorkItemRecord,
    result: ValidationTemplateResult,
    result_path: Path,
    evidence_artifacts: tuple[ValidationArtifactIdentity, ...],
    token: ValidationCancellationToken,
    external_cancellation_check: ExternalCancellationCheck | None,
) -> tuple[ValidationWorkflowCheckpoint, bool]:
    accepted = ValidationAcceptedTemplateResult(
        work_item_identity_digest=record.identity_digest,
        attempt=record.attempts,
        result=result,
        result_path=str(result_path),
        result_sha256=file_sha256(result_path),
        evidence_artifacts=evidence_artifacts,
    )
    committed = False

    def mutation(
        current: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowCheckpoint:
        nonlocal committed
        latest = _checkpoint_record(current, record.work_item_id)
        if latest.attempts != record.attempts:
            raise ValidationCheckpointError(
                f"Late result for stale attempt of {record.work_item_id}"
            )
        cancelled = current.cancellation_requested or _cancellation_requested(
            token,
            external_cancellation_check,
        )
        if cancelled:
            updated = latest.model_copy(
                update={
                    "state": ValidationWorkItemState.CANCELLED,
                    "accepted_result": None,
                    "last_error": (
                        current.cancellation_reason
                        or "Validation template completed after cancellation."
                    ),
                    "finished_at": datetime.now(UTC),
                }
            )
            return _replace_record(
                current,
                updated,
                cancellation_requested=True,
                cancellation_reason=(
                    current.cancellation_reason
                    or "Validation workflow cancellation requested."
                ),
            )
        if latest.state != ValidationWorkItemState.RUNNING:
            raise ValidationCheckpointError(
                f"Late result for inactive attempt of {record.work_item_id}"
            )
        committed = True
        updated = latest.model_copy(
            update={
                "state": ValidationWorkItemState.COMPLETED,
                "accepted_result": accepted,
                "last_error": None,
                "finished_at": datetime.now(UTC),
            }
        )
        return _replace_record(current, updated)

    return store.update(mutation), committed


def _previous_results(
    checkpoint: ValidationWorkflowCheckpoint,
    *,
    before_template: str,
) -> tuple[ValidationTemplateResult, ...]:
    results: list[ValidationTemplateResult] = []
    for record in checkpoint.records:
        if record.template_name == before_template:
            break
        if record.accepted_result is not None:
            results.append(record.accepted_result.result)
    return tuple(results)


def _raw_result(
    *,
    status: ValidationWorkflowStatus,
    request: ValidationRequest,
    plan: ValidationPlan,
    checkpoint: ValidationWorkflowCheckpoint,
    source_before: tuple[ValidationArtifactIdentity, ...],
    source_after: tuple[ValidationArtifactIdentity, ...],
    final_integrity_issues: tuple[ValidationIssue, ...] = (),
) -> ValidationResult:
    template_results = tuple(
        record.accepted_result.result
        for record in checkpoint.records
        if record.accepted_result is not None
    )
    issues = tuple(
        issue
        for template_result in template_results
        for issue in template_result.issues
    )
    issues = (*issues, *final_integrity_issues)
    verdict = aggregate_validation_verdict(template_results)
    recommended_action: str | None
    if final_integrity_issues:
        verdict = "fail"
        recommended_action = (
            "Restore the source asset and regenerate any missing or changed "
            "validation evidence before rerunning validation."
        )
    elif status == ValidationWorkflowStatus.CANCELLED:
        cancellation_issue = ValidationIssue(
            code="validation.workflow_cancelled",
            severity="warn",
            message=(
                "Validation was cancelled after preserving all completed "
                "template evidence. Resume the run to continue."
            ),
        )
        issues = (*issues, cancellation_issue)
        if verdict in {"pass", "planned"}:
            verdict = "warn"
        recommended_action = "Resume this run to execute the next unfinished check."
    elif verdict == "pass" and any(
        result.template_name == "look_right" and result.passed
        for result in template_results
    ):
        recommended_action = (
            "The asset rendered successfully and matched the supplied visual "
            "reference; no validation action is required."
        )
    elif verdict == "pass":
        recommended_action = (
            "The asset rendered successfully; no visual-reference comparison "
            "was requested, so no render action is required."
        )
    elif verdict == "needs_refinement":
        recommended_action = (
            "Review the visual findings and refine the generated asset before "
            "rerunning validation."
        )
    elif verdict == "fail":
        recommended_action = (
            "Resolve the failed render, evidence, or visual-reference checks "
            "and rerun validation."
        )
    elif verdict == "warn":
        recommended_action = (
            "Review the validation warnings and configure any unavailable "
            "renderer, VLM, or reference evidence before release."
        )
    else:
        recommended_action = "Run the planned validation checks."

    metadata: dict[str, object] = {
        "workflow_status": status.value,
        "workflow_identity_digest": (checkpoint.workflow_identity.identity_digest),
        "plan_digest": checkpoint.plan_digest,
        "checkpoint_revision": checkpoint.revision,
        "source_asset_unchanged": source_before == source_after,
        "reference_evidence_unchanged": not any(
            issue.code == "validation.reference_evidence_stale"
            for issue in final_integrity_issues
        ),
    }
    execution_context = domain_execution_context_from_metadata(
        request.metadata,
        expected_domain="validation",
    )
    if execution_context is not None and execution_context.mode == "embedded":
        # Preserve the native V1 report shape and verdict for factual inspection,
        # but make its authority boundary explicit. In embedded mode look_right
        # remains a critique proposal and this aggregate is never composed-stage
        # completion authority.
        metadata.update(
            {
                "embedded_execution": True,
                "native_verdict_role": "factual_evidence_only",
                "look_right_role": "proposal_or_critique_only",
                "semantic_completion_authority": (
                    "completed_embedded_validation_decision_receipt"
                ),
            }
        )

    return ValidationResult(
        verdict=verdict,
        request=request,
        plan=plan,
        template_results=template_results,
        issues=issues,
        metrics={result.template_name: result.metrics for result in template_results},
        evidence={result.template_name: result.evidence for result in template_results},
        recommended_action=recommended_action,
        metadata=metadata,
    )


def run_validation_workflow(
    request: ValidationRequest,
    *,
    output_dir: str | Path,
    config_base_dir: str | Path,
    executor: ValidationStepExecutor | None = None,
    resume: bool = False,
    cancellation_token: ValidationCancellationToken | None = None,
    external_cancellation_check: ExternalCancellationCheck | None = None,
    progress_callback: ProgressCallback | None = None,
    checkpoint_store: ValidationCheckpointStore | None = None,
) -> ValidationWorkflowRun:
    """Run or resume the ordered ``render_valid`` -> ``look_right`` workflow."""

    if os.name == "posix" and sys.platform != "linux":
        raise ValidationWorkflowError(
            "Resumable Validation workflow execution requires Linux, a Linux "
            "container, or WSL2 for descriptor-backed attempt isolation."
        )
    paths = ValidationWorkflowPaths.from_output_dir(output_dir)
    if checkpoint_store is None:
        _validate_output_artifact_locations(paths)
    store = checkpoint_store or FileValidationCheckpointStore(paths.checkpoint)
    finalize_method = getattr(store, "finalize", None)
    inherited_protocol_stub = (
        getattr(finalize_method, "__func__", None) is ValidationCheckpointStore.finalize
    )
    if not callable(finalize_method) or inherited_protocol_stub:
        raise ValidationWorkflowError(
            "Validation checkpoint_store must implement atomic finalize(callback) "
            "before the workflow can execute."
        )
    selected_checkpoint = store.path.expanduser().absolute()
    ensure_no_inline_secrets(
        selected_checkpoint,
        context="validation checkpoint store path",
    )
    paths = replace(paths, checkpoint=selected_checkpoint)
    _validate_output_artifact_locations(paths)
    paths = replace(paths, checkpoint=selected_checkpoint.resolve())
    _validate_checkpoint_artifact_collision(paths)
    base_dir = Path(config_base_dir).expanduser().resolve()
    effective_request = _effective_request(request, output_dir=paths.output_dir)
    ensure_no_inline_secrets(
        effective_request.model_dump(mode="json"),
        context="validation workflow request",
    )
    executor_impl = executor or ScaffoldValidationStepExecutor(base_dir)
    resume_checkpoint_exists = resume and store.path.is_file()
    try:
        identity = _workflow_identity(
            effective_request,
            config_base_dir=base_dir,
            executor=executor_impl,
        )
    except _ValidationArtifactFingerprintError as exc:
        if resume_checkpoint_exists:
            raise ValidationWorkflowIdentityMismatch(
                "Validation checkpoint input identity cannot be reproduced "
                "because a USD dependency closure changed or became invalid."
            ) from exc
        raise
    _validate_source_artifacts(
        identity,
        resume_checkpoint_exists=resume_checkpoint_exists,
    )
    _validate_input_artifact_collisions(paths, identity)
    _validate_output_location(paths.output_dir, identity)
    _validate_checkpoint_location(paths.checkpoint, identity)
    existing = store.load()
    if resume:
        if existing is None:
            raise ValidationCheckpointError(
                "Cannot resume because no validation checkpoint exists."
            )
    elif existing is not None:
        raise ValidationCheckpointError(
            "Validation run already exists; pass resume=True or use a new "
            "output directory."
        )
    _claim_output_directory(paths)
    output_dir_stat = paths.output_dir.stat()
    output_dir_identity = (output_dir_stat.st_dev, output_dir_stat.st_ino)
    plan = _bind_plan(
        executor_impl.plan(
            effective_request.model_copy(deep=True),
            working_dir=paths.output_dir,
        ),
        identity=identity,
        artifact_paths=paths.artifact_paths(),
    )
    ensure_no_inline_secrets(
        plan.model_dump(mode="json"),
        context="validation workflow plan",
    )
    plan_digest = _plan_digest(plan)
    expected_records = _records_from_plan(plan)

    def drop_terminal_bundle() -> None:
        _invalidate_terminal_bundle(
            paths,
            output_dir_identity=output_dir_identity,
        )

    if resume:
        assert existing is not None
        checkpoint = _prepare_resume_checkpoint(
            store,
            paths=paths,
            identity=identity,
            plan_digest=plan_digest,
            expected_records=expected_records,
            invalidate_terminal_bundle=drop_terminal_bundle,
        )
    else:
        checkpoint = store.create(
            ValidationWorkflowCheckpoint(
                workflow_identity=identity,
                plan_digest=plan_digest,
                ordered_work_item_ids=tuple(
                    record.work_item_id for record in expected_records
                ),
                records=expected_records,
            )
        )
    if any(record.accepted_result is None for record in checkpoint.records):
        drop_terminal_bundle()
    write_validation_planning_artifacts(
        effective_request,
        plan,
        paths,
        output_dir_identity=output_dir_identity,
    )
    _report(checkpoint, progress_callback)

    token = cancellation_token or ValidationCancellationToken()
    source_before = identity.source_artifacts
    reference_paths = _reference_paths(effective_request.policy)
    for planned_record in expected_records:
        loaded_checkpoint = store.load()
        if loaded_checkpoint is None:
            raise ValidationCheckpointError("Validation checkpoint disappeared.")
        checkpoint = loaded_checkpoint
        current_record = _checkpoint_record(checkpoint, planned_record.work_item_id)
        if current_record.accepted_result is not None:
            continue
        if checkpoint.cancellation_requested or _cancellation_requested(
            token, external_cancellation_check
        ):
            checkpoint = _cancel_remaining(
                store,
                reason=(
                    checkpoint.cancellation_reason
                    or "Validation workflow cancellation requested."
                ),
            )
            _report(checkpoint, progress_callback)
            break

        checkpoint, claimed = _mark_running(store, planned_record.work_item_id)
        if not claimed:
            _report(checkpoint, progress_callback)
            if checkpoint.cancellation_requested:
                break
            if (
                _checkpoint_record(
                    checkpoint, planned_record.work_item_id
                ).accepted_result
                is not None
            ):
                continue
            raise ValidationCheckpointError(
                f"Could not claim {planned_record.work_item_id} for execution."
            )
        current_record = _checkpoint_record(checkpoint, planned_record.work_item_id)
        try:
            _report(checkpoint, progress_callback)
        except BaseException:
            _release_unstarted_claim(
                store,
                record=current_record,
            )
            raise
        latest_checkpoint = store.load()
        if latest_checkpoint is None:
            raise ValidationCheckpointError("Validation checkpoint disappeared.")
        checkpoint = latest_checkpoint
        if checkpoint.cancellation_requested or _cancellation_requested(
            token, external_cancellation_check
        ):
            cancellation_reason = (
                checkpoint.cancellation_reason
                or "Validation workflow cancellation requested."
            )
            if not checkpoint.cancellation_requested:
                checkpoint = _cancel_remaining(
                    store,
                    reason=cancellation_reason,
                )
            checkpoint = _acknowledge_cancelled_attempt(
                store,
                record=current_record,
                reason=cancellation_reason,
            )
            _report(checkpoint, progress_callback)
            break
        current_record = _checkpoint_record(
            checkpoint,
            planned_record.work_item_id,
        )
        attempt_dir = (
            paths.attempts
            / current_record.template_name
            / f"attempt-{current_record.attempts:04d}"
        )
        _validate_attempt_artifact_location(attempt_dir, paths=paths)
        try:
            attempt_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise ValidationWorkflowError(
                "Validation attempt directory already exists and cannot be reused: "
                f"{attempt_dir}"
            ) from exc
        attempt_stat = attempt_dir.stat()
        attempt_identity = (attempt_stat.st_dev, attempt_stat.st_ino)
        previous_results = _previous_results(
            checkpoint,
            before_template=current_record.template_name,
        )

        def execute_current_template() -> ValidationTemplateResult:
            with _pinned_attempt_working_directory(
                attempt_dir,
                paths=paths,
                expected_identity=attempt_identity,
            ) as working_dir:
                return _canonicalize_result_artifact_paths(
                    executor_impl.run(
                        current_record.template_name,
                        ValidationTemplateContext(
                            request=effective_request.model_copy(deep=True),
                            plan=plan.model_copy(deep=True),
                            working_dir=working_dir,
                            previous_template_results=tuple(
                                previous_result.model_copy(deep=True)
                                for previous_result in previous_results
                            ),
                        ),
                    ),
                    base_dir=working_dir,
                )

        result: ValidationTemplateResult
        if current_record.template_name == "look_right":
            missing_reference = _missing_reference_result(identity, effective_request)
            render_record = next(
                record
                for record in checkpoint.records
                if record.template_name == "render_valid"
            )
            render_accepted = render_record.accepted_result
            render_handoff_valid = (
                render_accepted is not None
                and render_accepted.result.passed
                and _accepted_result_is_valid(
                    render_accepted,
                    paths=paths,
                    expected_template_name=render_record.template_name,
                    expected_attempt=render_record.attempts,
                    forbidden_artifacts=(
                        *identity.source_artifacts,
                        *identity.reference_artifacts,
                    ),
                )
            )
            current_references = _best_effort_artifact_identities(
                reference_paths,
                role="reference",
                base_dir=base_dir,
            )
            if missing_reference is not None:
                result = missing_reference
            elif current_references != identity.reference_artifacts:
                result = _stale_reference_result(
                    identity.reference_artifacts,
                    current_references,
                )
            elif (
                render_accepted is not None
                and render_accepted.result.status == "skipped"
            ):
                result = _skipped_render_handoff_result(render_accepted.result)
            elif (
                render_accepted is not None
                and not render_accepted.result.passed
                and _accepted_result_is_valid(
                    render_accepted,
                    paths=paths,
                    expected_template_name=render_record.template_name,
                    expected_attempt=render_record.attempts,
                    forbidden_artifacts=(
                        *identity.source_artifacts,
                        *identity.reference_artifacts,
                    ),
                )
            ):
                result = _nonpassing_render_handoff_result(render_accepted.result)
            elif not render_handoff_valid:
                result = _stale_render_handoff_result()
            else:
                assert render_accepted is not None
                try:
                    result = execute_current_template()
                except Exception as exc:
                    result = _template_execution_error(
                        current_record.template_name, exc
                    )
                current_references = _best_effort_artifact_identities(
                    reference_paths,
                    role="reference",
                    base_dir=base_dir,
                )
                if current_references != identity.reference_artifacts:
                    result = _stale_reference_result(
                        identity.reference_artifacts,
                        current_references,
                        result=result,
                    )
                if not _accepted_result_is_valid(
                    render_accepted,
                    paths=paths,
                    expected_template_name=render_record.template_name,
                    expected_attempt=render_record.attempts,
                    forbidden_artifacts=(
                        *identity.source_artifacts,
                        *identity.reference_artifacts,
                    ),
                ):
                    result = _stale_render_handoff_result(result)
        else:
            try:
                result = execute_current_template()
            except Exception as exc:
                result = _template_execution_error(current_record.template_name, exc)
            result = _validate_render_evidence_result(
                result,
                base_dir=attempt_dir,
                qualified_evidence_base_dir=base_dir,
                request=effective_request,
                identity=identity,
                forbidden_artifacts=(
                    *identity.source_artifacts,
                    *identity.reference_artifacts,
                ),
            )

        try:
            _validate_attempt_artifact_location(attempt_dir, paths=paths)
            current_attempt_stat = attempt_dir.stat()
            if (
                current_attempt_stat.st_dev,
                current_attempt_stat.st_ino,
            ) != attempt_identity:
                raise ValidationWorkflowError(
                    f"Validation attempt directory identity changed: {attempt_dir}"
                )
        except (OSError, ValidationWorkflowError):
            cancellation_reason = (
                f"Validation attempt directory identity changed during "
                f"{current_record.template_name} execution."
            )
            checkpoint = _cancel_remaining(
                store,
                reason=cancellation_reason,
            )
            checkpoint = _acknowledge_cancelled_attempt(
                store,
                record=current_record,
                reason=cancellation_reason,
            )
            _report(checkpoint, progress_callback)
            break
        if result.template_name != current_record.template_name:
            result = _template_result_name_mismatch(
                current_record.template_name,
                result.template_name,
            )
        result = _canonicalize_result_artifact_paths(
            result,
            base_dir=attempt_dir,
        )

        source_after_attempt = _best_effort_artifact_identities(
            effective_request.inputs,
            role="source",
            base_dir=base_dir,
        )
        if source_after_attempt != source_before:
            result = _source_modified_result(
                current_record.template_name,
                source_before,
                source_after_attempt,
            )

        try:
            ensure_no_inline_secrets(
                result.model_dump(mode="json"),
                context=f"{current_record.template_name} validation result",
            )
        except InlineSecretError as exc:
            result = _inline_credential_result(
                current_record.template_name,
                exc,
            )
        candidate_evidence, missing_evidence = _evidence_artifacts(
            result,
            base_dir=attempt_dir,
        )
        if missing_evidence:
            result = _missing_declared_evidence_result(result, missing_evidence)
        result_path = attempt_dir / "template_result.json"
        _validate_attempt_artifact_location(result_path, paths=paths)
        _, workflow_owned_evidence = _workflow_owned_evidence_artifacts(
            candidate_evidence,
            paths=paths,
            result_path=result_path,
        )
        if workflow_owned_evidence:
            result = _workflow_owned_evidence_result(
                result,
                workflow_owned_evidence,
            )
        if os.name == "posix":
            with _pinned_directory_fd(
                attempt_dir,
                expected_identity=attempt_identity,
                description="attempt directory",
            ) as attempt_fd:
                atomic_write_json_at(attempt_fd, result_path.name, result)
        else:  # pragma: win32 cover
            atomic_write_json(result_path, result)
        evidence, _ = _evidence_artifacts(result, base_dir=attempt_dir)
        evidence, _ = _workflow_owned_evidence_artifacts(
            evidence,
            paths=paths,
            result_path=result_path,
        )
        checkpoint, committed = _commit_template_result(
            store,
            record=current_record,
            result=result,
            result_path=result_path,
            evidence_artifacts=evidence,
            token=token,
            external_cancellation_check=external_cancellation_check,
        )
        _report(checkpoint, progress_callback)
        if not committed:
            break

    verification_output_dir_fd: int | None = None

    def finalize_checkpoint(
        checkpoint: ValidationWorkflowCheckpoint,
    ) -> ValidationWorkflowRun:
        unfinished_work_item_ids = tuple(
            record.work_item_id
            for record in checkpoint.records
            if record.accepted_result is None
        )
        if unfinished_work_item_ids and not checkpoint.cancellation_requested:
            raise ValidationCheckpointError(
                "Validation checkpoint has unfinished work and cannot be finalized; "
                "resume the run to complete: " + ", ".join(unfinished_work_item_ids)
            )
        status = (
            ValidationWorkflowStatus.CANCELLED
            if checkpoint.cancellation_requested
            else ValidationWorkflowStatus.COMPLETED
        )
        _validate_output_artifact_locations(paths)

        def integrity_snapshot() -> tuple[
            tuple[ValidationArtifactIdentity, ...],
            tuple[ValidationArtifactIdentity, ...],
            tuple[ValidationIssue, ...],
        ]:
            current_sources = _best_effort_artifact_identities(
                effective_request.inputs,
                role="source",
                base_dir=base_dir,
            )
            current_references = _best_effort_artifact_identities(
                reference_paths,
                role="reference",
                base_dir=base_dir,
            )
            return (
                current_sources,
                current_references,
                _final_integrity_issues(
                    checkpoint,
                    paths=paths,
                    source_before=source_before,
                    source_after=current_sources,
                    reference_before=identity.reference_artifacts,
                    reference_after=current_references,
                ),
            )

        snapshot = integrity_snapshot()
        for _ in range(2):
            source_after, _, final_integrity_issues = snapshot
            raw_result = _raw_result(
                status=status,
                request=effective_request,
                plan=plan,
                checkpoint=checkpoint,
                source_before=source_before,
                source_after=source_after,
                final_integrity_issues=final_integrity_issues,
            )
            run = finalize_validation_workflow(
                status=status,
                request=effective_request,
                plan=plan,
                raw_result=raw_result,
                checkpoint=checkpoint,
                source_before=source_before,
                source_after=source_after,
                paths=paths,
                output_dir_identity=output_dir_identity,
            )
            verified_snapshot = integrity_snapshot()
            if (
                verified_snapshot == snapshot
                and finalized_validation_artifacts_are_valid(
                    run,
                    checkpoint=checkpoint,
                    source_before=source_before,
                    source_after=source_after,
                    paths=paths,
                    output_dir_fd=verification_output_dir_fd,
                )
            ):
                return run
            snapshot = verified_snapshot
        source_after, _, final_integrity_issues = snapshot
        unstable_issue = ValidationIssue(
            code="validation.artifact_publication_unstable",
            severity="fail",
            message=(
                "Validation inputs, evidence, or report artifacts kept changing "
                "while the final report bundle was being published."
            ),
        )
        forced_issues = (*final_integrity_issues, unstable_issue)
        for _ in range(2):
            forced_run = finalize_validation_workflow(
                status=status,
                request=effective_request,
                plan=plan,
                raw_result=_raw_result(
                    status=status,
                    request=effective_request,
                    plan=plan,
                    checkpoint=checkpoint,
                    source_before=source_before,
                    source_after=source_after,
                    final_integrity_issues=forced_issues,
                ),
                checkpoint=checkpoint,
                source_before=source_before,
                source_after=source_after,
                paths=paths,
                output_dir_identity=output_dir_identity,
            )
            if finalized_validation_artifacts_are_valid(
                forced_run,
                checkpoint=checkpoint,
                source_before=source_before,
                source_after=source_after,
                paths=paths,
                output_dir_fd=verification_output_dir_fd,
            ):
                break
        else:
            if os.name == "posix":
                with suppress(OSError, ValidationWorkflowError):
                    with _pinned_directory_fd(
                        paths.output_dir,
                        expected_identity=output_dir_identity,
                        description="output directory",
                    ) as output_dir_fd:
                        for unsafe_pass_artifact in (
                            paths.result,
                            paths.evidence,
                            paths.final_summary,
                        ):
                            with suppress(OSError):
                                os.unlink(
                                    unsafe_pass_artifact.name,
                                    dir_fd=output_dir_fd,
                                )
            else:  # pragma: win32 cover
                for unsafe_pass_artifact in (
                    paths.result,
                    paths.evidence,
                    paths.final_summary,
                ):
                    with suppress(OSError):
                        unsafe_pass_artifact.unlink()
        raise ValidationCheckpointError(
            "Validation inputs or accepted evidence kept changing while final "
            "artifacts were being published."
        )

    terminal_artifact_names = (
        paths.result.name,
        paths.evidence.name,
        paths.final_summary.name,
    )
    if os.name == "posix":
        with _pinned_directory_fd(
            paths.output_dir,
            expected_identity=output_dir_identity,
            description="output directory",
            cleanup_names_on_error=terminal_artifact_names,
        ) as output_dir_fd:
            verification_output_dir_fd = output_dir_fd
            try:
                return store.finalize(finalize_checkpoint)
            finally:
                verification_output_dir_fd = None
    try:  # pragma: win32 cover
        return store.finalize(finalize_checkpoint)
    except BaseException:  # pragma: win32 cover
        for unsafe_terminal_artifact in (
            paths.result,
            paths.evidence,
            paths.final_summary,
        ):
            with suppress(OSError):
                unsafe_terminal_artifact.unlink()
        raise
