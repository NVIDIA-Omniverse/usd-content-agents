# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trusted execution boundary for one frozen asset-graph leaf.

The reasoning child authors and binds an invocation, while this installed
repository entrypoint resolves the exact frozen descriptor and invokes it.  A
managed child therefore needs permission for only the asset-state executable;
domain executables do not become ambient shell capabilities.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Literal, cast

from filelock import FileLock, Timeout
from pydantic import BaseModel, ConfigDict
from world_understanding.utils.artifacts import (
    ArtifactPathError,
    open_confined_directory,
    open_confined_regular_file,
)

from content_agent_workflows.common.artifacts import (
    _stable_ctime_ns,
    atomic_write_json,
)

from .catalog import repository_asset_leaf_runtime_catalog
from .models import ArtifactBinding, AssetCompositionRun
from .state import (
    AssetCompositionStateError,
    _exclusive_lock,
    _load_execution_graph,
    leaf_directory,
    load_verified_run,
)

_MAX_INVOCATION_BYTES = 4 * 1024 * 1024
_MAX_RESULT_BYTES = 16 * 1024 * 1024
_CONTENT_WORKFLOW_CLI = "content-workflow-cli"
_COMPOSED_LEAF_CLI = "content-workflow-composed-leaf"
_ADMITTED_CLI_EXECUTABLES = frozenset({_CONTENT_WORKFLOW_CLI, _COMPOSED_LEAF_CLI})
_COMPOSED_LEAF_OPERATIONS = frozenset(
    {
        "combined",
        "material",
        "package",
        "physics-apply",
        "physics-inspect",
        "simready-conform",
        "simready-validate",
    }
)
_CANONICAL_OVRTX_ENTRYPOINT = (
    "content_agent_workflows.validation.produce_canonical_visual_evidence"
)
_FOCUSED_VALIDATION_ENTRYPOINT = (
    "content_agent_workflows.validation.run_validation_operation"
)
_PROVIDED_VALIDATION_ENTRYPOINT = (
    "content_agent_workflows.validation.ingest_verified_operation_result"
)


class _LeafExecutionRecord(BaseModel):
    """Durable identity of one native execution sealed for an attempt."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[
        "content-agent-workflows.asset-leaf-execution-record.v1"
    ] = "content-agent-workflows.asset-leaf-execution-record.v1"
    run_id: str
    leaf_id: str
    attempt: int
    invocation: ArtifactBinding
    result: ArtifactBinding


def _confined_regular_file(
    path: Path,
    *,
    within: Path,
    label: str,
    max_bytes: int = _MAX_INVOCATION_BYTES,
) -> bytes:
    """Read one bounded, canonical, single-link file below ``within``."""

    if not path.is_absolute():
        raise AssetCompositionStateError(f"{label} path must be absolute")
    try:
        candidate = path.resolve(strict=True)
        root = within.resolve(strict=True)
        relative_key = candidate.relative_to(root).as_posix()
        if candidate != path:
            raise ValueError("path is not canonical")
        with open_confined_directory(root) as root_descriptor:
            with open_confined_regular_file(
                root_descriptor,
                relative_key,
            ) as (stream, metadata):
                current = candidate.lstat()
                identity = (metadata.st_dev, metadata.st_ino)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or not stat.S_ISREG(current.st_mode)
                    or metadata.st_nlink != 1
                    or current.st_nlink != 1
                    or (current.st_dev, current.st_ino) != identity
                ):
                    raise AssetCompositionStateError(
                        f"{label} must be a canonical single-link regular file"
                    )
                if metadata.st_size > max_bytes:
                    raise AssetCompositionStateError(
                        f"{label} exceeds {max_bytes} bytes"
                    )
                document = cast(bytes, stream.read(max_bytes + 1))
                after = os.fstat(stream.fileno())
                current = candidate.lstat()
                before_state = (
                    metadata.st_dev,
                    metadata.st_ino,
                    metadata.st_mode,
                    metadata.st_nlink,
                    metadata.st_size,
                    metadata.st_mtime_ns,
                    _stable_ctime_ns(metadata),
                )
                after_state = (
                    after.st_dev,
                    after.st_ino,
                    after.st_mode,
                    after.st_nlink,
                    after.st_size,
                    after.st_mtime_ns,
                    _stable_ctime_ns(after),
                )
                current_state = (
                    current.st_dev,
                    current.st_ino,
                    current.st_mode,
                    current.st_nlink,
                    current.st_size,
                    current.st_mtime_ns,
                    _stable_ctime_ns(current),
                )
                if (
                    len(document) != metadata.st_size
                    or len(document) > max_bytes
                    or before_state != after_state
                    or before_state != current_state
                ):
                    raise AssetCompositionStateError(f"{label} changed while read")
        return document
    except AssetCompositionStateError:
        raise
    except (ArtifactPathError, OSError, ValueError) as exc:
        raise AssetCompositionStateError(
            f"{label} must be a canonical file below the active leaf attempt"
        ) from exc


def _result_destination(
    path: Path,
    *,
    within: Path,
    require_new: bool = True,
) -> Path:
    """Validate one new result destination in the active attempt directory."""

    if not path.is_absolute():
        raise AssetCompositionStateError("leaf result path must be absolute")
    try:
        root = within.resolve(strict=True)
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise AssetCompositionStateError("leaf result parent is unavailable") from exc
    if path.parent != parent or parent != root:
        raise AssetCompositionStateError(
            "leaf result must be a canonical direct child of the active attempt"
        )
    if require_new and (path.exists() or path.is_symlink()):
        raise AssetCompositionStateError("leaf result already exists")
    if path.suffix.lower() != ".json":
        raise AssetCompositionStateError("leaf result must be a JSON artifact")
    return path


def _leaf_execution_paths(
    state_path: Path,
    *,
    leaf_id: str,
    attempt: int,
) -> tuple[Path, Path]:
    """Keep the execution lease and record outside the child-writable run."""

    run_dir = state_path.parent
    identity = hashlib.sha256(
        f"{state_path}\0{leaf_id}\0{attempt}".encode()
    ).hexdigest()[:20]
    lease_path = run_dir.parent / (
        f".{run_dir.name}.{identity}.asset-leaf-attempt-{attempt}.lock"
    )
    return lease_path, lease_path.with_suffix(".result.json")


def _acquire_leaf_execution_lease(path: Path) -> FileLock:
    """Fail fast when another process already executes this exact attempt."""

    lease = FileLock(str(path))
    try:
        lease.acquire(timeout=0)
    except Timeout as exc:
        raise AssetCompositionStateError(
            "Another executor already owns this active leaf attempt"
        ) from exc
    return lease


def _artifact_binding(path: Path, document: bytes) -> ArtifactBinding:
    return ArtifactBinding(
        path=str(path),
        sha256=hashlib.sha256(document).hexdigest(),
        size_bytes=len(document),
    )


def _installed_leaf_executable(name: str) -> Path:
    """Resolve one exact admitted launcher without accepting caller-selected code."""

    if name not in _ADMITTED_CLI_EXECUTABLES:
        raise AssetCompositionStateError("asset leaf executable is not admitted")
    executable = shutil.which(name)
    if executable is None and os.name == "nt":
        adjacent = Path(sys.executable).with_name(f"{name}.exe")
        if adjacent.is_file():
            executable = str(adjacent)
    if executable is None:
        raise AssetCompositionStateError(f"{name} is unavailable")
    try:
        resolved = Path(executable).resolve(strict=True)
    except OSError as exc:
        raise AssetCompositionStateError(f"{name} executable is unavailable") from exc
    if not resolved.is_file():
        raise AssetCompositionStateError(f"{name} executable is not a regular file")
    return resolved


def _content_workflow_cli() -> Path:
    """Resolve the installed outer workflow launcher."""

    return _installed_leaf_executable(_CONTENT_WORKFLOW_CLI)


def _content_workflow_composed_leaf() -> Path:
    """Resolve the installed composed-leaf launcher."""

    return _installed_leaf_executable(_COMPOSED_LEAF_CLI)


def _catalog_cli_command(entrypoint: str) -> tuple[Path, list[str]]:
    """Validate one exact catalog CLI form and select its installed executable."""

    arguments = shlex.split(entrypoint, posix=True)
    if len(arguments) < 3 or arguments[-1] != "--invocation":
        raise AssetCompositionStateError(
            "asset leaf catalog CLI entrypoint is not an admitted invocation form"
        )
    if arguments[0] == _CONTENT_WORKFLOW_CLI:
        executable = _content_workflow_cli()
    elif (
        arguments[0] == _COMPOSED_LEAF_CLI
        and len(arguments) == 3
        and arguments[1] in _COMPOSED_LEAF_OPERATIONS
    ):
        executable = _content_workflow_composed_leaf()
    else:
        raise AssetCompositionStateError(
            "asset leaf catalog CLI entrypoint is not an admitted invocation form"
        )
    return executable, arguments[1:]


def _execute_cli_entrypoint(
    entrypoint: str,
    *,
    invocation_path: Path,
    result_model: type[BaseModel],
) -> BaseModel:
    """Invoke one exact catalog CLI entrypoint and validate its JSON result."""

    executable, arguments = _catalog_cli_command(entrypoint)
    completed = subprocess.run(
        [str(executable), *arguments, str(invocation_path)],
        cwd=str(invocation_path.parent),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=3600,
        check=False,
    )
    encoded = completed.stdout.encode("utf-8")
    validation_error: ValueError | None = None
    if encoded and len(encoded) <= _MAX_RESULT_BYTES:
        try:
            return result_model.model_validate_json(encoded)
        except ValueError as exc:
            validation_error = exc
    if completed.returncode != 0:
        raise AssetCompositionStateError(
            "frozen asset leaf entrypoint failed with exit code "
            f"{completed.returncode} without a valid typed result"
        ) from validation_error
    if not encoded or len(encoded) > _MAX_RESULT_BYTES:
        raise AssetCompositionStateError("frozen asset leaf returned unsafe JSON size")
    raise AssetCompositionStateError(
        "frozen asset leaf returned invalid typed JSON"
    ) from validation_error


def _execute_shared_entrypoint(
    entrypoint: str,
    *,
    invocation: BaseModel,
    result_model: type[BaseModel],
) -> BaseModel:
    """Invoke one repository-owned shared Validation boundary."""

    from content_agent_workflows.validation import (
        ingest_verified_operation_result,
        produce_canonical_visual_evidence,
        run_validation_operation,
    )

    values = invocation.model_dump(mode="python")
    if entrypoint == _CANONICAL_OVRTX_ENTRYPOINT:
        result = produce_canonical_visual_evidence(**values)
    elif entrypoint == _FOCUSED_VALIDATION_ENTRYPOINT:
        output_dir = values.pop("output_dir")
        result = run_validation_operation(output_dir, **values)
    elif entrypoint == _PROVIDED_VALIDATION_ENTRYPOINT:
        envelope_path = values.pop("envelope_path")
        result = ingest_verified_operation_result(envelope_path, **values)
    else:  # pragma: no cover - caller selects this only for known shared IDs
        raise AssetCompositionStateError(
            f"asset leaf has no trusted shared executor: {entrypoint}"
        )
    return result_model.model_validate(result.model_dump(mode="json"))


def execute_active_leaf(
    run_state_path: str | Path,
    leaf_id: str,
    *,
    invocation_path: str | Path,
    result_path: str | Path,
) -> ArtifactBinding:
    """Execute exactly the frozen active leaf and seal its typed result."""

    state_path = Path(run_state_path).expanduser().resolve(strict=True)
    snapshot = load_verified_run(state_path)
    state = snapshot.leaf_states.get(leaf_id)
    if (
        snapshot.selected_mode != "agentic"
        or snapshot.execution_graph is None
        or snapshot.coordinator.next_action != "execute_leaf"
        or snapshot.current_leaf_id != leaf_id
        or state is None
        or state.status != "running"
    ):
        raise AssetCompositionStateError(
            "invoke-leaf requires the exact active running graph leaf"
        )
    attempt = state.attempt_count
    lease_path, record_path = _leaf_execution_paths(
        state_path,
        leaf_id=leaf_id,
        attempt=attempt,
    )
    lease = _acquire_leaf_execution_lease(lease_path)
    try:
        # Every leaf transition already uses this same state lock. Holding it
        # across validation, native execution, and result sealing prevents a
        # concurrent fail/cancel/recover from making the invocation stale.
        with _exclusive_lock(state_path):
            run = load_verified_run(state_path)
            current = run.leaf_states.get(leaf_id)
            if (
                run.selected_mode != "agentic"
                or run.execution_graph is None
                or run.coordinator.next_action != "execute_leaf"
                or run.current_leaf_id != leaf_id
                or current is None
                or current.status != "running"
                or current.attempt_count != attempt
            ):
                raise AssetCompositionStateError(
                    "invoke-leaf requires the exact active running graph leaf attempt"
                )
            return _execute_active_leaf_under_lease(
                state_path=state_path,
                run=run,
                leaf_id=leaf_id,
                attempt=attempt,
                invocation_path=invocation_path,
                result_path=result_path,
                record_path=record_path,
            )
    finally:
        lease.release()


def _execute_active_leaf_under_lease(
    *,
    state_path: Path,
    run: AssetCompositionRun,
    leaf_id: str,
    attempt: int,
    invocation_path: str | Path,
    result_path: str | Path,
    record_path: Path,
) -> ArtifactBinding:
    """Execute one already-validated attempt while both leases are held."""

    execution_graph = run.execution_graph
    if execution_graph is None:
        raise AssetCompositionStateError("active leaf run identity is invalid")

    attempt_root = leaf_directory(state_path, leaf_id)
    invocation_candidate = Path(invocation_path).expanduser()
    document = _confined_regular_file(
        invocation_candidate,
        within=attempt_root,
        label="leaf invocation",
    )
    invocation_binding = _artifact_binding(invocation_candidate, document)
    destination = _result_destination(
        Path(result_path).expanduser(),
        within=attempt_root,
        require_new=False,
    )

    graph = _load_execution_graph(execution_graph)
    node = next((item for item in graph.nodes if item.leaf_id == leaf_id), None)
    if node is None:
        raise AssetCompositionStateError("active leaf is absent from the frozen graph")
    runtime_catalog = repository_asset_leaf_runtime_catalog()
    descriptor = next(
        (
            item
            for item in runtime_catalog.catalog.descriptors
            if item.leaf_id == leaf_id
        ),
        None,
    )
    if descriptor is None or descriptor.descriptor_digest != node.descriptor_digest:
        raise AssetCompositionStateError(
            "active leaf descriptor differs from the repository catalog"
        )
    binding = runtime_catalog.resolve(descriptor)
    try:
        invocation = binding.invocation_model.model_validate_json(document)
    except ValueError as exc:
        raise AssetCompositionStateError("leaf invocation is invalid") from exc

    if record_path.exists() or record_path.is_symlink():
        record_document = _confined_regular_file(
            record_path,
            within=record_path.parent,
            label="leaf execution record",
        )
        try:
            record = _LeafExecutionRecord.model_validate_json(record_document)
        except ValueError as exc:
            raise AssetCompositionStateError(
                "leaf execution record is invalid"
            ) from exc
        if (
            record.run_id != run.run_id
            or record.leaf_id != leaf_id
            or record.attempt != attempt
            or record.invocation != invocation_binding
            or record.result.path != str(destination)
        ):
            raise AssetCompositionStateError(
                "active leaf attempt was already invoked with different bindings"
            )
        result_document = _confined_regular_file(
            destination,
            within=attempt_root,
            label="leaf result",
            max_bytes=_MAX_RESULT_BYTES,
        )
        observed_result = _artifact_binding(destination, result_document)
        if observed_result != record.result:
            raise AssetCompositionStateError(
                "sealed leaf result differs from its execution record"
            )
        try:
            binding.result_model.model_validate_json(result_document)
        except ValueError as exc:
            raise AssetCompositionStateError("sealed leaf result is invalid") from exc
        return observed_result

    destination = _result_destination(
        destination,
        within=attempt_root,
    )

    if any(
        descriptor.entrypoint.startswith(f"{executable} ")
        for executable in _ADMITTED_CLI_EXECUTABLES
    ):
        result = _execute_cli_entrypoint(
            descriptor.entrypoint,
            invocation_path=invocation_candidate,
            result_model=binding.result_model,
        )
    else:
        result = _execute_shared_entrypoint(
            descriptor.entrypoint,
            invocation=invocation,
            result_model=binding.result_model,
        )
    atomic_write_json(destination, result, within=attempt_root)
    result_document = _confined_regular_file(
        destination,
        within=attempt_root,
        label="leaf result",
        max_bytes=_MAX_RESULT_BYTES,
    )
    result_binding = _artifact_binding(destination, result_document)
    atomic_write_json(
        record_path,
        _LeafExecutionRecord(
            run_id=run.run_id,
            leaf_id=leaf_id,
            attempt=attempt,
            invocation=invocation_binding,
            result=result_binding,
        ),
    )
    return result_binding


__all__ = ["execute_active_leaf"]
