# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared immutable source projection for Joint Agent contract validation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from world_understanding.functions.physics.joint_rigger import (
    ArtifactIdentityV1,
    JointRiggerContractError,
)
from world_understanding.functions.physics.joint_rigger.source_binding import (
    BoundInputDirectory,
    SealedSourceBinding,
    bound_input_dependency_snapshots,
    close_source_binding,
    create_sealed_source_binding,
    materialize_bound_input,
    remove_bound_input_directory,
    require_sealed_source_binding,
)


@dataclass(frozen=True, slots=True)
class SourceProjectionMessages:
    """Domain-specific messages for one shared source-binding transaction."""

    binding_mismatch: str
    materialization_failure: str
    changed_before_use: str
    changed_during_use: str


@contextmanager
def bound_source_projection(
    *,
    source_path: Path,
    expected_source: ArtifactIdentityV1,
    error_prefix: str,
    messages: SourceProjectionMessages,
) -> Iterator[Path]:
    """Yield exact source bytes and preserve primary failures through cleanup."""

    binding: SealedSourceBinding | None = None
    bound_directory: BoundInputDirectory | None = None
    primary_error: BaseException | None = None
    try:
        try:
            binding = create_sealed_source_binding(
                source_path,
                expected=expected_source,
            )
        except Exception as exc:
            raise JointRiggerContractError(
                f"{error_prefix}_source_mutated",
                messages.binding_mismatch,
            ) from exc
        try:
            bound_source, bound_directory, _ = materialize_bound_input(
                descriptor=binding.descriptor,
                expected_sha256=binding.sha256,
                logical_input_path=source_path,
                dependencies=bound_input_dependency_snapshots(binding),
            )
        except Exception as exc:
            raise JointRiggerContractError(
                f"{error_prefix}_source_invalid",
                messages.materialization_failure,
            ) from exc
        try:
            require_sealed_source_binding(binding)
        except Exception as exc:
            raise JointRiggerContractError(
                f"{error_prefix}_source_mutated",
                messages.changed_before_use,
            ) from exc
        yield bound_source
        try:
            require_sealed_source_binding(binding)
        except Exception as exc:
            raise JointRiggerContractError(
                f"{error_prefix}_source_mutated",
                messages.changed_during_use,
            ) from exc
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        cleanup_errors: list[BaseException] = []
        if bound_directory is not None:
            try:
                remove_bound_input_directory(bound_directory)
            except BaseException as exc:
                cleanup_errors.append(exc)
        if binding is not None:
            try:
                cleanup_errors.extend(close_source_binding(binding))
            except BaseException as exc:
                cleanup_errors.append(exc)
        if cleanup_errors:
            detail = "; ".join(str(error) for error in cleanup_errors)
            if primary_error is not None:
                primary_error.add_note("Bound source cleanup also failed: " + detail)
            else:
                raise JointRiggerContractError(
                    f"{error_prefix}_source_cleanup_failed",
                    "bound source cleanup failed: " + detail,
                ) from cleanup_errors[0]


__all__ = ["SourceProjectionMessages", "bound_source_projection"]
