# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Isolated entrypoint for one approved deterministic repair operation."""

from __future__ import annotations

import argparse
import sys
from importlib import metadata
from pathlib import Path

from .artifacts import atomic_write_json, file_sha256
from .models import RepairOperation
from .policy import assert_worker_operations_enabled
from .process_limits import (
    ADDRESS_SPACE_LIMIT_MODE_HARD,
    ADDRESS_SPACE_LIMIT_MODE_SOFT,
    ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
    OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
    OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
    limit_address_space,
    limit_address_space_soft,
    limit_cpu_affinity,
)
from .sdf_backend_qualification import (
    DEFAULT_SDF_BACKEND_ID,
    SDF_REBUILD_IMPLEMENTATION_VERSION,
)
from .worker_ids import SDF_REBUILD_WORKER, canonical_worker_name
from .workers import (
    CollisionGeometryWorker,
    GeogramLocalRepairWorker,
    ManifoldSeamWorker,
    OcpShapeHealWorker,
    PmpPatchWorker,
    SceneOptimizerDeinstanceWorker,
    SdfRebuildWorker,
    TrimeshBoundedHoleFillWorker,
    TrimeshCleanupWorker,
    UsdStructureRepairWorker,
)
from .workers.base import RepairWorker, WorkerResult
from .workers.collision_geometry import CollisionGeometryParameters

_WORKERS: dict[str, RepairWorker] = {
    CollisionGeometryWorker.name: CollisionGeometryWorker(),
    TrimeshCleanupWorker.name: TrimeshCleanupWorker(),
    TrimeshBoundedHoleFillWorker.name: TrimeshBoundedHoleFillWorker(),
    OcpShapeHealWorker.name: OcpShapeHealWorker(),
    UsdStructureRepairWorker.name: UsdStructureRepairWorker(),
    GeogramLocalRepairWorker.name: GeogramLocalRepairWorker(),
    SceneOptimizerDeinstanceWorker.name: SceneOptimizerDeinstanceWorker(),
    SdfRebuildWorker.name: SdfRebuildWorker(),
    ManifoldSeamWorker.name: ManifoldSeamWorker(),
    PmpPatchWorker.name: PmpPatchWorker(),
}
_WORKER_DISTRIBUTIONS = {
    "trimesh_conservative_cleanup": "trimesh",
    "trimesh_bounded_hole_fill": "trimesh",
    "manifold_restore_merge_vectors": "manifold3d",
}


def _worker_version(worker: str) -> str:
    if worker == CollisionGeometryWorker.name:
        try:
            return metadata.version("geometry-repair")
        except metadata.PackageNotFoundError:
            return "source-tree"
    if worker == "pmp_patch":
        return "pmp-library:2a2ad502743724ba90e09af816364c84032f9015"
    if worker == "geogram_local_repair":
        return f"vorpalite-{GeogramLocalRepairWorker.version}"
    if worker == "scene_optimizer_deinstance":
        return "nvidia-scene-optimizer-core"
    if worker == OcpShapeHealWorker.name:
        available, _reason = OcpShapeHealWorker().available()
        if not available:
            return "unavailable"
        try:
            import OCP

            return str(getattr(OCP, "__version__", "externally-supplied"))
        except ImportError:
            return "unavailable"
    if worker == SDF_REBUILD_WORKER:
        return SDF_REBUILD_IMPLEMENTATION_VERSION
    if worker == "usd_structure_repair":
        try:
            from pxr import Usd

            runtime = Usd.GetVersion()
            runtime_version = (
                f"{runtime[1]}.{runtime[2]}"
                if len(runtime) >= 3 and runtime[0] == 0
                else ".".join(str(item) for item in runtime)
            )
            return f"usd-exchange-{metadata.version('usd-exchange')}/openusd-{runtime_version}"
        except (ImportError, metadata.PackageNotFoundError):
            return "unknown"
    try:
        return metadata.version(_WORKER_DISTRIBUTIONS[worker])
    except metadata.PackageNotFoundError:
        return "unknown"


def _requested_operations(
    operation: RepairOperation,
    worker: RepairWorker,
) -> frozenset[str]:
    """Return explicit per-request operations or the worker's fixed capability set."""

    requested = operation.parameters.get("operation")
    if requested is None:
        return worker.operations
    if not isinstance(requested, str) or not requested:
        raise ValueError("operation.parameters.operation must be a non-empty string")
    return frozenset({requested})


def _assert_worker_binding(operation: RepairOperation, worker_name: str) -> None:
    if operation.worker != worker_name:
        raise ValueError(
            f"operation worker {operation.worker!r} does not match runner worker {worker_name!r}"
        )


def _worker_availability(
    worker: RepairWorker,
    operation: RepairOperation,
) -> tuple[bool, str | None]:
    """Check availability against the backend selected by this operation."""

    if isinstance(worker, SdfRebuildWorker):
        backend_id = operation.parameters.get("backend_id", DEFAULT_SDF_BACKEND_ID)
        if isinstance(backend_id, str):
            return worker.available_for_backend(backend_id)
    return worker.available()


def _address_space_limit_policy(
    *,
    worker_name: str,
    operation: RepairOperation,
    memory_mb: int,
) -> tuple[str, str, str | None]:
    """Return the exact address-space policy for this worker invocation."""

    if worker_name == CollisionGeometryWorker.name:
        parameters = CollisionGeometryParameters.model_validate(operation.parameters)
        if parameters.budgets.max_memory_mb != memory_mb:
            raise ValueError(
                "collision operation memory budget does not match runner memory budget"
            )
        if parameters.runtime_engine == "ovphysx" and parameters.profile in {
            "rigid_pick_place",
            "static_environment",
        }:
            # Keep preprocessing under the requested cap while preserving the
            # inherited hard ceiling. The actual OvPhysX daemon is the only child
            # that opts in to raising this soft limit for CUDA VA reservations.
            return (
                ADDRESS_SPACE_LIMIT_MODE_SOFT,
                OVPHYSX_ADDRESS_SPACE_LIMIT_SCOPE,
                OVPHYSX_DAEMON_ADDRESS_SPACE_LIMIT_EXEMPTION,
            )
    return (
        ADDRESS_SPACE_LIMIT_MODE_HARD,
        ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE,
        None,
    )


def _execution_metadata(
    *,
    worker_name: str,
    memory_budget_mb: int,
    memory_limit_mb: int | None,
    address_space_limit_mode: str,
    address_space_limit_scope: str,
    memory_limit_exemption: str | None,
) -> dict[str, str | int | None]:
    metadata: dict[str, str | int | None] = {
        "execution_scope": "isolated_subprocess",
        "memory_budget_mb": memory_budget_mb,
        "memory_limit_mb": memory_limit_mb,
        "address_space_limit_mode": address_space_limit_mode,
        "address_space_limit_scope": address_space_limit_scope,
        "implementation_version": _worker_version(worker_name),
    }
    if memory_limit_exemption is not None:
        metadata["memory_limit_exemption"] = memory_limit_exemption
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="geometry-repair-worker")
    parser.add_argument(
        "--worker",
        required=True,
        type=canonical_worker_name,
        choices=sorted(_WORKERS),
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operation", required=True, type=Path)
    parser.add_argument("--result", required=True, type=Path)
    parser.add_argument("--memory-mb", required=True, type=int)
    args = parser.parse_args(argv)

    result: WorkerResult
    applied_memory_limit_mb: int | None = None
    address_space_limit_mode = ADDRESS_SPACE_LIMIT_MODE_HARD
    address_space_limit_scope = ADDRESS_SPACE_LIMIT_SCOPE_PROCESS_TREE
    memory_limit_exemption: str | None = None
    try:
        operation = RepairOperation.model_validate_json(args.operation.read_text(encoding="utf-8"))
        _assert_worker_binding(operation, args.worker)
        worker = _WORKERS[args.worker]
        assert_worker_operations_enabled(args.worker, worker.operations)
        assert_worker_operations_enabled(
            args.worker,
            _requested_operations(operation, worker),
        )
        if args.worker == SdfRebuildWorker.name:
            limit_cpu_affinity(1)
        (
            address_space_limit_mode,
            address_space_limit_scope,
            memory_limit_exemption,
        ) = _address_space_limit_policy(
            worker_name=args.worker,
            operation=operation,
            memory_mb=args.memory_mb,
        )
        if address_space_limit_mode == ADDRESS_SPACE_LIMIT_MODE_SOFT:
            limit_address_space_soft(args.memory_mb)
        else:
            limit_address_space(args.memory_mb)
        applied_memory_limit_mb = args.memory_mb
        available, reason = _worker_availability(worker, operation)
        if not available:
            result = WorkerResult(
                status="unavailable",
                failures=[reason or f"{args.worker} is unavailable"],
            )
        else:
            result = worker.execute(
                source=args.source.resolve(),
                output=args.output.resolve(),
                operation=operation,
            )
        if result.output_path and Path(result.output_path).is_file():
            result.output_sha256 = file_sha256(result.output_path)
        result.metadata = {
            **result.metadata,
            **_execution_metadata(
                worker_name=args.worker,
                memory_budget_mb=args.memory_mb,
                memory_limit_mb=applied_memory_limit_mb,
                address_space_limit_mode=address_space_limit_mode,
                address_space_limit_scope=address_space_limit_scope,
                memory_limit_exemption=memory_limit_exemption,
            ),
        }
    except Exception as exc:
        result = WorkerResult(
            status="failed",
            failures=[f"{type(exc).__name__}: {exc}"],
            metadata=_execution_metadata(
                worker_name=args.worker,
                memory_budget_mb=args.memory_mb,
                memory_limit_mb=applied_memory_limit_mb,
                address_space_limit_mode=address_space_limit_mode,
                address_space_limit_scope=address_space_limit_scope,
                memory_limit_exemption=memory_limit_exemption,
            ),
        )
    atomic_write_json(args.result, result)
    return 0 if result.status == "completed" else 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
