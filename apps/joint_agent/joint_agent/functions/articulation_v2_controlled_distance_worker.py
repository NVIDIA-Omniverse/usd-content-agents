# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal fresh-process bootstrap for retained controlled-distance readback."""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, cast

_SOURCE_READBACK_WORKER_MODE = "controlled-distance-retained-source-readback-v1"
_SOURCE_READBACK_WORKER_MODE_V2 = "controlled-distance-retained-source-readback-v2"
_SOURCE_READBACK_WORKER_MODE_V2_FIXTURE_AUDIT = (
    "controlled-distance-retained-source-readback-v2-fixture-audit"
)
_PYCACHE_SINK = "/dev/null"
_SUPPLEMENTAL_PACKAGE_SOURCE_ROOT = "packages"


def _sealed_package(name: str, root: Path) -> ModuleType:
    """Install one exact package namespace without executing its initializer."""

    module = ModuleType(name)
    module.__file__ = str(root / "__init__.py")
    module.__package__ = name
    module.__path__ = [str(root)]
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    assert spec.submodule_search_locations is not None
    spec.submodule_search_locations.clear()
    spec.submodule_search_locations.append(str(root))
    module.__spec__ = spec
    sys.modules[name] = module
    return module


def _load_contract_module() -> Any:
    """Load only the declared Joint function package and contract module."""

    functions_root = Path(__file__).resolve(strict=True).parent
    joint_root = functions_root.parent
    _sealed_package("joint_agent", joint_root)
    _sealed_package("joint_agent.functions", functions_root)
    return cast(
        Any,
        importlib.import_module(
            "joint_agent.functions.articulation_v2_controlled_distance_usd"
        ),
    )


def _install_reviewed_snapshot_paths(source_root: str, dependency_root: str) -> None:
    """Bind a sealed V2 child to the bootstrap-captured source and dependencies."""

    if not (
        sys.flags.isolated
        and sys.flags.no_site
        and sys.flags.dont_write_bytecode
        and sys.flags.safe_path
        and sys.pycache_prefix == _PYCACHE_SINK
    ):
        raise ValueError("reviewed readback worker requires a sealed interpreter")
    source = _canonical_directory(source_root, label="reviewed source snapshot")
    dependency = _canonical_directory(
        dependency_root,
        label="reviewed dependency snapshot",
    )
    current_worker = Path(__file__).resolve(strict=True)
    expected_worker = source / (
        "apps/joint_agent/joint_agent/functions/"
        "articulation_v2_controlled_distance_worker.py"
    )
    if current_worker != expected_worker:
        raise ValueError("readback worker escaped the reviewed source snapshot")
    app = _canonical_directory(
        str(source / "apps/joint_agent"),
        label="reviewed Joint package snapshot",
    )
    supplemental = _canonical_single_child_directory(
        source / _SUPPLEMENTAL_PACKAGE_SOURCE_ROOT,
        label="reviewed supplemental package snapshot",
    )
    paths = (
        source,
        app,
        supplemental,
        dependency,
    )
    for path in paths:
        token = str(path)
        if token not in sys.path:
            sys.path.append(token)


def _canonical_directory(value: str, *, label: str) -> Path:
    if type(value) is not str or not value or "\x00" in value:
        raise ValueError(f"{label} is invalid")
    path = Path(os.path.abspath(value))
    if not path.is_dir() or path.resolve(strict=True) != path:
        raise ValueError(f"{label} is not an exact canonical directory")
    return path


def _canonical_single_child_directory(parent: Path, *, label: str) -> Path:
    """Require the one package root admitted beneath a reviewed container."""

    container = _canonical_directory(
        str(parent),
        label=f"{label} container",
    )
    entries = tuple(container.iterdir())
    if len(entries) != 1:
        raise ValueError(f"{label} container must have exactly one entry")
    return _canonical_directory(str(entries[0]), label=label)


def _emit(payload: dict[str, Any]) -> None:
    sys.stdout.write(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    sys.stdout.flush()


def main(argv: list[str]) -> int:
    """Run exact source validation in one fresh interpreter process."""

    sealed_v2 = False
    if len(argv) == 7 and argv[0] == _SOURCE_READBACK_WORKER_MODE_V2:
        _install_reviewed_snapshot_paths(argv[1], argv[2])
        argv = [argv[0], *argv[3:]]
        sealed_v2 = True
    worker = _load_contract_module()
    if len(argv) != 5 or not (
        argv[0]
        in {
            _SOURCE_READBACK_WORKER_MODE,
            _SOURCE_READBACK_WORKER_MODE_V2_FIXTURE_AUDIT,
        }
        or (argv[0] == _SOURCE_READBACK_WORKER_MODE_V2 and sealed_v2)
    ):
        return 64
    mode, stage_path, source_joint_path, body0_path, body1_path = argv
    failure_code = "controlled_distance_source_stage_open_failed"
    failure_detail = "could not open retained controlled-distance source"
    try:
        from pxr import Sdf, Usd

        detached_layer = Sdf.Layer.OpenAsAnonymous(stage_path)
        stage = Usd.Stage.Open(detached_layer) if detached_layer else None
        if not stage:
            raise worker.ControlledDistanceError(
                "controlled_distance_source_stage_open_failed",
                "could not open retained controlled-distance source",
            )
        failure_code = "controlled_distance_source_worker_failed"
        failure_detail = "retained controlled-distance source readback failed"
        worker._require_single_source_joint(
            stage,
            source_joint_path=source_joint_path,
        )
        readback_function = (
            worker.readback_controlled_distance_stage_v1
            if mode == worker._SOURCE_READBACK_WORKER_MODE
            else worker.readback_controlled_distance_stage_v2
        )
        readback = readback_function(
            stage,
            joint_path=source_joint_path,
            expected_body0_prim_path=body0_path,
            expected_body1_prim_path=body1_path,
        )
        payload: dict[str, Any] = {
            "status": "ok",
            "readback": readback.model_dump(mode="json", exclude_none=True),
        }
        if mode != worker._SOURCE_READBACK_WORKER_MODE:
            payload["mode"] = mode
        returncode = 0
    except worker.ControlledDistanceError as exc:
        payload = {
            "status": "error",
            "code": exc.code,
            "detail": exc.detail,
        }
        returncode = 2
    except Exception as exc:
        payload = {
            "status": "error",
            "code": failure_code,
            "detail": f"{failure_detail}: {type(exc).__name__}",
        }
        returncode = 2
    _emit(payload)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
