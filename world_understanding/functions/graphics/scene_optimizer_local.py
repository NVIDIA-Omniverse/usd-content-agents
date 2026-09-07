# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local USD scene optimization using Scene Optimizer package in an isolated subprocess.

The Scene Optimizer (SO) package bundles native C++ libraries that conflict
with external ``pxr``/OpenUSD bindings at the ABI level. To avoid crashes, all
SO work runs in an isolated subprocess that uses the SO package's own stock USD
25.11 ``pxr`` Python bindings instead of the ones from the main virtualenv.

Setup:
    Download the public ``scene_optimizer_core_usd_25.11_py_3.12`` zip and
    unpack it. Point ``WU_SO_PACKAGE_DIR`` at the unpacked root (the one that
    contains ``python/``, ``lib/``, ``extraLibs/``, ``usdpy/``). The repo-level
    helper ``scripts/fetch_build_resources.sh`` does this into
    ``.build-resources/scene_optimizer_core/``.

Optional environment variables:
    WU_SO_PYTHON: Path to a Python 3.12 executable for the subprocess.
        When unset, defaults to ``sys.executable`` if the current
        interpreter is Python 3.12, otherwise to ``python3.12`` (PATH lookup,
        with the standard ``FileNotFoundError`` if missing). The SO native
        bindings are cpython-312-only, so a 3.13+ host must either have
        ``python3.12`` on ``PATH`` or set ``WU_SO_PYTHON`` explicitly.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the worker script (executed in the isolated subprocess)
_SO_WORKER_PATH = Path(__file__).parent / "so_worker.py"
_SO_EXPORT_PATH = Path(__file__).parent / "so_export.py"


def _build_operations_list(settings: dict[str, Any]) -> list[tuple[str, dict]]:
    """Convert scene_optimizer_settings dict to an ordered list of SO operations.

    Operation ordering matches the NVCF Kit service:
    deinstance (``utilityFunction``) → ``splitMeshes`` → ``deduplicateGeometry``.
    ``merge`` is NOT included — it combines all meshes into one prim which
    destroys the individual mesh structure needed for correspondence tracking
    and ``restore_usd``.

    Args:
        settings: Scene optimizer settings dict (snake_case keys from
            the validated Pydantic model).

    Returns:
        Ordered list of ``(operation_name, params_dict)`` tuples.
    """
    operations: list[tuple[str, dict]] = []

    # Deinstance via utilityFunction (function=0 = DEINSTANCE)
    # Must run first, before split/dedup, matching Kit service order
    if settings.get("enable_deinstance", True):
        deinstance_config = settings.get("deinstance", {})
        operations.append(
            (
                "utilityFunction",
                {
                    "function": 0,  # DEINSTANCE
                    "primPaths": deinstance_config.get("prim_paths", []),
                },
            )
        )

    # Split meshes — match Kit service defaults for correspondence tracking
    if settings.get("enable_split_meshes", True):
        operations.append(
            (
                "splitMeshes",
                {
                    "splitOn": 1,  # GEOM_SUBSETS
                    "method": 1,  # MESH_PRIM
                    "originalGeomOption": 1,  # DELETE
                },
            )
        )

    # Deduplicate geometry — match Kit service defaults for correspondence tracking
    if settings.get("enable_deduplicate", True):
        dedup_config = settings.get("deduplicate", {})
        dedup_params: dict[str, Any] = {
            "duplicateMethod": 2,  # INSTANCEABLE_REFERENCE
        }
        if "tolerance" in dedup_config:
            dedup_params["tolerance"] = dedup_config["tolerance"]
        if "consider_deep_transforms" in dedup_config:
            dedup_params["considerDeepTransforms"] = dedup_config[
                "consider_deep_transforms"
            ]
        if "fuzzy" in dedup_config:
            dedup_params["fuzzy"] = dedup_config["fuzzy"]
        if "use_gpu" in dedup_config:
            dedup_params["useGpu"] = dedup_config["use_gpu"]
        if "allow_scaling" in dedup_config:
            dedup_params["allowScaling"] = dedup_config["allow_scaling"]
        if "ignore_attributes" in dedup_config:
            dedup_params["ignoreAttributes"] = dedup_config["ignore_attributes"]
        operations.append(("deduplicateGeometry", dedup_params))

    return operations


SO_PACKAGE_SUBDIRS = ("python", "lib", "extraLibs", "usdpy")


def _is_valid_so_package_dir(path: Path) -> bool:
    """Return True when ``path`` contains the required SO Core subdirectories."""
    return all((path / sub).is_dir() for sub in SO_PACKAGE_SUBDIRS)


def _default_so_package_dir() -> Path:
    """Default unpack location written by ``./scripts/fetch_build_resources.sh``."""
    return Path.cwd() / ".build-resources" / "scene_optimizer_core"


def _resolve_so_package_dir() -> Path:
    """Resolve the Scene Optimizer Core package root.

    Resolution order:
        1. ``WU_SO_PACKAGE_DIR`` environment variable (explicit override).
        2. ``<cwd>/.build-resources/scene_optimizer_core/`` — the location
           populated by ``./scripts/fetch_build_resources.sh`` when run from
           the repo root.

    Either target must contain ``python/``, ``lib/``, ``extraLibs/``, ``usdpy/``.
    """
    so_package_dir_env = os.environ.get("WU_SO_PACKAGE_DIR")
    if so_package_dir_env:
        so_package_dir = Path(so_package_dir_env)
        for sub in SO_PACKAGE_SUBDIRS:
            if not (so_package_dir / sub).is_dir():
                raise RuntimeError(
                    f"Scene Optimizer package directory missing expected "
                    f"subdirectory: {so_package_dir / sub}"
                )
        return so_package_dir

    default = _default_so_package_dir()
    if _is_valid_so_package_dir(default):
        return default

    raise RuntimeError(
        "Scene Optimizer Core package not found. Run "
        "`./scripts/fetch_build_resources.sh` from the repo root to fetch it "
        f"into {default}, or set WU_SO_PACKAGE_DIR to point at an unpacked "
        "scene_optimizer_core_usd_25.11_py_3.12 package."
    )


def _resolve_so_python() -> str:
    """Return the Python executable to launch the SO subprocess with.

    Resolution order:
        1. ``WU_SO_PYTHON`` environment variable (explicit override). When
           absolute, the path must point to an existing file — fail fast
           with a clear ``ValueError`` instead of letting ``subprocess.run``
           emit an opaque ``FileNotFoundError`` later. Relative names are
           passed through unchanged so ``subprocess.run`` does its normal
           ``PATH`` lookup.
        2. ``sys.executable`` when the parent process is itself Python 3.12 —
           the SO Core bundle's ``cpython-312-...so`` extensions need that
           exact ABI, and using the current interpreter avoids relying on
           ``python3.12`` being on ``PATH``.
        3. ``"python3.12"`` (PATH lookup) when the parent is on a different
           Python version. ``subprocess.run`` will raise ``FileNotFoundError``
           if the binary is missing — callers (e.g. the ``optimize_usd`` task)
           translate that into a remote-backend fallback.
    """
    override = os.environ.get("WU_SO_PYTHON")
    if override:
        if os.path.isabs(override) and not os.path.isfile(override):
            raise ValueError(
                f"WU_SO_PYTHON is set to an absolute path that does not "
                f"exist: {override}"
            )
        return override
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    return "python3.12"


def _python_libdir(so_python: str) -> str | None:
    """Return the directory containing ``libpython3.x.so`` for ``so_python``.

    The SO Core bundle's compiled extensions (``_usd.so``, ``_usdGeom.so``,
    ``_omni_scene_optimizer_impl_core.cpython-312-...so``) all carry a
    ``DT_NEEDED libpython3.12.so.1.0``. With newer toolchains the python
    binary tags its private ``lib/`` with ``DT_RUNPATH`` (not searched for
    transitive deps), so the loader falls through to ``LD_LIBRARY_PATH`` —
    which is why we inject the interpreter's ``LIBDIR`` here.
    """
    if so_python == sys.executable:
        return sysconfig.get_config_var("LIBDIR")  # type: ignore[no-any-return]
    # Match the worker's isolation contract for the probe too:
    #   ``-S`` skips ``site.py`` so any parent ``sitecustomize.py`` / ``.pth``
    #   files cannot run before our snippet and pollute stdout.
    # Strict parsing: take the last non-empty stdout line. ``sysconfig`` may
    # legitimately emit nothing (some build configs report ``LIBDIR = None``);
    # ``site.py`` itself can't fire (we passed ``-S``), but third-party paths
    # injected via ``PYTHONPATH`` could still print on import. Validate the
    # result looks like an absolute path before trusting it in
    # ``LD_LIBRARY_PATH``.
    try:
        proc = subprocess.run(
            [
                so_python,
                "-S",
                "-c",
                "import sysconfig; print(sysconfig.get_config_var('LIBDIR') or '')",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None
    libdir = lines[-1]
    if not os.path.isabs(libdir):
        return None
    return libdir


def _subprocess_env(so_package_dir: Path, so_python: str) -> dict[str, str]:
    """Build the isolated environment variables for the SO subprocess.

    The whole point of this subprocess is to keep the SO bundle's stock
    OpenUSD 25.11 bindings away from the main environment's ``pxr`` provider
    (different layout, different C++ ABI). To preserve that isolation:

    - ``LD_LIBRARY_PATH`` is **replaced** (not appended-to) with exactly
      ``[SO/lib, SO/extraLibs, <SO Python's LIBDIR>]``. The libdir entry
      rescues ``libpython3.x.so`` for `DT_NEEDED` lookups in the bundle's
      compiled extensions; everything else the bundle needs is shipped in
      ``lib/`` or ``extraLibs/``. Inheriting the parent's
      ``LD_LIBRARY_PATH`` would let unrelated host paths (Kit, rendering
      stack, system libs) satisfy missing transitive deps and silently
      mix ABIs.
    - ``PYTHONPATH`` is replaced with the SO Python bindings + stock USD
      ``pxr`` bindings under ``usdpy/``.
    - ``PXR_PLUGINPATH_NAME`` points USD's plugin registry at
      ``extraLibs/usd`` so plugin discovery is independent of how the
      libs were loaded.
    - On Windows, ``PXR_USD_WINDOWS_DLL_PATH`` is **replaced** with the SO
      bundle's ``lib`` and ``extraLibs`` directories, and those directories
      are prepended to ``PATH``.  The parent process may already point
      ``PXR_USD_WINDOWS_DLL_PATH`` at a different OpenUSD provider; inheriting
      that value mixes incompatible DLLs before Python can import ``pxr.Tf``.

    Note that the worker is also launched with ``-S`` (see callers) so
    ``site.py`` doesn't auto-add the parent venv's ``site-packages`` to
    ``sys.path``. Together these prevent the app's ``pxr`` provider from
    leaking in even when ``WU_SO_PYTHON`` resolves to the project venv.
    """
    env = os.environ.copy()

    bundle_lib_paths = [
        str(so_package_dir / "lib"),
        str(so_package_dir / "extraLibs"),
    ]
    ld_paths = list(bundle_lib_paths)
    py_libdir = _python_libdir(so_python)
    if py_libdir:
        ld_paths.append(py_libdir)
    env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_paths)

    if sys.platform == "win32":
        # usdpy/pxr/Tf/__init__.py gives this variable priority over PATH and
        # registers every listed directory with os.add_dll_directory().  Make
        # it authoritative so a parent usd-exchange installation cannot leak
        # an ABI-incompatible OpenUSD DLL directory into the isolated worker.
        env["PXR_USD_WINDOWS_DLL_PATH"] = os.pathsep.join(bundle_lib_paths)
        inherited_path = env.get("PATH")
        env["PATH"] = os.pathsep.join(
            [*bundle_lib_paths, inherited_path] if inherited_path else bundle_lib_paths
        )

    env["PYTHONPATH"] = os.pathsep.join(
        [str(so_package_dir / "python"), str(so_package_dir / "usdpy")]
    )
    env["PXR_PLUGINPATH_NAME"] = str(so_package_dir / "extraLibs" / "usd")
    return env


def optimize_usd_local(
    input_path: Path | str,
    output_path: Path | str,
    optimization_config: dict[str, Any] | None = None,
    *,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Optimize a USD file locally using the Scene Optimizer package.

    Runs the SO operations in an isolated subprocess to avoid ABI conflicts
    with the app's pxr/OpenUSD bindings. The subprocess uses stock USD 25.11
    pxr bindings and the SO package's own native libraries, both shipped in
    the same package.

    Args:
        input_path: Path to the input USD file.
        output_path: Path where the optimized USD will be written.
        optimization_config: Dict with ``scene_optimizer_settings`` and
            optional ``generate_report``, ``capture_stats``, ``verbose`` keys.
        approved_dependency_roots: Filesystem roots from which the isolated
            worker may copy dependencies into the portable output sidecar.
            Defaults to the input USD parent.

    Returns:
        Result dict with keys matching the NVCF backend:
            ``status``, ``optimization_time``, ``stage_size_bytes``,
            ``operations_executed``, ``report``, ``correspondence_map``.

    Raises:
        RuntimeError: If ``WU_SO_PACKAGE_DIR`` is unset, the package
            directory structure is invalid, or the subprocess fails.
    """
    optimization_config = optimization_config or {}

    so_package_dir = _resolve_so_package_dir()
    so_python = _resolve_so_python()

    settings = optimization_config.get("scene_optimizer_settings", {})
    operations = _build_operations_list(settings)

    logger.info(
        "Local SO optimization: %d operation(s) — %s",
        len(operations),
        " -> ".join(op[0] for op in operations),
    )

    input_path = Path(input_path)
    output_path = Path(output_path)
    root_values: tuple[Path | str, ...]
    if approved_dependency_roots is None:
        root_values = (input_path.resolve().parent,)
    else:
        root_values = tuple(approved_dependency_roots)
    # Validate in the parent before launching the ABI-isolated worker so bad
    # trust roots surface as precise caller errors instead of subprocess
    # failures. Keep this contract aligned with the local UV worker path.
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )

    dependency_roots = [str(root) for root in _normalize_dependency_roots(root_values)]

    with tempfile.TemporaryDirectory(prefix="so_local_") as tmp_dir:
        worker_path = os.path.join(tmp_dir, "_so_worker.py")
        manifest_path = os.path.join(tmp_dir, "manifest.json")

        shutil.copy2(str(_SO_WORKER_PATH), worker_path)
        shutil.copy2(str(_SO_EXPORT_PATH), os.path.join(tmp_dir, "so_export.py"))

        params = {
            "input_usd_path": str(input_path),
            "output_usd_path": str(output_path),
            "approved_dependency_roots": dependency_roots,
            "operations": operations,
            "generate_report": settings.get("generate_report", True),
            "capture_stats": settings.get("capture_stats", True),
            "verbose": settings.get("verbose", False),
            "manifest_path": manifest_path,
        }

        env = _subprocess_env(so_package_dir, so_python)

        logger.info("Launching Scene Optimizer subprocess: %s", so_python)
        logger.debug("  LD_LIBRARY_PATH=%s", env["LD_LIBRARY_PATH"])
        logger.debug("  PYTHONPATH=%s", env["PYTHONPATH"])

        # Default timeout: 30 min (large scenes can take minutes)
        timeout = optimization_config.get("timeout", 1800)

        start_time = time.time()
        try:
            # ``-S`` disables ``site.py``'s auto-insertion of site-packages
            # into ``sys.path``. Combined with the explicit ``PYTHONPATH``
            # in ``_subprocess_env``, this keeps the app's ``pxr`` provider
            # out of the worker even when ``so_python`` is the project venv.
            proc = subprocess.run(
                [so_python, "-S", worker_path, json.dumps(params)],
                capture_output=True,
                text=True,
                env=env,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Scene Optimizer subprocess timed out after {timeout}s"
            ) from None

        if proc.returncode != 0:
            error_msg = (
                f"Scene Optimizer subprocess failed (exit code {proc.returncode})"
            )
            if proc.stdout:
                error_msg += f"\n--- stdout (last 1000) ---\n{proc.stdout[-1000:]}"
            if proc.stderr:
                error_msg += f"\n--- stderr (last 2000) ---\n{proc.stderr[-2000:]}"
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        # Read manifest
        if not os.path.exists(manifest_path):
            raise RuntimeError(
                "Scene Optimizer subprocess did not produce manifest. "
                f"stdout: {proc.stdout[-500:]}, "
                f"stderr: {proc.stderr[-500:]}"
            )

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    elapsed = time.time() - start_time
    logger.info("Local SO optimization completed in %.2fs", elapsed)

    # Return result dict matching NVCF format. ``error`` is passed through
    # so callers can surface the worker's traceback (e.g. ``ImportError:
    # libpython3.12.so.1.0``) instead of the generic "Unknown optimization
    # error" string the orchestrator falls back to when this key is absent.
    return {
        "status": manifest.get("status", "error"),
        "optimization_time": manifest.get("optimization_time", elapsed),
        "stage_size_bytes": manifest.get("stage_size_bytes", 0),
        "operations_executed": manifest.get("operations_executed", []),
        "report": manifest.get("report", ""),
        "correspondence_map": manifest.get("correspondence_map", {}),
        "error": manifest.get("error"),
    }
