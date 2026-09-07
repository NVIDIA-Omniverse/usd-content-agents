# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""UV generation using Scene Optimizer — local subprocess or remote endpoint.

Provides two UV generation modes via the Scene Optimizer C++ library:

- ``generate_atlas_uvs``: UV atlas unwrapping via autouv-core (Boundary First
  Flattening).  Requires the autouv-core library — currently Linux only.
- ``generate_projection_uvs``: Projection-based UV mapping (planar, spherical,
  cylindrical, triplanar, cube).  No external dependencies beyond USD —
  works on any platform.

Both functions write ``primvars:st`` (faceVarying, indexed) on processed meshes.

Backends:
- ``local`` (default): Runs SO in an isolated subprocess (same as
  ``scene_optimizer_local.py``).  Automatically falls back to ``remote`` if the
  local backend is unavailable (macOS, ``WU_SO_PACKAGE_DIR`` unset).
- ``remote``: Calls a remote Scene Optimizer service ``/generate-uvs`` endpoint,
  skipping the local backend.
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import Iterable
from enum import IntEnum
from pathlib import Path
from typing import Any

from world_understanding.functions.graphics.so_export import (
    _atomic_output_file,
    _lexical_absolute_path,
)
from world_understanding.utils.data_uri import should_use_data_uri
from world_understanding.utils.usd.stage import create_data_uri_from_file

logger = logging.getLogger(__name__)

_SO_UV_WORKER_PATH = Path(__file__).parent / "so_uv_worker.py"
_SO_EXPORT_PATH = Path(__file__).parent / "so_export.py"


class _LocalSOUnavailableError(RuntimeError):
    """The local Scene Optimizer runtime could not start or import."""


def _is_uv_worker_process_startup_failure(
    stderr: str,
    executable: str,
) -> bool:
    """Return whether a pre-manifest worker exit proves import/loader failure."""
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    if not lines:
        return False

    terminal = lines[-1]
    if terminal.startswith(("ModuleNotFoundError:", "ImportError:")):
        return True

    linux_loader_prefix = f"{executable}: error while loading shared libraries:"
    if any(line.startswith(linux_loader_prefix) for line in lines):
        return True
    return any(
        line.startswith("dyld[") and ": Library not loaded:" in line for line in lines
    )


def _is_local_so_unavailable(error: RuntimeError | FileNotFoundError) -> bool:
    """Return whether a local failure is safe to retry on the remote backend."""
    return isinstance(error, FileNotFoundError | _LocalSOUnavailableError)


class ProjectionType(IntEnum):
    """UV projection type (matches C++ ProjectionType enum)."""

    PLANAR = 0
    SPHERICAL = 1
    CYLINDRICAL = 2
    TRIPLANAR = 3
    CUBE = 4


def _resolve_so_paths() -> tuple[Path, str]:
    """Resolve Scene Optimizer package root and subprocess Python.

    Resolution order:
        1. ``WU_SO_PACKAGE_DIR`` environment variable (explicit override).
        2. ``<cwd>/.build-resources/scene_optimizer_core/`` - the location
           populated by ``./scripts/fetch_build_resources.sh``.

    Returns:
        (so_package_dir, python_executable)

    Raises:
        RuntimeError: If neither resolution source yields an unpacked
            package with ``python/``, ``lib/``, ``extraLibs/``, ``usdpy/``.
    """
    from world_understanding.functions.graphics.scene_optimizer_local import (
        _resolve_so_package_dir,
        _resolve_so_python,
    )

    return _resolve_so_package_dir(), _resolve_so_python()


def _build_uv_generation_settings(
    operation: str,
    op_params: dict[str, Any],
    output_format: str = "usdc",
) -> dict[str, Any]:
    """Build ``uv_generation_settings`` dict for the NVCF ``/generate-uvs`` endpoint.

    Maps a single operation + params into the service's request schema.
    """
    settings: dict[str, Any] = {
        "enable_generate_projection_uvs": operation == "generateProjectionUVs",
        "enable_generate_atlas_uvs": operation == "generateAtlasUVs",
        "output_format": output_format,
    }
    if operation == "generateProjectionUVs":
        settings["generate_projection_uvs"] = op_params
    elif operation == "generateAtlasUVs":
        settings["generate_atlas_uvs"] = op_params
    return settings


async def _generate_uvs_from_url(
    input_url: str,
    output_path: Path,
    operation: str,
    op_params: dict[str, Any],
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 600,
    max_retries: int = 3,
) -> dict[str, Any]:
    """Call the NVCF ``/generate-uvs`` endpoint with a USD URL.

    Mirrors ``optimize_usd_from_url`` from ``scene_optimizer_nvcf.py``
    but targets ``/generate-uvs`` and uses the UV-specific request/response
    schema.
    """
    from world_understanding.utils.nvcf_utils import (
        create_nvcf_headers,
        execute_nvcf_request_async,
        get_base_url,
        get_nvcf_api_key,
    )

    api_key = get_nvcf_api_key(api_key)
    base_url = get_base_url(
        base_url, "OPTIMIZER_ENDPOINT", "NVCF_OPTIMIZER_FUNCTION_ID"
    )
    full_url = f"{base_url.rstrip('/')}/generate-uvs"

    # Determine output format from extension
    output_suffix = output_path.suffix.lower().lstrip(".")
    output_format = (
        output_suffix if output_suffix in ("usd", "usda", "usdc") else "usdc"
    )

    uv_settings = _build_uv_generation_settings(operation, op_params, output_format)
    params = {
        "url": input_url,
        "uv_generation_settings": uv_settings,
        "timeout": timeout,
    }

    headers = create_nvcf_headers(api_key, timeout, poll_seconds=300)

    logger.info("NVCF UV generation (%s) from %s", operation, input_url[:100])
    start_time = time.time()

    try:
        result = await execute_nvcf_request_async(
            url=full_url,
            headers=headers,
            params=params,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )
    except RuntimeError as e:
        return {
            "status": "error",
            "total_time": time.time() - start_time,
            "error": str(e),
        }

    elapsed = time.time() - start_time
    logger.info("NVCF UV generation completed in %.2fs", elapsed)

    if not result.get("success"):
        return {
            "status": "error",
            "total_time": elapsed,
            "error": "UV generation failed (success=False in response)",
        }

    # Decode and write the output USD
    stage_b64 = result.get("generated_stage_base64")
    if not stage_b64:
        return {
            "status": "error",
            "total_time": elapsed,
            "error": "No generated_stage_base64 in response",
        }

    stage_bytes = base64.b64decode(stage_b64)
    output_path = _lexical_absolute_path(output_path)
    with _atomic_output_file(
        output_path,
        clear_portable_sidecar=True,
    ) as transaction_output:
        transaction_output.write_bytes(stage_bytes)
    output_size = output_path.stat().st_size

    logger.info("Wrote UV-generated USD to %s (%d bytes)", output_path, output_size)

    # Count meshes and UVs to match local backend return shape
    mesh_count = 0
    meshes_with_uvs = 0
    try:
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(output_path))
        if stage:
            for prim in stage.Traverse():
                if prim.IsA(UsdGeom.Mesh):
                    mesh_count += 1
                    st = prim.GetAttribute("primvars:st")
                    if st and st.HasAuthoredValue():
                        meshes_with_uvs += 1
    except Exception:  # noqa: BLE001
        logger.debug("Could not count meshes in NVCF output (pxr unavailable)")

    return {
        "status": "success",
        "operation": operation,
        "total_time": elapsed,
        "stage_size_bytes": output_size,
        "mesh_count": mesh_count,
        "meshes_with_uvs": meshes_with_uvs,
        "operations_executed": result.get("operations_executed", []),
    }


async def _generate_uvs_from_path(
    input_path: Path,
    output_path: Path,
    operation: str,
    op_params: dict[str, Any],
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: int = 600,
    max_retries: int = 3,
    use_data_uri: bool | None = None,
) -> dict[str, Any]:
    """Call the NVCF ``/generate-uvs`` endpoint with a local USD file.

    Handles S3 upload or data URI encoding, then delegates to
    ``_generate_uvs_from_url``.  Mirrors ``optimize_usd_from_path``
    from ``scene_optimizer_nvcf.py``.
    """
    from world_understanding.config.s3 import (
        WU_S3_BUCKET,
        WU_S3_PROFILE,
        WU_S3_REGION,
    )
    from world_understanding.utils.nvcf_utils import s3_uri_to_https_url
    from world_understanding.utils.s3_utils import delete_s3_path, upload_file_to_s3

    if not input_path.exists():
        raise ValueError(f"Input file does not exist: {input_path}")

    use_data_uri_val = should_use_data_uri(use_data_uri)

    if use_data_uri_val:
        logger.info("Using data URI for NVCF UV generation (no S3)")
        input_url = create_data_uri_from_file(input_path)
        return await _generate_uvs_from_url(
            input_url=input_url,
            output_path=output_path,
            operation=operation,
            op_params=op_params,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )

    # S3 upload path
    suffix = input_path.suffix.lower() if input_path.suffix else ".usd"
    if suffix not in (".usd", ".usda", ".usdc"):
        suffix = ".usd"
    unique_id = uuid.uuid4().hex
    s3_key = f"nvcf-uv-generation/{unique_id}/input{suffix}"
    s3_uri = None

    try:
        logger.info("Uploading input USD to S3 for UV generation...")
        s3_uri = upload_file_to_s3(
            file_path=str(input_path),
            s3_path=f"s3://{WU_S3_BUCKET}/{s3_key}",
            profile_name=WU_S3_PROFILE,
        )
        input_url = s3_uri_to_https_url(s3_uri, WU_S3_REGION)
        logger.info("Uploaded to S3: %s", input_url)

        return await _generate_uvs_from_url(
            input_url=input_url,
            output_path=output_path,
            operation=operation,
            op_params=op_params,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
        )
    finally:
        if s3_uri:
            try:
                delete_s3_path(s3_uri, profile_name=WU_S3_PROFILE)
                logger.info("Cleaned up S3 file: %s", s3_uri)
            except Exception as e:
                logger.warning("Failed to clean up S3 file %s: %s", s3_uri, e)


def _run_uv_nvcf(
    input_path: Path,
    output_path: Path,
    operation: str,
    op_params: dict[str, Any],
    timeout: int = 600,
) -> dict[str, Any]:
    """Run UV generation via NVCF cloud service.

    Synchronous wrapper around ``_generate_uvs_from_path``.
    """
    result = asyncio.run(
        _generate_uvs_from_path(
            input_path=input_path,
            output_path=output_path,
            operation=operation,
            op_params=op_params,
            timeout=timeout,
        )
    )
    if result.get("status") != "success":
        raise RuntimeError(
            f"NVCF UV generation failed: {result.get('error', 'unknown error')}"
        )
    return result


def _run_uv_worker(
    input_path: Path,
    output_path: Path,
    operation: str,
    op_params: dict[str, Any],
    timeout: int = 600,
    *,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Run a UV generation operation in the SO subprocess.

    Args:
        input_path: Path to the input USD file.
        output_path: Path for the output USD file.
        operation: SO operation name.
        op_params: Operation parameters dict (camelCase keys).
        timeout: Subprocess timeout in seconds.
        approved_dependency_roots: Filesystem roots from which the worker may
            copy dependencies into the portable output sidecar. Defaults to
            the input USD parent.

    Returns:
        Result dict with status, timing, and mesh UV stats.

    Raises:
        RuntimeError: If SO paths are missing or subprocess fails.
    """
    root_values: tuple[Path | str, ...]
    if approved_dependency_roots is None:
        root_values = (input_path.resolve().parent,)
    else:
        root_values = tuple(approved_dependency_roots)
    # Validate in the parent process before launch. In particular, a broad
    # filesystem-root approval must not be mistaken for local-backend
    # unavailability and silently sent to the remote fallback.
    from world_understanding.functions.graphics.so_export import (
        _normalize_dependency_roots,
    )

    dependency_roots = [str(root) for root in _normalize_dependency_roots(root_values)]

    try:
        so_package_dir, so_python = _resolve_so_paths()
    except RuntimeError as exc:
        raise _LocalSOUnavailableError(str(exc)) from exc

    with tempfile.TemporaryDirectory(prefix="so_uv_") as tmp_dir:
        worker_path = os.path.join(tmp_dir, "_so_uv_worker.py")
        manifest_path = os.path.join(tmp_dir, "manifest.json")

        shutil.copy2(str(_SO_UV_WORKER_PATH), worker_path)
        shutil.copy2(str(_SO_EXPORT_PATH), os.path.join(tmp_dir, "so_export.py"))

        params = {
            "input_usd_path": str(input_path),
            "output_usd_path": str(output_path),
            "approved_dependency_roots": dependency_roots,
            "operation": operation,
            "op_params": op_params,
            "manifest_path": manifest_path,
        }

        from world_understanding.functions.graphics.scene_optimizer_local import (
            _python_libdir,
        )

        env = os.environ.copy()
        lib_var = "DYLD_LIBRARY_PATH" if sys.platform == "darwin" else "LD_LIBRARY_PATH"
        # Replace (don't inherit) the parent's library path. Inheriting it
        # would let unrelated host paths satisfy missing transitive deps and
        # silently mix ABIs — see ``scene_optimizer_local._subprocess_env``
        # for the same isolation reasoning.
        lib_paths = [
            str(so_package_dir / "lib"),
            str(so_package_dir / "extraLibs"),
        ]
        # Inject the SO Python's libdir (not the parent's) so cpython-312
        # extensions in the SO bundle find ``libpython3.12.so`` even when
        # ``WU_SO_PYTHON`` points at a different interpreter.
        python_lib_dir = _python_libdir(so_python)
        if python_lib_dir:
            lib_paths.append(python_lib_dir)
        env[lib_var] = os.pathsep.join(lib_paths)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(so_package_dir / "python"), str(so_package_dir / "usdpy")]
        )
        env["PXR_PLUGINPATH_NAME"] = str(so_package_dir / "extraLibs" / "usd")

        logger.info("Running %s via SO subprocess: %s", operation, so_python)

        start_time = time.time()
        try:
            # ``-S`` keeps the parent venv's site-packages (including the
            # app's ``pxr`` provider) off the worker's ``sys.path``.
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
                f"UV generation subprocess timed out after {timeout}s"
            ) from None

        if proc.returncode != 0:
            error_msg = f"UV generation subprocess failed (exit code {proc.returncode})"
            if proc.stdout:
                error_msg += f"\n--- stdout ---\n{proc.stdout[-1000:]}"
            if proc.stderr:
                error_msg += f"\n--- stderr ---\n{proc.stderr[-2000:]}"
            if _is_uv_worker_process_startup_failure(proc.stderr or "", so_python):
                raise _LocalSOUnavailableError(error_msg)
            raise RuntimeError(error_msg)

        if not os.path.exists(manifest_path):
            raise RuntimeError(
                "UV generation subprocess did not produce manifest. "
                f"stdout: {proc.stdout[-500:]}, stderr: {proc.stderr[-500:]}"
            )

        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)

    elapsed = time.time() - start_time
    logger.info("%s completed in %.2fs", operation, elapsed)

    if manifest.get("status") != "success":
        error_msg = f"{operation} failed: {manifest.get('error', 'unknown error')}"
        if manifest.get("failure_phase") == "runtime_import" and manifest.get(
            "error_type"
        ):
            raise _LocalSOUnavailableError(error_msg)
        raise RuntimeError(error_msg)

    return manifest


def _run_with_fallback(
    input_path: Path,
    output_path: Path,
    operation: str,
    op_params: dict[str, Any],
    backend: str,
    timeout: int,
    allow_remote_fallback: bool = True,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Dispatch UV generation to local or NVCF backend.

    When ``backend="local"`` (the default), tries the local SO subprocess
    first and automatically falls back to NVCF if the local backend is
    unavailable (``WU_SO_PACKAGE_DIR`` unset, wrong paths, macOS, etc.).

    Args:
        backend: ``"local"`` (default, with NVCF auto-fallback) or ``"remote"``.
    """
    if backend == "remote":
        return _run_uv_nvcf(input_path, output_path, operation, op_params, timeout)

    if backend == "local":
        try:
            return _run_uv_worker(
                input_path,
                output_path,
                operation,
                op_params,
                timeout,
                approved_dependency_roots=approved_dependency_roots,
            )
        except (RuntimeError, FileNotFoundError) as local_err:
            # Auto-fallback to NVCF when local SO is unavailable.
            # Covers: missing bundle, missing env vars, missing directories,
            # macOS (.so/dlopen failures), missing python3.12 binary,
            # subprocess import errors.
            if _is_local_so_unavailable(local_err):
                if not allow_remote_fallback:
                    raise
                logger.warning(
                    "Local SO backend unavailable (%s), falling back to NVCF",
                    local_err,
                )
                try:
                    return _run_uv_nvcf(
                        input_path, output_path, operation, op_params, timeout
                    )
                except Exception as nvcf_err:
                    # If NVCF also fails, raise the original local error
                    # with NVCF failure noted — avoids confusing 404s when
                    # the /generate-uvs endpoint isn't deployed yet.
                    raise RuntimeError(
                        f"Local SO unavailable ({local_err}) and NVCF "
                        f"fallback also failed ({nvcf_err})"
                    ) from local_err
            raise

    raise ValueError(f"Invalid backend: '{backend}'. Must be 'local' or 'remote'.")


def generate_projection_uvs(
    input_path: Path | str,
    output_path: Path | str,
    projection_type: ProjectionType = ProjectionType.CUBE,
    paths: list[str] | None = None,
    use_world_space_scales: bool = True,
    scale_factor: float = 0.01,
    scale_units: float = 0.0,
    overwrite_existing: bool = True,
    preprojection_xform: list[float] | None = None,
    timeout: int = 600,
    backend: str = "local",
    allow_remote_fallback: bool = True,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Generate projection-based UVs on a USD file.

    Uses the Scene Optimizer's ``generateProjectionUVs`` operation.
    No external dependencies beyond USD — works on any platform.

    Args:
        input_path: Path to the input USD file.
        output_path: Path for the output USD file with UVs.
        projection_type: Projection mode (default: CUBE).
        paths: Prim paths to process (empty/None = all meshes).
        use_world_space_scales: Scale by local-to-world transform.
        scale_factor: Uniform scale for texel density.
        scale_units: 0.0 = stage units; 1.0 = meters.
        overwrite_existing: When False, skip meshes with existing UVs.
        preprojection_xform: 4x4 row-major matrix (16 floats) applied
            before projection. None/empty = identity.
        timeout: Subprocess timeout in seconds.
        backend: ``"local"`` (default, auto-falls back to NVCF) or ``"remote"``.
        allow_remote_fallback: When using ``backend="local"``, fall back to
            NVCF if the local Scene Optimizer subprocess is unavailable.
        approved_dependency_roots: Filesystem roots from which the local
            worker may copy dependencies into the portable output sidecar.
            Defaults to the input USD parent. Ignored by the remote backend.

    Returns:
        Dict with status, timing, mesh_count, meshes_with_uvs.

    Raises:
        RuntimeError: If backend is unavailable or operation fails.
    """
    op_params: dict[str, Any] = {
        "projectionType": int(projection_type),
        "useWorldSpaceScales": use_world_space_scales,
        "scaleFactor": scale_factor,
        "scaleUnits": scale_units,
        "overwriteExisting": overwrite_existing,
    }
    if paths:
        op_params["paths"] = paths
    if preprojection_xform:
        if len(preprojection_xform) != 16:
            raise ValueError(
                f"preprojection_xform must be exactly 16 floats (4x4 matrix), "
                f"got {len(preprojection_xform)}"
            )
        op_params["preprojectionXform"] = preprojection_xform

    return _run_with_fallback(
        input_path=Path(input_path),
        output_path=Path(output_path),
        operation="generateProjectionUVs",
        op_params=op_params,
        backend=backend,
        timeout=timeout,
        allow_remote_fallback=allow_remote_fallback,
        approved_dependency_roots=approved_dependency_roots,
    )


def generate_atlas_uvs(
    input_path: Path | str,
    output_path: Path | str,
    paths: list[str] | None = None,
    distortion_threshold: float = 3.0,
    enable_atlas_packing: bool = True,
    use_world_space_scales: bool = True,
    scale_factor: float = 0.01,
    scale_units: float = 0.0,
    overwrite_existing: bool = True,
    timeout: int = 600,
    backend: str = "local",
    allow_remote_fallback: bool = True,
    approved_dependency_roots: Iterable[Path | str] | None = None,
) -> dict[str, Any]:
    """Generate atlas-unwrapped UVs on a USD file.

    Uses the Scene Optimizer's ``generateAtlasUVs`` operation (autouv-core
    library, Boundary First Flattening).  Requires autouv-core — currently
    Linux only.

    Args:
        input_path: Path to the input USD file.
        output_path: Path for the output USD file with UVs.
        paths: Prim paths to process (empty/None = all meshes).
        distortion_threshold: Lower = less distortion, more UV islands.
            Internally clamped to max(1.05, value). Default: 3.0.
        enable_atlas_packing: Use hybrid atlas packing. Default: True.
        use_world_space_scales: Scale by local-to-world transform.
        scale_factor: Uniform scale for texel density.
        scale_units: 0.0 = stage units; 1.0 = meters.
        overwrite_existing: When False, skip meshes with existing UVs.
        timeout: Subprocess timeout in seconds.
        backend: ``"local"`` (default, auto-falls back to NVCF) or ``"remote"``.
        allow_remote_fallback: When using ``backend="local"``, fall back to
            NVCF if the local Scene Optimizer subprocess is unavailable.
        approved_dependency_roots: Filesystem roots from which the local
            worker may copy dependencies into the portable output sidecar.
            Defaults to the input USD parent. Ignored by the remote backend.

    Returns:
        Dict with status, timing, mesh_count, meshes_with_uvs.

    Raises:
        RuntimeError: If backend is unavailable or operation fails.
    """
    op_params: dict[str, Any] = {
        "distortionThreshold": distortion_threshold,
        "enableAtlasPacking": enable_atlas_packing,
        "useWorldSpaceScales": use_world_space_scales,
        "scaleFactor": scale_factor,
        "scaleUnits": scale_units,
        "overwriteExisting": overwrite_existing,
    }
    if paths:
        op_params["paths"] = paths

    return _run_with_fallback(
        input_path=Path(input_path),
        output_path=Path(output_path),
        operation="generateAtlasUVs",
        op_params=op_params,
        backend=backend,
        timeout=timeout,
        allow_remote_fallback=allow_remote_fallback,
        approved_dependency_roots=approved_dependency_roots,
    )
