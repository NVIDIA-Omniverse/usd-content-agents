# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX render and camera-observation backend.

The ordinary :class:`RenderBackend` surface remains RGB/beauty-only.  Camera-rig
verification additionally uses a typed observation path that requests metric,
normal, position, semantic, albedo, and beauty AOVs.  Those buffers stay on CUDA:
the isolated worker maps them into Warp through DLPack, performs compact reductions
on the mapping's synchronization stream, and copies a full image to the host only
when the caller explicitly requests an artifact.

OVRTX runs in an **isolated subprocess** with its own exact-pinned venv
(`ovrtx==0.4.1.364340`, `ovstage==0.1.1.355824`, `warp-lang==1.16.0`). OVRTX
bundles USD C libraries that clash with pxr, so the daemon boots one renderer and one
attached ovstage in that subprocess, then renders or observes on a JSON-line
stdin/stdout protocol. This is
"local OVRTX is just a local daemon" — the per-cwd usd-cli server owns one instance.

To get a correct render we author, in the main process (which has pxr), a small overlay:

  * a `RenderProduct` per camera carrying `resolution` (so OVRTX renders at the requested
    size) and `omni:rtx:rendermode` (so fast/quality actually pick an RTX mode), unlocked by
    the RTX advanced-settings API-schema prepend; and
  * the original Content Workbench `studio.exr` dome-light rig when the scene has no
    lights (else the path tracer renders black).

These are sublayered onto the exported working stage into one `_combined.usda` that the
daemon opens. Resolution/render-mode/light authoring lives here, not in the isolated venv.

⚠️ Requires Linux + NVIDIA RTX GPU + Vulkan/DISPLAY. It CANNOT run on macOS arm64; on a
non-GPU host construction/first render raises with a clear message.
"""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import math
import os
import platform
import queue
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

from usd_core.render.base import RenderResult, prepare_render_input
from usd_core.render.capabilities import (
    CAMERA_OBSERVATION_CAPABILITY,
    WARP_DLPACK_REDUCTION_CAPABILITY,
)
from usd_core.windows_files import (
    advisory_file_lock,
    open_confined_directory,
    open_confined_regular_file_at,
)

logger = logging.getLogger(__name__)

_QUALIFIED_RUNTIME_PROFILE = {
    "ovrtx": "0.4.1.364340",
    "ovstage": "0.1.1.355824",
    "warp-lang": "1.16.0",
}
_QUALIFIED_WORKER_RUNTIME_VERSIONS = {
    "ovrtx": _QUALIFIED_RUNTIME_PROFILE["ovrtx"],
    "ovstage": _QUALIFIED_RUNTIME_PROFILE["ovstage"],
    "warp": _QUALIFIED_RUNTIME_PROFILE["warp-lang"],
}
OVRTX_PIN = f"ovrtx=={_QUALIFIED_RUNTIME_PROFILE['ovrtx']}"
OVSTAGE_PIN = f"ovstage=={_QUALIFIED_RUNTIME_PROFILE['ovstage']}"
WARP_PIN = f"warp-lang=={_QUALIFIED_RUNTIME_PROFILE['warp-lang']}"

#: Written by _provision_venv after probing every worker dependency. A venv is
#: ready only when this file records the complete qualified runtime profile and
#: the digest of the exact lock that supplied its companion packages.
_READY_MARKER_NAME = ".usd-cli-ovrtx-ready"
_READY_MARKER_SCHEMA_VERSION = "usd-cli.ovrtx-runtime-ready.v3"
_LOCK_DIGEST_KEY = "runtime_lock_sha256"

#: Written before the venv is created so a run that dies mid-install still
#: leaves the tree recognisably ours. Without it a failed ~2.5 GB download
#: produces a python with no ready marker, which the ownership check below
#: would refuse to replace forever.
_PROVISIONING_MARKER_NAME = ".usd-cli-ovrtx-provisioning"

# Run-owned staging and backup trees also carry a sidecar outside the tree.
# ``shutil.rmtree`` may delete the in-tree ownership markers before failing on
# a deeper entry; the sidecar lets the next locked provisioning run recognize
# and finish cleaning that otherwise multi-gigabyte orphan.
_SIBLING_OWNER_SUFFIX = ".usd-cli-owned"
_SIBLING_OWNER_BODY = "usd-cli ovrtx runtime sibling\n"

#: world_understanding stamps its managed runtime with this. A directory can
#: carry it *and* our ready marker, because the previous usd-cli provisioner
#: installed into whatever tree was already there. Our marker is therefore not
#: proof of exclusive ownership, and wu's presence always wins.
_WU_MANAGED_MARKER_NAME = ".wu-managed-ovrtx-venv"
#: world_understanding stamps this before it installs and removes it on
#: success, so an interrupted wu provision carries only this one. Treating
#: the tree as unowned invites stamping our marker onto wu's half-built
#: runtime, or rmtree-ing it outright when we had provisioned there before.
_WU_PROVISIONING_MARKER_NAME = ".wu-managed-ovrtx-venv.provisioning"
OVRTX_INDEX = "https://pypi.nvidia.com"
#: Hash-pinned PEP 751 lock the isolated venv installs from (the full reviewed
#: runtime: ovrtx + ovstage + Warp + numpy + pillow, exact wheel URLs + sha256). Installed with
#: `uv pip install --require-hashes --no-deps --no-config --no-sources` so two
#: provisions are byte-identical and no index resolution can substitute an
#: artifact — the previous bare `pip install ... numpy pillow` floated with
#: whatever PyPI served that day. Regenerate via the command in
#: ovrtx_runtime_profile.in (the pins must match the profile constants above).
#: WU_OVRTX_RUNTIME_LOCK overrides the lock (e.g. a different platform's).
OVRTX_RUNTIME_LOCK = Path(__file__).with_name("pylock.ovrtx-runtime.toml")
_OVRTX_RUNTIME_LOCK_ENV = "WU_OVRTX_RUNTIME_LOCK"
_UV_EXECUTABLE_ENV = "USD_CLI_UV_EXECUTABLE"
# Honor WU_OVRTX_VENV_DIR so a Docker image can ship a pre-built venv (parity with WU).
DEFAULT_VENV = Path(
    os.environ.get("WU_OVRTX_VENV_DIR", str(Path.home() / ".cache" / "usd-cli" / "ovrtx_venv"))
).expanduser()

# Short mode tokens -> the omni:rtx:rendermode token the RTX engine expects.
_RENDER_MODE_TOKENS = {
    "rt1": "RaytracedLighting",
    "rt2": "RealTimePathTracing",
    "pt": "PathTracing",
}
_CAMERA_OBSERVATION_AOVS = frozenset(
    {
        "LdrColor",
        "HdrColor",
        "NormalSD",
        "DistanceToCameraSD",
        "DistanceToImagePlaneSD",
        "DiffuseAlbedoSD",
        "Camera3dPositionSD",
        "SemanticSegmentation",
        "SemanticIdMap",
    }
)
_DEFAULT_CAMERA_OBSERVATION_AOVS = (
    "LdrColor",
    "DistanceToCameraSD",
    "DistanceToImagePlaneSD",
    "Camera3dPositionSD",
    "NormalSD",
    "SemanticSegmentation",
    "SemanticIdMap",
    "DiffuseAlbedoSD",
)
_IMAGE_AOV_CHANNELS = {
    "LdrColor": 4,
    "HdrColor": 4,
    "NormalSD": 4,
    "DistanceToCameraSD": 1,
    "DistanceToImagePlaneSD": 1,
    "DiffuseAlbedoSD": 4,
    "Camera3dPositionSD": 4,
    "SemanticSegmentation": 1,
}
SEMANTIC_OVERLAY_CAPABILITY = "semantic_overlay_v1"
# usd-cli's fast/quality axis -> ovrtx short token. (`pt` reachable only via explicit config.)
_MODE_TO_RENDER_MODE = {"fast": "rt1", "quality": "rt2"}

_DEFAULT_HDRI_PATH = Path(__file__).with_name("data") / "studio.exr"
_DEFAULT_HDRI_INTENSITY = 600.0
_CUSTOM_HDRI_DEFAULT_INTENSITY = 1.0

# Applying these on the RenderProduct is required so `omni:rtx:rendermode` is honored —
# without the schema prepend the engine ignores the token. Schema names come from ovrtx's
# bundled rtx_settings plugin (generatedSchema.usda).
_RTX_RENDER_PRODUCT_API_SCHEMAS = (
    "OmniRtxSettingsCommonAdvancedAPI_1",
    "OmniRtxSettingsRtAdvancedAPI_1",
    "OmniRtxSettingsPtAdvancedAPI_1",
)

# Daemon timeouts (seconds), env-overridable for slow cold-start / heavy scenes.
_START_TIMEOUT_S = float(os.environ.get("OVRTX_DAEMON_START_TIMEOUT", "600"))
_RENDER_TIMEOUT_S = float(os.environ.get("OVRTX_DAEMON_RENDER_TIMEOUT", "1800"))
# A camera verification is deliberately non-preemptible once it enters OVRTX, so
# reject an accidentally hostile quality setting before the worker can enter an
# effectively unbounded accumulation loop.  The reviewed profile uses 64 updates;
# 1024 leaves ample explicit-quality headroom while keeping the request finite.
MAX_OVRTX_SENSOR_UPDATES = 1024


def _validated_sensor_updates(value: int) -> int:
    """Return a bounded OVRTX accumulation count before worker dispatch."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("OVRTX sensor-update count must be an integer")
    if value < 1 or value > MAX_OVRTX_SENSOR_UPDATES:
        raise ValueError(
            "OVRTX sensor-update count must be between 1 and "
            f"{MAX_OVRTX_SENSOR_UPDATES}"
        )
    return value


@dataclass(frozen=True)
class OvRTXObservationResult:
    """Compact result for one independently rendered camera.

    ``aovs`` contains JSON-safe shape/dtype/statistics records.  ``artifacts`` maps
    only explicitly requested AOVs to files; absent channels were never copied to
    host memory.  Semantic labels are decoded from ``SemanticIdMap`` by the worker
    and paired with GPU-counted segmentation pixels.
    """

    camera: str
    render_product: str
    aovs: dict[str, dict]
    artifacts: dict[str, str]
    ovrtx_render_mode: str
    ovrtx_num_sensor_updates: int
    semantic_assignments: tuple[dict[str, str], ...] = ()


def _product_path(camera_path: str) -> str:
    """Stable, collision-resistant RenderProduct path for one camera."""

    safe = camera_path.strip("/").replace("/", "_") or "camera"
    # Replacing separators alone aliases paths such as /A_B/C and /A/B_C. Keep
    # a readable prefix while binding the product identity to the exact SdfPath.
    digest = hashlib.sha256(camera_path.encode("utf-8")).hexdigest()[:16]
    return f"/Render/Product_{safe[:96]}_{digest}"


def _build_render_products_usda(
    cameras: list[str],
    width: int,
    height: int,
    render_mode: str,
    *,
    render_vars: tuple[str, ...] = ("LdrColor",),
) -> tuple[str, list[str]]:
    """USDA overlay defining one RenderProduct per camera + the product prim paths.

    Each product references its camera, pins `resolution`, and carries
    `omni:rtx:rendermode` (unlocked by the API-schema prepend). The ordinary RGB path
    uses the default `LdrColor`; camera verification can request the qualified metric,
    semantic, normal, position, albedo, and beauty channels explicitly.
    """
    if render_mode not in _RENDER_MODE_TOKENS:
        raise ValueError(
            f"unknown render_mode {render_mode!r} (expected one of {sorted(_RENDER_MODE_TOKENS)})"
        )
    token = _RENDER_MODE_TOKENS[render_mode]
    api_schemas = ", ".join(f'"{s}"' for s in _RTX_RENDER_PRODUCT_API_SCHEMAS)
    if not render_vars:
        raise ValueError("at least one OVRTX render variable is required")
    unknown = sorted(set(render_vars) - _CAMERA_OBSERVATION_AOVS)
    if unknown:
        raise ValueError(f"unsupported OVRTX camera observation AOV: {unknown[0]}")
    if len(set(render_vars)) != len(render_vars):
        raise ValueError("OVRTX camera observation AOVs must be unique")
    ordered_vars = ", ".join(f"</Render/Vars/{name}>" for name in render_vars)

    product_paths: list[str] = []
    product_defs: list[str] = []
    for cam in cameras:
        path = _product_path(cam)
        product_paths.append(path)
        name = path.rsplit("/", 1)[-1]
        product_defs.append(
            f'    def RenderProduct "{name}" (\n'
            f"        prepend apiSchemas = [{api_schemas}]\n"
            f"    )\n"
            f"    {{\n"
            f"        rel camera = <{cam}>\n"
            f"        rel orderedVars = [{ordered_vars}]\n"
            f"        uniform int2 resolution = ({width}, {height})\n"
            f'        token omni:rtx:rendermode = "{token}"\n'
            f"    }}\n"
        )

    render_var_defs = "".join(
        f'        def RenderVar "{name}"\n'
        "        {\n"
        f'            uniform string sourceName = "{name}"\n'
        "        }\n"
        for name in render_vars
    )
    usda = (
        "#usda 1.0\n"
        "(\n)\n\n"
        'def Scope "Render"\n'
        "{\n"
        f"{chr(10).join(product_defs)}\n"
        '    def Scope "Vars"\n'
        "    {\n"
        f"{render_var_defs}"
        "    }\n"
        "}\n"
    )
    return usda, product_paths


def _stage_has_lights(stage) -> bool:
    """True if the stage already has at least one UsdLux light prim.

    Traverses instance proxies — a light packaged inside an instanceable asset still lights
    the render, and missing it would stack the default dome on top.
    """
    from pxr import Usd, UsdLux

    return any(
        p.IsA(UsdLux.BoundableLightBase) or p.IsA(UsdLux.NonboundableLightBase)
        for p in stage.Traverse(Usd.TraverseInstanceProxies())
    )


def _default_hdri_path() -> Path:
    """Return the bundled Workbench studio HDRI, or its explicit compatibility override."""
    raw = os.environ.get("WU_OVRTX_DEFAULT_HDRI", "").strip()
    path = Path(raw).expanduser() if raw else _DEFAULT_HDRI_PATH
    path = path.resolve()
    if not path.is_file():
        raise RuntimeError(f"OVRTX default HDRI does not exist: {path}")
    if any(char in str(path) for char in ("@", "\n", "\r")):
        raise RuntimeError(f"OVRTX default HDRI path cannot be represented in USDA: {path}")
    return path


def _default_hdri_intensity() -> float:
    """Return the Workbench-compatible intensity for the effective HDRI."""

    hdri_path = _default_hdri_path()
    fallback = (
        _DEFAULT_HDRI_INTENSITY
        if hdri_path == _DEFAULT_HDRI_PATH.resolve()
        else _CUSTOM_HDRI_DEFAULT_INTENSITY
    )
    raw = os.environ.get("OVRTX_DEFAULT_HDRI_INTENSITY")
    if raw is None:
        raw = os.environ.get("WU_OVRTX_DEFAULT_HDRI_INTENSITY")
    if raw is None or not raw.strip():
        return fallback
    try:
        return float(raw)
    except ValueError:
        logger.warning(
            "Invalid OVRTX default HDRI intensity %r; using default %g",
            raw,
            fallback,
        )
        return fallback


def _build_default_lights_usda(intensity: float) -> str:
    """Build the original Content Workbench studio dome-light overlay."""
    hdri_asset = _default_hdri_path()
    return (
        "#usda 1.0\n"
        "(\n)\n\n"
        'def "OvRTXDefaultLights" (\n'
        "    hide_in_stage_window = true\n"
        "    no_delete = true\n"
        ")\n"
        "{\n"
        '    def DomeLight "DomeLight"\n'
        "    {\n"
        f"        float inputs:intensity = {float(intensity)}\n"
        '        token inputs:texture:format = "latlong"\n'
        f"        asset inputs:texture:file = @{hdri_asset}@\n"
        "        custom bool visibleInPrimaryRay = 0\n"
        "    }\n"
        "}\n"
    )


def _validate_ovrtx_cameras(stage, cameras: list[str]) -> None:
    """Reject camera paths that OVRTX cannot address independently."""

    from pxr import Sdf, UsdGeom

    for camera_path in cameras:
        prim = stage.GetPrimAtPath(Sdf.Path(camera_path))
        if (
            not prim.IsValid()
            or not prim.IsA(UsdGeom.Camera)
            or prim.IsInstanceProxy()
        ):
            raise ValueError(
                "OVRTX camera does not exist, is not a camera, or is an "
                f"instance proxy: {camera_path}"
            )


def _write_camera_aspect_overlay(
    stage, cameras: list[str], width: int, height: int, path: Path
) -> None:
    """Author render-only filmback overrides without touching the caller's stage."""

    from pxr import Sdf, UsdGeom

    # Validate before creating the disposable layer so rejected paths leave no
    # workflow-owned artifact behind, even when this helper is called directly.
    _validate_ovrtx_cameras(stage, cameras)
    layer = Sdf.Layer.CreateNew(str(path))
    if layer is None:
        raise RuntimeError(f"could not create OVRTX camera overlay: {path}")
    for camera_path in cameras:
        prim = stage.GetPrimAtPath(Sdf.Path(camera_path))
        camera = UsdGeom.Camera(prim)
        horizontal = camera.GetHorizontalApertureAttr().Get()
        if horizontal is None:
            horizontal = 20.955
        spec = Sdf.CreatePrimInLayer(layer, Sdf.Path(camera_path))
        spec.specifier = Sdf.SpecifierOver
        attribute = Sdf.AttributeSpec(
            spec,
            "verticalAperture",
            Sdf.ValueTypeNames.Float,
            Sdf.VariabilityVarying,
        )
        attribute.default = float(horizontal) * int(height) / int(width)
    layer.Save()


def _write_semantic_overlay(
    stage, semantic_labels: dict[str, str], path: Path
) -> list[dict[str, str]]:
    """Author scoped SemanticsAPI labels into a stronger, disposable layer.

    The source stage is never edited.  Labels and paths are returned in canonical
    order so evidence can bind the requested USD identity to the renderer's dynamic
    32-bit semantic IDs.  Instance-proxy descendants cannot receive independent USD
    opinions and therefore fail explicitly instead of being mislabeled at a prototype.
    """

    from pxr import Sdf

    if len(semantic_labels) > 256:
        raise ValueError("OVRTX verification accepts at most 256 semantic labels")
    layer = Sdf.Layer.CreateNew(str(path))
    if layer is None:
        raise RuntimeError(f"could not create OVRTX semantic overlay: {path}")
    assignments: list[dict[str, str]] = []
    for prim_path, label in sorted(semantic_labels.items()):
        if not isinstance(prim_path, str) or not Sdf.Path.IsValidPathString(prim_path):
            raise ValueError(f"invalid OVRTX semantic prim path: {prim_path!r}")
        parsed = Sdf.Path(prim_path)
        if not parsed.IsAbsolutePath() or not parsed.IsPrimPath():
            raise ValueError(f"OVRTX semantic path must be an absolute prim path: {prim_path}")
        prim = stage.GetPrimAtPath(parsed)
        if not prim.IsValid():
            raise ValueError(f"OVRTX semantic prim does not exist: {prim_path}")
        if prim.IsInstanceProxy():
            raise ValueError(
                "OVRTX semantic overlays cannot target an instance-proxy descendant: "
                f"{prim_path}; label the occurrence's instance root"
            )
        if not isinstance(label, str) or not label or len(label.encode("utf-8")) > 1024:
            raise ValueError(f"OVRTX semantic label for {prim_path} is empty or too long")
        if "\x00" in label:
            raise ValueError(f"OVRTX semantic label for {prim_path} contains NUL")
        spec = Sdf.CreatePrimInLayer(layer, parsed)
        spec.specifier = Sdf.SpecifierOver
        schemas = Sdf.TokenListOp()
        schemas.prependedItems = ["SemanticsAPI:usd_cli"]
        spec.SetInfo("apiSchemas", schemas)
        data = Sdf.AttributeSpec(
            spec,
            "semantic:usd_cli:params:semanticData",
            Sdf.ValueTypeNames.String,
            Sdf.VariabilityUniform,
        )
        data.default = label
        semantic_type = Sdf.AttributeSpec(
            spec,
            "semantic:usd_cli:params:semanticType",
            Sdf.ValueTypeNames.String,
            Sdf.VariabilityUniform,
        )
        semantic_type.default = "usd_cli"
        assignments.append({"path": parsed.pathString, "label": label})
    layer.Save()
    return assignments


def _validated_observation_aovs(aovs) -> tuple[str, ...]:
    requested = tuple(str(name) for name in (aovs or _DEFAULT_CAMERA_OBSERVATION_AOVS))
    if not requested:
        raise ValueError("OVRTX camera observation requires at least one AOV")
    if len(set(requested)) != len(requested):
        raise ValueError("OVRTX camera observation AOVs must be unique")
    unknown = sorted(set(requested) - _CAMERA_OBSERVATION_AOVS)
    if unknown:
        raise ValueError(f"unsupported OVRTX camera observation AOV: {unknown[0]}")
    if ("SemanticSegmentation" in requested) != ("SemanticIdMap" in requested):
        raise ValueError(
            "SemanticSegmentation and SemanticIdMap must be requested together"
        )
    return requested


def _observation_artifact_suffix(aov: str) -> str:
    if aov in {"LdrColor", "DiffuseAlbedoSD"}:
        return ".png"
    if aov == "SemanticIdMap":
        return ".json"
    return ".npy"


def _safe_output_stems(cameras: list[str], names: list[str] | None) -> list[str]:
    stems = (
        [str(name) for name in names]
        if names is not None
        else [camera.strip("/").replace("/", "_") or "camera" for camera in cameras]
    )
    if len(stems) != len(cameras):
        raise ValueError("OVRTX output names and cameras disagree")
    for stem in stems:
        if (
            not stem
            or stem in {".", ".."}
            or Path(stem).name != stem
            or "/" in stem
            or "\\" in stem
            or "\x00" in stem
        ):
            raise ValueError(f"unsafe OVRTX output artifact name: {stem!r}")
    if len(set(stems)) != len(stems):
        raise ValueError("OVRTX output artifact names must be unique")
    return stems


def _combined_stage_metadata(stage) -> str:
    """Root metadata that USD does not inherit from sublayers.

    In particular, dropping ``metersPerUnit`` made OVRTX metric AOVs differ from
    Newton by exactly 100× for metre-authored stages because the synthetic root
    silently fell back to USD's centimetre default.
    """

    from pxr import UsdGeom

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(meters_per_unit) or meters_per_unit <= 0.0:
        raise ValueError("OVRTX stage metersPerUnit must be finite and positive")
    up_axis = str(UsdGeom.GetStageUpAxis(stage)).upper()
    if up_axis not in {"Y", "Z"}:
        raise ValueError(f"OVRTX stage has unsupported up axis: {up_axis!r}")
    lines = [
        f"    metersPerUnit = {meters_per_unit:.17g}\n",
        f'    upAxis = "{up_axis}"\n',
    ]
    try:
        lines.extend(
            [
                f"    timeCodesPerSecond = {stage.GetTimeCodesPerSecond()}\n",
                f"    framesPerSecond = {stage.GetFramesPerSecond()}\n",
                f"    startTimeCode = {stage.GetStartTimeCode()}\n",
                f"    endTimeCode = {stage.GetEndTimeCode()}\n",
            ]
        )
    except Exception:  # noqa: BLE001 — stage without time metadata
        pass
    return "".join(lines)


def _pinned_ovrtx_version() -> str:
    """The bare version from ``OVRTX_PIN`` (``ovrtx==X`` -> ``X``)."""
    return OVRTX_PIN.partition("==")[2]


def _ovrtx_venv_python_path(venv_dir: Path) -> Path:
    """Return the platform-specific Python executable for an OVRTX venv."""

    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _runtime_lock_digest(runtime_lock: Path) -> str | None:
    """SHA-256 of the lock as installed, or None when it cannot be read."""
    try:
        return hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
    except OSError:
        return None


def _readiness_marker_body(
    runtime_lock: Path, *, runtime_lock_digest: str | None = None
) -> str:
    """Return the canonical readiness record for one qualified runtime.

    Provisioning passes the digest it captured before installation, after
    proving the lock still has that identity. Other callers may resolve it
    here, but an unreadable lock can never produce a ready marker.
    """
    digest = runtime_lock_digest
    if digest is None:
        digest = _runtime_lock_digest(runtime_lock)
    if digest is None:
        raise RuntimeError(
            f"{runtime_lock} could not be read to record its identity in the "
            "readiness marker"
        )
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError(
            f"{runtime_lock} produced an invalid SHA-256 identity for the "
            "readiness marker"
        )
    return (
        json.dumps(
            {
                "packages": _QUALIFIED_RUNTIME_PROFILE,
                _LOCK_DIGEST_KEY: digest,
                "schema_version": _READY_MARKER_SCHEMA_VERSION,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )


def _venv_matches_pin(venv_dir: Path) -> bool:
    """Whether a provisioned venv carries the complete qualified profile marker.

    The former marker contained only ``OVRTX_PIN``.  That is now deliberately
    treated as legacy: the observation worker also imports ovstage and Warp, so
    an ovrtx-only marker cannot attest that the worker is runnable.
    """
    py = _ovrtx_venv_python_path(venv_dir)
    marker = venv_dir / _READY_MARKER_NAME
    if not (py.exists() and marker.exists()):
        return False
    try:
        runtime_lock = _ovrtx_runtime_lock()
    except RuntimeError:
        # The lock is part of readiness. If it cannot be resolved, neither the
        # native profile nor its numpy/Pillow companions can be proven current.
        return False
    try:
        expected = _readiness_marker_body(runtime_lock)
    except RuntimeError:
        return False
    try:
        recorded = marker.read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return False
    if recorded == expected:
        return True
    logger.info(
        "ovrtx venv at %s has a stale or incomplete runtime marker; it "
        "will be reprovisioned for schema %s and profile %s",
        venv_dir,
        _READY_MARKER_SCHEMA_VERSION,
        _QUALIFIED_RUNTIME_PROFILE,
    )
    return False


_RUNTIME_PROBE_SCRIPT = "; ".join(
    (
        "import importlib.metadata as metadata",
        "import json",
        "import numpy, ovrtx, ovstage, warp",
        "from PIL import Image",
        "profile = {name: metadata.version(name) for name in "
        f"{tuple(_QUALIFIED_RUNTIME_PROFILE)!r}" + "}",
        f"expected = {_QUALIFIED_RUNTIME_PROFILE!r}",
        "assert profile == expected, (profile, expected)",
        "print(json.dumps(profile, sort_keys=True))",
    )
)


def _installed_runtime_profile(py: Path) -> dict[str, str] | None:
    """Return the exact worker profile importable by ``py``, or ``None``.

    Importing every module catches incomplete installs; reading every distribution
    version prevents a compatible-looking ovrtx wheel from masking a stale ovstage or
    Warp.  The worker uses the same imports, under the same sanitized environment.
    """

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    try:
        proc = subprocess.run(
            [str(py), "-I", "-c", _RUNTIME_PROBE_SCRIPT],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
            check=False,
        )
    except Exception:  # noqa: BLE001 - an unrunnable interpreter is not ready
        return None
    if proc.returncode != 0:
        return None
    output = proc.stdout.strip()
    if not output:
        return None
    try:
        profile = json.loads(output.splitlines()[-1])
    except (json.JSONDecodeError, TypeError):
        return None
    if profile != _QUALIFIED_RUNTIME_PROFILE:
        return None
    return dict(profile)


def _installed_ovrtx_version(py: Path) -> str | None:
    """Return the OVRTX version only when the complete worker profile qualifies.

    Keep the historical scalar return value for callers and tests that report the
    renderer version; qualification itself is performed by
    :func:`_installed_runtime_profile` across ovrtx, ovstage, and Warp.
    """

    profile = _installed_runtime_profile(py)
    return profile["ovrtx"] if profile is not None else None


def _env_ovrtx_python() -> str | None:
    """The current interpreter, iff it can already run the daemon script.

    Covers `uv sync --extra ovrtx` / a manual `pip install ovrtx` into the project env —
    users who did that expect it to be used, not to be told ovrtx "is not installed".
    Sharing the interpreter is safe because the daemon subprocess never imports pxr
    (the pxr↔ovrtx clash is per-process). A broken or incomplete side-by-side
    install fails the complete runtime probe and falls back to the isolated venv.
    """
    if _installed_ovrtx_version(Path(sys.executable)) is None:
        return None
    return sys.executable


def _shared_runtime_matching_pin(venv_dir: Path) -> str | None:
    """A runtime another provisioner owns, reusable because it matches the pin.

    The CAD docs tell operators to export WU_OVRTX_VENV_DIR at
    world_understanding's managed runtime, and usd-cli reads the same variable.
    That tree must never be provisioned into or deleted, but refusing it
    outright breaks a configuration the docs instruct. When its interpreter
    already has the complete qualified profile, reading it is safe and is what
    was asked for.
    """
    py = _ovrtx_venv_python_path(venv_dir)
    if not (py.exists() and (venv_dir / _WU_MANAGED_MARKER_NAME).exists()):
        return None
    if _installed_ovrtx_version(py) is None:
        return None
    logger.info(
        "using the world-understanding managed ovrtx runtime at %s: it "
        "already has %s, and usd-cli will not modify it",
        venv_dir,
        _QUALIFIED_RUNTIME_PROFILE,
    )
    return str(py)


def _ovrtx_python(venv_dir: Path, auto_install: bool = False,
                  block: bool = True) -> str:
    """Locate (or provision) the isolated ovrtx venv and return its python exe.

    Resolution order: a provisioned isolated venv (the pinned, known-good install —
    Docker images pre-ship one via WU_OVRTX_VENV_DIR) -> ovrtx importable in the
    current environment (see _env_ovrtx_python) -> provision the isolated venv if
    allowed, else fail fast.

    Readiness is gated on a marker written only after a successful install — the venv
    python existing is not enough (a half-provisioned venv must be repaired, not trusted).
    The reviewed lock contains every exact runtime artifact and a file lock serializes
    provisioning so two processes or threads cannot corrupt one venv with concurrent
    installs.

    Provisioning means pip downloading a ~2.5 GB ovrtx wheel — silently kicking that off
    at render time hung `render` for ~10 minutes with no output (benchmark task-13), so
    it only runs when explicitly allowed: `auto_install=True` (render.ovrtx_auto_install
    in config, or a warm-up path like the rendering service's `ensure_ready`), or the
    WU_OVRTX_AUTO_PROVISION env var, which when set overrides in either direction
    ("1" = allow, anything else = refuse).
    """
    py = _ovrtx_venv_python_path(venv_dir)
    if _venv_matches_pin(venv_dir):
        return str(py)
    shared = _shared_runtime_matching_pin(venv_dir)
    if shared:
        return shared
    # A superseded venv and an absent one both fall through here, but they need
    # different actions from the operator, and the mismatch is otherwise only
    # visible at a log level the CLI does not show by default.
    stale_venv = py.exists() and (venv_dir / _READY_MARKER_NAME).exists()
    env_py = _env_ovrtx_python()
    if env_py:
        return env_py
    env_flag = os.environ.get("WU_OVRTX_AUTO_PROVISION")
    allowed = (env_flag == "1") if env_flag is not None else auto_install
    if not allowed:
        # Before printing a bootstrap recipe, check whose runtime this is.
        # WU_OVRTX_VENV_DIR may point at world_understanding's managed venv
        # -- the CAD docs tell operators to do exactly that -- and --clear
        # would erase ~2.5 GB plus its bookkeeping.
        # either marker: an interrupted wu provision leaves only the
        # in-flight one, and that tree is no more ours to clear
        wu_marker = next(
            (venv_dir / n for n in (_WU_MANAGED_MARKER_NAME,
                                    _WU_PROVISIONING_MARKER_NAME)
             if (venv_dir / n).exists()), None)
        if wu_marker is not None:
            raise RuntimeError(
                f"ovrtx renderer is not usable: {venv_dir} is the "
                f"world-understanding managed ovrtx runtime "
                f"({wu_marker.name} is present) and does not "
                "match the exact qualified profile "
                f"({OVRTX_PIN}, {OVSTAGE_PIN}, {WARP_PIN}). Do not --clear "
                "it; reprovision it "
                f"with world_understanding:\n  python -m "
                f"world_understanding.functions.graphics.render_ovrtx "
                f"--provision-only --ovrtx-venv-dir "
                f"{shlex.quote(str(venv_dir))}\n"
                "Or point WU_OVRTX_VENV_DIR at a usd-cli-owned "
                "directory.")
        # A copy-paste bootstrap recipe is offered only for an absent or empty
        # directory. A typo'd WU_OVRTX_VENV_DIR that resolves to $HOME must
        # never come with a command that modifies it.
        # Name the lock provisioning would actually use. WU_OVRTX_RUNTIME_LOCK
        # overrides the shipped one, and an operator following a recipe that
        # names a different lock installs a different set of artifacts than
        # the automated path would -- then stamps readiness on the result.
        # Do not quietly fall back to the shipped lock: the recipe would then
        # install artifacts the operator did not choose and stamp the pin on
        # the result, and the bad override would never be noticed.
        try:
            selected_lock = _ovrtx_runtime_lock()
            lock_error = None
        except RuntimeError as bad_lock:
            selected_lock = None
            lock_error = str(bad_lock)
        owned = ((venv_dir / _READY_MARKER_NAME).exists()
                 or (venv_dir / _PROVISIONING_MARKER_NAME).exists())
        vacant = not venv_dir.exists() or not _has_entries(venv_dir)
        # The recipe ends by writing the complete profile and selected lock
        # digest into the readiness marker, which _venv_matches_pin trusts.
        # A selected lock that cannot produce that profile must not come with
        # a recipe at all. The automated path applies the same refusal.
        # Report both axes. `or` kept only the first, so an operator whose
        # lock names the wrong ovrtx AND needs another interpreter fixed
        # one and hit the same refusal with the other reason hidden.
        if selected_lock is None:
            selected_digest = None
            lock_unusable = ""
        else:
            selected_digest = _runtime_lock_digest(selected_lock)
            lock_reasons: tuple[str | None, str | None, str | None] = (
                _lock_provides_pin(selected_lock),
                _interpreter_satisfies_lock(selected_lock),
                None
                if selected_digest is not None
                else f"{selected_lock} could not be read to record its "
                "identity in the readiness marker",
            )
            lock_unusable = "; ".join(
                reason for reason in lock_reasons if reason is not None
            )
        # uv is not a usd-cli dependency. Resolve its concrete command before
        # offering a recipe rather than handing the operator a command that is
        # known to fail after creating a partial venv.
        try:
            recipe_uv = shlex.join(_uv_command())
        except RuntimeError as missing_uv:
            recipe_uv = None
            uv_reason = str(missing_uv)
        # _provision_venv refuses this state outright; the message must not
        # then hand over the command. auto_install is off by default, so
        # this branch is the common path, not an edge case.
        if lock_error:
            recipe = f"no recipe is offered: {lock_error}{chr(10)}"
        elif _is_active_environment(venv_dir):
            recipe = (
                f"no recipe is offered: {venv_dir} is the environment "
                f"usd-cli is running from. Clearing it would remove this "
                f"interpreter, usd_core and everything else installed "
                f"beside them. Point WU_OVRTX_VENV_DIR at a directory of "
                f"its own.{chr(10)}")
        elif lock_unusable:
            recipe = (
                f"no recipe is offered: {lock_unusable}. A recipe built on "
                "this lock could not produce the qualified OVRTX runtime "
                "profile here, and its last step records that profile, so every later "
                f"render would trust a runtime that does not match. Point "
                f"WU_OVRTX_RUNTIME_LOCK at a lock for "
                f"{_QUALIFIED_RUNTIME_PROFILE!r} on this "
                f"host, or unset it to use the shipped one.{chr(10)}")
        elif recipe_uv is None:
            recipe = (
                f"no recipe is offered: {uv_reason}\n")
        elif (
            vacant
            and selected_lock is not None
            and selected_digest is not None
        ):
            # This recipe is only offered for an absent or empty directory.
            # A stale owned runtime is replaced by the automated provisioner,
            # which builds and verifies a sibling before moving either tree.
            # Keeping --clear out of the copy-paste path means a later change
            # to the vacancy predicate cannot turn the recipe destructive.
            q_py = shlex.quote(sys.executable)
            q_dir = shlex.quote(str(venv_dir))
            q_venv_py = shlex.quote(str(py))
            q_lock = shlex.quote(str(selected_lock))
            q_marker = shlex.quote(str(venv_dir / _READY_MARKER_NAME))
            q_provisioning = shlex.quote(
                str(venv_dir / _PROVISIONING_MARKER_NAME)
            )
            q_marker_value = shlex.quote(
                _readiness_marker_body(
                    selected_lock, runtime_lock_digest=selected_digest
                ).rstrip("\n")
            )
            q_probe = shlex.quote(_RUNTIME_PROBE_SCRIPT)
            recipe = (
                f"or into an isolated venv from the reviewed hash-pinned "
                f"lock with:\n"
                f"  {q_py} -m venv {q_dir} \\\n"
                f"    && printf '%s' '{OVRTX_PIN}' > {q_provisioning} \\\n"
                f"    && {recipe_uv} pip install --python {q_venv_py} "
                f"--require-hashes \\\n"
                f"       --no-deps --no-config --no-sources \\\n"
                f"       -r {q_lock} \\\n"
                f"    && {q_venv_py} -I -c {q_probe} \\\n"
                f"    && printf '%s\\n' {q_marker_value} > "
                f"{q_marker}\n"
                f"  (only for this absent or empty directory, on an "
                f"interpreter the lock supports)\n")
        elif vacant:
            recipe = (
                "no recipe is offered: the selected runtime lock could "
                f"not be validated.{chr(10)}"
            )
        elif owned:
            recipe = (
                f"no destructive recipe is offered for the existing "
                f"usd-cli-owned runtime at {venv_dir}. Set "
                f"WU_OVRTX_AUTO_PROVISION=1 and retry the command; the "
                f"automated provisioner builds and verifies a sibling venv "
                f"before replacing this one.{chr(10)}")
        else:
            recipe = (
                f"{venv_dir} holds files usd-cli did not create, so no "
                f"recipe that clears it is offered here: point "
                f"WU_OVRTX_VENV_DIR at a fresh directory and provision "
                f"that instead.\n")
        state = (
            f"the venv at {venv_dir} was provisioned for a different or incomplete "
            "OVRTX runtime profile and must be reprovisioned"
            if stale_venv
            else f"no provisioned venv at {venv_dir}"
        )
        raise RuntimeError(
            f"ovrtx renderer is not usable ({state}, and "
            "ovrtx is not importable in the current environment), and installing it "
            "means downloading a ~2.5 GB wheel — refusing to start that mid-render. "
            "Pre-install it into the project environment with:\n"
            "  uv sync --extra ovrtx        (or: pip install --extra-index-url "
            f"{shlex.quote(os.environ.get('WU_OVRTX_INDEX_URL', OVRTX_INDEX))} "
            f"'{OVRTX_PIN}' '{OVSTAGE_PIN}' '{WARP_PIN}')\n"
            f"{recipe}"
            "or set render.ovrtx_auto_install = true (config) / WU_OVRTX_AUTO_PROVISION=1 "
            "(env) to allow the one-time download (both are read by the per-project "
            "daemon — `usd-cli server stop` first so it restarts with the new setting), or "
            "render with --renderer remote instead.")
    if block:
        _provision_venv(venv_dir)
        return str(py)
    # Non-blocking (the per-project daemon's render path): a synchronous ~10-min
    # pip download inside the command handler held the session/root lock and
    # WEDGED the daemon — every other command queued behind the install. Hand
    # provisioning to a background thread and fail THIS command fast with
    # progress; the daemon stays responsive and a retry lands on the ready venv.
    with _PROVISION_LOCK:
        th = _PROVISION["thread"]
        if th is not None and not th.is_alive():
            err, _PROVISION["thread"], _PROVISION["error"] = \
                _PROVISION["error"], None, None
            if err is not None:
                raise RuntimeError(
                    f"ovrtx auto-install FAILED: {err} — full log in the daemon "
                    "log; fix the cause and render again to retry")
            if _venv_matches_pin(venv_dir):
                return str(py)
            th = None  # finished without error but venv not ready: start over
        if th is None:
            th = threading.Thread(target=_provision_worker, args=(venv_dir,),
                                  name="ovrtx-provision", daemon=True)
            _PROVISION.update(thread=th, started=time.monotonic(), error=None)
            th.start()
            raise RuntimeError(
                "ovrtx auto-install STARTED in the background (~2.5 GB one-time "
                "download; replacing an existing runtime may temporarily need "
                "~5 GB free while the old and staged environments coexist; "
                "progress lines stream into the daemon log) — the "
                "daemon stays responsive; retry this render in a few minutes, "
                "or use --renderer remote meanwhile")
        raise RuntimeError(
            f"ovrtx auto-install in progress "
            f"({time.monotonic() - _PROVISION['started']:.0f}s elapsed; progress "
            "in the daemon log) — retry when it completes, or use --renderer "
            "remote meanwhile")


#: background-provisioning state — one install at a time per process
_PROVISION_LOCK = threading.Lock()
_PROVISION: dict = {"thread": None, "started": 0.0, "error": None}


def _provision_worker(venv_dir: Path) -> None:
    try:
        _provision_venv(venv_dir)
    except Exception as exc:  # noqa: BLE001 — recorded for the next render attempt
        logger.error("ovrtx auto-install failed: %s", exc)
        with _PROVISION_LOCK:
            _PROVISION["error"] = str(exc)


def _ovrtx_runtime_lock() -> Path:
    """Return the hash-pinned ovrtx runtime lock, failing closed when absent."""
    override = os.environ.get(_OVRTX_RUNTIME_LOCK_ENV)
    if override:
        lock = Path(override).expanduser()
        if not lock.is_file():
            raise RuntimeError(
                f"{_OVRTX_RUNTIME_LOCK_ENV} does not point to a readable "
                f"ovrtx runtime lock: {lock}")
        return lock
    if not OVRTX_RUNTIME_LOCK.is_file():
        raise RuntimeError(
            "No hash-pinned ovrtx runtime lock was found. Auto-provisioning "
            "installs only exact, reproducible artifacts; restore "
            f"{OVRTX_RUNTIME_LOCK} (shipped with usd-cli), set "
            f"{_OVRTX_RUNTIME_LOCK_ENV} to a pylock file, or pre-provision the "
            "venv and mark it ready.")
    return OVRTX_RUNTIME_LOCK


def _uv_command() -> list[str]:
    """Return an invocation of ``uv`` without trusting the ambient PATH alone."""
    import importlib.util
    pinned_executable = os.environ.get(_UV_EXECUTABLE_ENV)
    if pinned_executable:
        executable_path = Path(pinned_executable).expanduser()
        if (
            not executable_path.is_absolute()
            or not executable_path.is_file()
            or not os.access(executable_path, os.X_OK)
        ):
            raise RuntimeError(
                f"{_UV_EXECUTABLE_ENV} does not identify an executable file: "
                f"{pinned_executable}"
            )
        return [str(executable_path.resolve(strict=True))]
    if importlib.util.find_spec("uv") is not None:
        return [sys.executable, "-m", "uv"]
    executable = shutil.which("uv")
    if executable:
        return [executable]
    raise RuntimeError(
        "uv is required to provision the pinned ovrtx runtime venv "
        "(dependencies are never installed with bare pip). Install uv or "
        "pre-provision the venv and mark it ready.")


def _wheel_runs_here(reference: str) -> bool:
    """Whether a wheel reference -- url or local path -- fits this machine.

    Deliberately permissive: a platform-agnostic wheel or an architecture we
    do not recognise returns True and lets uv decide. It exists to catch the
    one case that costs a runtime -- a lock built for another architecture,
    which today reaches uv only after the rmtree has already run.
    """
    name = reference.replace(chr(92), "/").rsplit("/", 1)[-1].lower()
    if "-any.whl" in name:
        return True
    aliases = {
        "x86_64": ("x86_64", "amd64"),
        "amd64": ("x86_64", "amd64"),
        "aarch64": ("aarch64", "arm64"),
        "arm64": ("aarch64", "arm64"),
    }.get(platform.machine().lower())
    if aliases is None:
        return True
    if not any(alias in name for alias in aliases):
        return False
    # Architecture alone is not enough: win_amd64 and manylinux_x86_64
    # share the amd64/x86_64 token, so a Windows-only lock passed on Linux
    # and the rmtree ran before uv rejected the wheel.
    families = {
        "linux": ("linux",),      # manylinux/musllinux both carry it
        "win32": ("win",),
        "cygwin": ("win",),
        "darwin": ("macosx",),
    }.get(sys.platform)
    if families is None:
        return True
    return any(family in name for family in families)


def _is_active_environment(venv_dir: Path) -> bool:
    """Whether this process is running out of that directory.

    Deleting it removes sys.executable and everything the running process
    depends on, mid-render. A stale marker is enough to get there: the
    provisioner this PR replaced installed in place, so WU_OVRTX_VENV_DIR
    aimed at the project venv still carries our readiness marker, and the
    version check then routes it straight to the replacement path."""
    try:
        target = venv_dir.expanduser().resolve(strict=False)
        running = Path(sys.executable).resolve(strict=False)
        prefix = Path(sys.prefix).resolve(strict=False)
    except OSError:  # pragma: no cover - resolve failing is not 'inactive'
        return True
    return any(target == p or target in p.parents for p in (running, prefix))


def _has_entries(directory: Path) -> bool:
    """Whether a directory holds anything, treating unreadable as occupied.

    Every caller is deciding whether a tree is ours to touch. A permission
    error is not an empty directory, and letting it escape replaces the
    actionable message with a traceback."""
    try:
        return any(directory.iterdir())
    except OSError:
        return True


_ActivationTarget = tuple[tuple[int, int] | None, bool, bool]
_RuntimePathIdentity = tuple[int, int]


def _validate_runtime_parent(
    parent: Path,
    expected_identity: _RuntimePathIdentity | None = None,
) -> _RuntimePathIdentity:
    """Require a stable parent that another OS identity cannot redirect."""
    try:
        metadata = parent.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect the ovrtx runtime parent at {parent}: {exc}"
        ) from exc
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError(f"ovrtx runtime parent is not a directory: {parent}")
    identity = (metadata.st_dev, metadata.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError(
            f"refusing to continue ovrtx provisioning at {parent}: the runtime "
            "parent changed while the staged environment was being built"
        )

    get_euid = getattr(os, "geteuid", None)
    if get_euid is None:
        return identity
    euid = get_euid()
    if metadata.st_uid != euid:
        raise RuntimeError(
            f"ovrtx runtime parent {parent} is owned by uid {metadata.st_uid}, "
            f"not the current uid {euid}"
        )
    if stat.S_IMODE(metadata.st_mode) & 0o022:
        raise RuntimeError(
            f"ovrtx runtime parent {parent} is group- or world-writable; "
            "choose an owner-controlled directory"
        )

    # A private directory is still replaceable if one of its ancestors lets a
    # different user rename the directory entry. A sticky shared ancestor is
    # safe only while the child entry is owned by this uid (or root).
    child = parent
    child_metadata = metadata
    while child.parent != child:
        container = child.parent
        try:
            container_metadata = container.stat(follow_symlinks=False)
        except OSError as exc:
            raise RuntimeError(
                f"cannot inspect ovrtx runtime ancestor {container}: {exc}"
            ) from exc
        if not stat.S_ISDIR(container_metadata.st_mode):
            raise RuntimeError(
                f"ovrtx runtime ancestor is not a directory: {container}"
            )
        shared = bool(stat.S_IMODE(container_metadata.st_mode) & 0o022)
        sticky = bool(container_metadata.st_mode & stat.S_ISVTX)
        if shared and not (
            sticky and child_metadata.st_uid in {0, euid}
        ):
            raise RuntimeError(
                f"ovrtx runtime path crosses replaceable shared ancestor "
                f"{container}; choose an owner-controlled directory"
            )
        child = container
        child_metadata = container_metadata
    return identity


def _prepare_runtime_parent(parent: Path) -> tuple[Path, _RuntimePathIdentity]:
    """Create the selected parent, canonicalize symlinks, and bind its identity."""
    try:
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        canonical = parent.resolve(strict=True)
    except OSError as exc:
        raise RuntimeError(
            f"cannot prepare the ovrtx runtime parent at {parent}: {exc}"
        ) from exc
    return canonical, _validate_runtime_parent(canonical)


def _validate_runtime_sibling(
    candidate: Path,
    expected_identity: _RuntimePathIdentity,
    parent_identity: _RuntimePathIdentity,
) -> None:
    """Require the same owner-private staged directory beneath the bound parent."""
    _validate_runtime_parent(candidate.parent, parent_identity)
    try:
        metadata = candidate.stat(follow_symlinks=False)
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect the staged ovrtx runtime at {candidate}: {exc}"
        ) from exc
    get_euid = getattr(os, "geteuid", None)
    wrong_owner = get_euid is not None and metadata.st_uid != get_euid()
    unsafe_mode = get_euid is not None and bool(
        stat.S_IMODE(metadata.st_mode) & 0o077
    )
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or wrong_owner
        or unsafe_mode
        or (metadata.st_dev, metadata.st_ino) != expected_identity
    ):
        raise RuntimeError(
            f"refusing to write the staged ovrtx runtime at {candidate}: "
            "its identity, owner, or permissions changed"
        )


def _target_identity(path: Path) -> tuple[int, int] | None:
    """Stable identity for one activation target, or None when absent."""
    try:
        entry = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(
            f"cannot inspect the ovrtx runtime target at {path}: {exc}"
        ) from exc
    return entry.st_dev, entry.st_ino


def _validate_activation_target(
    venv_dir: Path, expected: _ActivationTarget
) -> None:
    """Fail if the target changed while its replacement was staged."""
    expected_identity, expected_empty, expected_owned = expected
    current_identity = _target_identity(venv_dir)
    if current_identity != expected_identity:
        raise RuntimeError(
            f"refusing to activate the staged ovrtx runtime at {venv_dir}: "
            "the target changed while its replacement was being built. "
            "The current target and staged runtime have been left untouched."
        )
    if venv_dir.is_symlink():
        raise RuntimeError(
            f"refusing to replace the ovrtx runtime at {venv_dir}: the target "
            "is a symlink. Point WU_OVRTX_VENV_DIR at the real runtime "
            "directory so an atomic replacement cannot discard the storage "
            "indirection."
        )
    wu_managed = any(
        (venv_dir / name).exists()
        for name in (_WU_MANAGED_MARKER_NAME, _WU_PROVISIONING_MARKER_NAME)
    )
    if wu_managed:
        raise RuntimeError(
            f"refusing to activate the staged ovrtx runtime at {venv_dir}: "
            "the target became world-understanding managed while its "
            "replacement was being built."
        )
    ours = any(
        (venv_dir / name).exists()
        for name in (_READY_MARKER_NAME, _PROVISIONING_MARKER_NAME)
    )
    if expected_owned and not ours:
        raise RuntimeError(
            f"refusing to activate the staged ovrtx runtime at {venv_dir}: "
            "the usd-cli ownership marker changed while its replacement was "
            "being built."
        )
    if not expected_owned and current_identity is not None:
        if not expected_empty or _has_entries(venv_dir):
            raise RuntimeError(
                f"refusing to activate the staged ovrtx runtime at {venv_dir}: "
                "the previously empty target was populated while its "
                "replacement was being built."
            )


def _capture_activation_target(
    venv_dir: Path, *, owned: bool
) -> _ActivationTarget:
    """Capture and validate the target state before a long staging install."""
    identity = _target_identity(venv_dir)
    expected = (
        identity,
        identity is not None and not _has_entries(venv_dir),
        owned,
    )
    _validate_activation_target(venv_dir, expected)
    return expected


def _lock_provides_pin(runtime_lock: Path) -> str | None:
    """Why this lock cannot produce the qualified worker profile, or ``None``.

    WU_OVRTX_RUNTIME_LOCK is an operator override, so the selected lock is
    not necessarily the one this usd-cli ships. Checking only
    requires-python let a stale or partial lock delete the runtime, install
    whatever subset it names, and get stamped current -- the exact staleness
    the readiness marker exists to prevent, reintroduced one directory up.
    """
    try:
        data = tomllib.loads(runtime_lock.read_text(encoding="utf-8"))
    except OSError as error:
        return f"{runtime_lock} could not be read ({error})"
    except tomllib.TOMLDecodeError as error:
        return f"{runtime_lock.name} is not valid TOML ({error})"
    packages = data.get("packages")
    if not isinstance(packages, list):
        return f"{runtime_lock.name} lists no packages"
    entries = {
        entry.get("name"): entry
        for entry in packages
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    }
    for package_name, want in _QUALIFIED_RUNTIME_PROFILE.items():
        entry = entries.get(package_name)
        if entry is None:
            return f"{runtime_lock.name} contains no {package_name} package"
        found = entry.get("version")
        if found != want:
            return (
                f"{runtime_lock.name} pins {package_name} {found!r}, but this "
                f"usd-cli is built for {want!r}"
            )
        wheels = entry.get("wheels")
        # A locally mirrored wheel is referenced by `path` rather than `url`,
        # but the filename still names the platform, so it is still checkable.
        # Reading a missing url as an empty string refused every install;
        # skipping path entries let a wrong-architecture mirror reach the
        # rmtree. Take whichever is present.
        references = [
            reference
            for w in (wheels if isinstance(wheels, list) else [])
            if isinstance(w, dict)
            for reference in (w.get("url") or w.get("path"),)
            if reference
        ]
        if references and not any(_wheel_runs_here(r) for r in references):
            return (
                f"{runtime_lock.name} carries no {package_name} wheel for this "
                f"machine ({platform.machine()})"
            )
    return None


def _interpreter_satisfies_lock(runtime_lock: Path) -> str | None:
    """Why this interpreter cannot install the lock, or None if it can.

    The replacement is built and verified as a sibling before activation, but
    an unsupported lock can never produce that sibling. Checking up front
    avoids a multi-gigabyte doomed install and preserves the active runtime.
    It gates a fresh provision as well: uv rejects a lock whose requires-python
    this interpreter does not satisfy. It fails
    closed: a lock we cannot read or a requires-python we cannot evaluate is
    reported as unsatisfiable rather than assumed fine. WU_OVRTX_RUNTIME_LOCK
    is an operator override, so unfamiliar specs genuinely arrive here -- the
    world_understanding lock, the natural override on non-x86_64 hosts,
    declares ">=3.12,<3.13".

    A lock stating no requires-python states no constraint, so it allows.
    """
    try:
        data = tomllib.loads(runtime_lock.read_text(encoding="utf-8"))
    except OSError as error:
        return f"{runtime_lock} could not be read ({error})"
    except tomllib.TOMLDecodeError as error:
        return f"{runtime_lock.name} is not valid TOML ({error})"
    spec = data.get("requires-python")
    if not isinstance(spec, str) or not spec.strip():
        return None
    # Compare full versions: a two-component bound normalises to X.Y.0, so
    # 3.12.11 satisfies >=3.12 but not ==3.12 or <=3.12.
    here = sys.version_info[:3]
    for clause in spec.split(","):
        clause = clause.strip()
        # Accept the shapes an operator lock realistically carries: X, X.Y,
        # X.Y.Z, the ==/!= prefix form X.Y.*, and ~=. Anything else -- an
        # epoch, a pre-release bound, an arbitrary-equality clause -- still
        # fails closed, but a spec as ordinary as >=3.12.0 no longer does.
        parsed = re.fullmatch(
            r"(~=|>=|<=|==|!=|>|<)\s*(\d+(?:\.\d+)*)(\.\*)?", clause
        )
        if not parsed:
            return (
                f"{runtime_lock.name} declares requires-python {spec!r}; "
                f"the clause {clause!r} cannot be evaluated"
            )
        op, release, star = parsed.group(1), parsed.group(2), parsed.group(3)
        parts = tuple(int(part) for part in release.split("."))
        if star:
            if op not in {"==", "!="}:
                # PEP 440 allows .* only after == and !=
                return (
                    f"{runtime_lock.name} declares requires-python {spec!r}; "
                    f"the clause {clause!r} cannot be evaluated"
                )
            # ==X.Y.* / !=X.Y.* match on the series, ignoring what follows
            same_series = here[: len(parts)] == parts
            ok = same_series if op == "==" else not same_series
        elif op == "~=":
            if len(parts) < 2:
                # ~=X is not a valid PEP 440 clause
                return (
                    f"{runtime_lock.name} declares requires-python {spec!r}; "
                    f"the clause {clause!r} cannot be evaluated"
                )
            # ~=X.Y is >=X.Y with the last component free to move
            floor = parts + (0,) * (3 - len(parts))
            series = parts[:-1]
            ok = here >= floor and here[: len(series)] == series
        else:
            bound = parts + (0,) * (3 - len(parts))
            ok = {
                ">=": here >= bound,
                "<=": here <= bound,
                "==": here == bound,
                "!=": here != bound,
                ">": here > bound,
                "<": here < bound,
            }[op]
        if not ok:
            return (
                f"{runtime_lock.name} requires Python {spec}, but this "
                f"interpreter is {here[0]}.{here[1]}.{here[2]}"
            )
    return None


def _activate_staged_venv(
    staged_venv: Path,
    venv_dir: Path,
    expected_target: _ActivationTarget,
    *,
    parent_identity: _RuntimePathIdentity,
    staged_identity: _RuntimePathIdentity,
) -> None:
    """Activate a verified sibling while preserving the previous runtime.

    ``os.replace`` cannot replace a non-empty directory directly, so move the
    old owned tree to a unique sibling first, then replace the now-vacant target
    with the staged tree. If the second rename fails, restore the old tree
    before returning the error. The caller holds the external provisioning
    lock throughout this operation.
    """
    backup: Path | None = None
    _validate_runtime_parent(venv_dir.parent, parent_identity)
    _validate_runtime_sibling(
        staged_venv,
        staged_identity,
        parent_identity,
    )
    _validate_activation_target(venv_dir, expected_target)
    if venv_dir.exists():
        backup = venv_dir.parent / (
            f".{venv_dir.name}.previous-{uuid.uuid4().hex}"
        )
        _claim_runtime_sibling(backup)
        try:
            _validate_runtime_parent(venv_dir.parent, parent_identity)
            os.replace(venv_dir, backup)
            _validate_runtime_parent(venv_dir.parent, parent_identity)
            _validate_activation_target(backup, expected_target)
        except BaseException as target_error:
            if backup.exists() and not venv_dir.exists():
                try:
                    _validate_runtime_parent(venv_dir.parent, parent_identity)
                    os.replace(backup, venv_dir)
                except OSError as rollback_error:
                    raise RuntimeError(
                        f"the ovrtx runtime target changed during activation "
                        f"and could not be restored at {venv_dir}; its "
                        f"recoverable contents remain at {backup}: "
                        f"{rollback_error}"
                    ) from target_error
            _release_runtime_sibling(backup)
            raise
    try:
        _validate_runtime_sibling(
            staged_venv,
            staged_identity,
            parent_identity,
        )
        os.replace(staged_venv, venv_dir)
        _validate_runtime_parent(venv_dir.parent, parent_identity)
        _release_runtime_sibling(staged_venv)
    except BaseException as activation_error:
        if backup is not None:
            try:
                _validate_runtime_parent(venv_dir.parent, parent_identity)
                os.replace(backup, venv_dir)
                _release_runtime_sibling(backup)
            except OSError as rollback_error:
                raise RuntimeError(
                    f"could not activate the new ovrtx runtime at {venv_dir} "
                    f"and could not restore the previous runtime; it remains "
                    f"recoverable at {backup}: {rollback_error}"
                ) from activation_error
        raise
    if backup is not None:
        _validate_runtime_parent(venv_dir.parent, parent_identity)
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            # The new runtime is already active and verified. A cleanup error
            # must not roll it back, but leave an exact recoverable path.
            logger.warning(
                "ovrtx auto-install: new runtime is active at %s, but the "
                "previous runtime could not be removed from %s: %s",
                venv_dir,
                backup,
                exc,
            )
        else:
            _release_runtime_sibling(backup)


def _sibling_owner_marker(candidate: Path) -> Path:
    return candidate.with_name(candidate.name + _SIBLING_OWNER_SUFFIX)


def _claim_runtime_sibling(candidate: Path) -> None:
    marker = _sibling_owner_marker(candidate)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(marker, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RuntimeError(
                f"ovrtx runtime ownership sidecar is not a regular file: {marker}"
            )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            descriptor = -1
            stream.write(_SIBLING_OWNER_BODY)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _release_runtime_sibling(candidate: Path) -> None:
    marker = _sibling_owner_marker(candidate)
    try:
        marker.unlink(missing_ok=True)
    except OSError as exc:
        # The payload is already activated, restored, or removed. Leaving a
        # small sidecar is safe; recovery removes it when the sibling is gone.
        logger.warning(
            "ovrtx auto-install: could not remove runtime ownership sidecar "
            "at %s: %s",
            marker,
            exc,
        )


def _runtime_sibling_is_claimed(candidate: Path) -> bool:
    marker = _sibling_owner_marker(candidate)
    try:
        return (
            not marker.is_symlink()
            and marker.is_file()
            and marker.read_text(encoding="utf-8") == _SIBLING_OWNER_BODY
        )
    except OSError:
        return False


def _recover_interrupted_venv_swap(venv_dir: Path) -> None:
    """Recover or clean run-owned siblings left by a hard process exit.

    A normal exception rolls the old runtime back immediately. A power loss
    between the two directory renames can instead leave the target absent and
    the old tree at its unique ``.previous-*`` path. Restore that sole owned
    backup before attempting another install. Verified or partial staging
    siblings are never activated after a restart; they are removed so the
    current lock and import probes are evaluated again.
    """
    parent = venv_dir.parent
    escaped_name = glob.escape(venv_dir.name)
    owned_markers = (_READY_MARKER_NAME, _PROVISIONING_MARKER_NAME)

    def has_in_tree_claim(candidate: Path) -> bool:
        return any((candidate / name).exists() for name in owned_markers)

    def run_owned(candidate: Path) -> bool:
        return (
            not candidate.is_symlink()
            and candidate.is_dir()
            and (
                has_in_tree_claim(candidate)
                or _runtime_sibling_is_claimed(candidate)
            )
        )

    backups = [
        candidate
        for candidate in parent.glob(f".{escaped_name}.previous-*")
        if run_owned(candidate)
    ]
    if not venv_dir.exists() and len(backups) > 1:
        names = ", ".join(str(path) for path in sorted(backups))
        raise RuntimeError(
            f"cannot recover {venv_dir}: multiple usd-cli runtime backups "
            f"exist ({names})"
        )
    if not venv_dir.exists() and backups:
        logger.warning(
            "ovrtx auto-install: restoring the previous runtime at %s after "
            "an interrupted activation",
            venv_dir,
        )
        backup = backups.pop()
        if not has_in_tree_claim(backup):
            raise RuntimeError(
                f"cannot safely restore the previous runtime at {backup}: "
                "its external ownership sidecar remains, but cleanup already "
                "removed its in-tree readiness markers, so the payload may be "
                "partial. It has been retained for operator inspection."
            )
        os.replace(backup, venv_dir)
        _release_runtime_sibling(backup)

    # If the target exists, any remaining owned backup is from a successful
    # activation whose best-effort cleanup was interrupted or failed earlier.
    # Existence alone is not proof of success: retain the only working backup
    # if the target was recreated as an empty or partial tree after a crash.
    target_verified = venv_dir.exists() and _venv_matches_pin(venv_dir)
    for backup in backups:
        if not target_verified:
            logger.warning(
                "ovrtx auto-install: retaining the previous runtime at %s "
                "because the active target at %s is not verified ready",
                backup,
                venv_dir,
            )
            continue
        try:
            shutil.rmtree(backup)
        except OSError as exc:
            logger.warning(
                "ovrtx auto-install: could not remove an old recoverable "
                "runtime at %s: %s",
                backup,
                exc,
            )
        else:
            _release_runtime_sibling(backup)

    for staged in parent.glob(f".{escaped_name}.staging-*"):
        if not run_owned(staged):
            continue
        try:
            shutil.rmtree(staged)
        except OSError as exc:
            logger.warning(
                "ovrtx auto-install: could not remove an interrupted staging "
                "runtime at %s: %s",
                staged,
                exc,
            )
        else:
            _release_runtime_sibling(staged)

    # A hard exit can land after a sibling was moved or removed but before its
    # small external claim was released. It names no existing payload, so it
    # is safe to discard without making any ownership decision about a tree.
    for pattern in (
        f".{escaped_name}.previous-*{_SIBLING_OWNER_SUFFIX}",
        f".{escaped_name}.staging-*{_SIBLING_OWNER_SUFFIX}",
    ):
        for marker in parent.glob(pattern):
            sibling_name = marker.name.removesuffix(_SIBLING_OWNER_SUFFIX)
            if not marker.with_name(sibling_name).exists():
                try:
                    marker.unlink()
                except OSError as exc:
                    logger.warning(
                        "ovrtx auto-install: could not remove orphan runtime "
                        "ownership sidecar at %s: %s",
                        marker,
                        exc,
                    )


def _provision_venv(venv_dir: Path) -> None:
    """The actual install (venv + the hash-pinned runtime lock), file-locked so
    concurrent processes/threads can't corrupt one venv; output streamed."""
    # Resolve the lock and uv before creating anything so a missing pin fails
    # closed without leaving a half-provisioned venv behind.
    venv_dir = Path(venv_dir)
    venv_name = venv_dir.name
    if not venv_name or venv_name in {".", ".."}:
        raise RuntimeError(
            f"WU_OVRTX_VENV_DIR must name one runtime directory, not {venv_dir}"
        )
    runtime_lock = _ovrtx_runtime_lock()
    uv = _uv_command()
    runtime_parent, parent_identity = _prepare_runtime_parent(venv_dir.parent)
    venv_dir = runtime_parent / venv_name
    lock_path = runtime_parent / f"{venv_name}.provision.lock"
    with _file_lock(lock_path):
        _validate_runtime_parent(runtime_parent, parent_identity)
        if venv_dir.is_symlink():
            raise RuntimeError(
                f"refusing to replace the ovrtx runtime at {venv_dir}: the "
                "target is a symlink. Point WU_OVRTX_VENV_DIR at the real "
                "runtime directory."
            )
        if venv_dir.exists() and os.path.ismount(venv_dir):
            raise RuntimeError(
                f"refusing to replace the ovrtx runtime at {venv_dir}: the "
                "target is a mount point and cannot be atomically renamed. "
                "Mount its parent instead and set WU_OVRTX_VENV_DIR to a "
                "child directory inside that mount."
            )
        _validate_runtime_parent(runtime_parent, parent_identity)
        _recover_interrupted_venv_swap(venv_dir)
        _validate_runtime_parent(runtime_parent, parent_identity)
        if _venv_matches_pin(venv_dir):  # another worker finished while we waited
            return
        try:
            runtime_lock_bytes = runtime_lock.read_bytes()
        except OSError as exc:
            raise RuntimeError(
                f"{runtime_lock} could not be read to record its identity in the "
                "readiness marker"
            ) from exc
        runtime_lock_digest = hashlib.sha256(runtime_lock_bytes).hexdigest()
        logger.info(
            "ovrtx auto-install: provisioning isolated venv at %s from the "
            "hash-pinned lock %s (~2.5 GB wheel — a one-time download; "
            "replacing an existing runtime may temporarily need ~5 GB free "
            "in %s while the old and staged environments coexist; progress below)",
            venv_dir,
            runtime_lock,
            venv_dir.parent,
        )
        t0 = time.monotonic()
        # Ownership is decided by markers alone, never by whether bin/python
        # happens to exist. Gating these checks on the interpreter asks 'is
        # this a working venv' when the question is 'whose directory is this',
        # and a tree can carry a marker while missing its interpreter: a
        # partially failed delete, a hand-removed python, or a Windows layout
        # that keeps it in Scripts/. Those all reached an in-place upgrade.
        ready = venv_dir / _READY_MARKER_NAME
        provisioning = venv_dir / _PROVISIONING_MARKER_NAME
        wu_managed_names = (_WU_MANAGED_MARKER_NAME,
                            _WU_PROVISIONING_MARKER_NAME)
        wu_managed = next(
            (venv_dir / n for n in wu_managed_names if (venv_dir / n).exists()),
            venv_dir / _WU_MANAGED_MARKER_NAME)
        ours = ready.exists() or provisioning.exists()

        # Nothing below may replace the environment we are executing from:
        # even a verified sibling cannot be swapped over sys.executable and
        # the dependencies this process is still using.
        if _is_active_environment(venv_dir):
            raise RuntimeError(
                f"{venv_dir} is the environment usd-cli is running from. "
                f"Provisioning replaces a superseded runtime as a whole, "
                f"which would remove this interpreter and its "
                f"dependencies mid-render. Point WU_OVRTX_VENV_DIR at a "
                f"directory of its own.")

        # world_understanding's marker vetoes everything, whatever else is
        # present: its runtime is not ours to replace or install into.
        if wu_managed.exists():
            raise RuntimeError(
                f"{venv_dir} is the world-understanding managed ovrtx runtime "
                f"({wu_managed.name} is present). usd-cli will not "
                "replace or provision into it, even if it also carries our own "
                "marker: the previous provisioner installed into whatever tree "
                "was already there, so that marker does not prove ownership. "
                "The runtime does not match the exact qualified profile "
                f"({OVRTX_PIN}, {OVSTAGE_PIN}, {WARP_PIN}); reprovision it "
                "with world_understanding rather than abandoning the shared "
                "directory:\n  python -m world_understanding.functions.graphics"
                ".render_ovrtx --provision-only --ovrtx-venv-dir "
                f"{shlex.quote(str(venv_dir))}\n"
                "Or point WU_OVRTX_VENV_DIR at a usd-cli-owned directory.")

        # Refuse an unsatisfiable lock before paying for a multi-gigabyte
        # staging install. Both halves gate provisioning: an interpreter that
        # cannot install the lock, and a lock that would not produce the pin
        # we stamp.
        # Both axes, same as the recipe path. `or` reported whichever failed
        # first, so an operator whose lock needs a different interpreter AND
        # names the wrong ovrtx fixed one and met the same refusal with the
        # other reason still hidden.
        blocked_reasons: tuple[str | None, str | None] = (
            _interpreter_satisfies_lock(runtime_lock),
            _lock_provides_pin(runtime_lock),
        )
        blocked = "; ".join(
            reason for reason in blocked_reasons if reason is not None
        )
        if blocked:
            remedy = ("Use an interpreter the lock supports, or regenerate "
                      "the lock for this one.")
            if venv_dir.exists() and _has_entries(venv_dir):
                raise RuntimeError(
                    f"refusing to replace the ovrtx venv at {venv_dir}: "
                    f"{blocked}. It could not be installed here, and replacing "
                    f"an existing runtime would leave none at all. {remedy}")
            raise RuntimeError(
                f"cannot provision an ovrtx venv at {venv_dir}: {blocked}. "
                f"{remedy}")
        if ours:
            # Build a clean sibling rather than upgrade in place: uv removes
            # only what the old distribution's RECORD lists, so files the
            # previous wheel produced after install -- ovrtx keeps an
            # in-package shader cache -- would otherwise survive.
            if ready.exists():
                try:
                    previous = ready.read_text(encoding="utf-8").strip()
                except (OSError, UnicodeError):
                    previous = "an unreadable readiness marker"
            else:
                previous = "an install that did not finish"
            logger.info(
                "ovrtx auto-install: staging a replacement for the usd-cli "
                "venv at %s (%s)",
                venv_dir,
                previous,
            )
        elif venv_dir.exists() and _has_entries(venv_dir):
            # Populated and unmarked: not a runtime we made. Claiming it would
            # arm the next retry to delete its contents -- $HOME from a typo'd
            # WU_OVRTX_VENV_DIR is the case that matters.
            adoption_python = shlex.quote(str(_ovrtx_venv_python_path(venv_dir)))
            adoption_probe = shlex.quote(_RUNTIME_PROBE_SCRIPT)
            adoption_marker = shlex.quote(str(venv_dir / _READY_MARKER_NAME))
            adoption_value = shlex.quote(
                _readiness_marker_body(
                    runtime_lock, runtime_lock_digest=runtime_lock_digest
                ).rstrip("\n")
            )
            raise RuntimeError(
                f"{venv_dir} already exists and is not empty, but carries no "
                f"{_READY_MARKER_NAME} or {_PROVISIONING_MARKER_NAME}, so it is "
                "not a runtime usd-cli created. Refusing to claim it: a later "
                "retry would delete everything in it. Point WU_OVRTX_VENV_DIR "
                "at a new or usd-cli-owned directory. If you built this venv "
                "yourself, check what is actually installed before adopting it "
                "-- install the reviewed lock into it before stamping the "
                "marker, so the digest attests every runtime package rather "
                "than the ovrtx version alone:\n"
                f"  {shlex.join(uv)} pip install --python "
                f"{adoption_python} "
                "--require-hashes --no-deps --no-config --no-sources "
                f"-r {shlex.quote(str(runtime_lock))} \\\n"
                f"    && {adoption_python} -I -c {adoption_probe} \\\n"
                f"    && printf '%s\\n' {adoption_value} > "
                f"{adoption_marker}\n")

        expected_target = _capture_activation_target(venv_dir, owned=ours)

        _validate_runtime_parent(runtime_parent, parent_identity)
        staged_venv = Path(
            tempfile.mkdtemp(
                prefix=f".{venv_name}.staging-",
                dir=str(runtime_parent),
            )
        )
        staged_metadata = staged_venv.stat(follow_symlinks=False)
        staged_identity = (staged_metadata.st_dev, staged_metadata.st_ino)
        _validate_runtime_sibling(
            staged_venv,
            staged_identity,
            parent_identity,
        )
        try:
            _validate_runtime_parent(runtime_parent, parent_identity)
            _claim_runtime_sibling(staged_venv)
        except BaseException:
            _validate_runtime_parent(runtime_parent, parent_identity)
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            staged_venv.rmdir()
            raise
        staged_py = _ovrtx_venv_python_path(staged_venv)
        staged_marker = staged_venv / _READY_MARKER_NAME
        staged_provisioning = staged_venv / _PROVISIONING_MARKER_NAME
        try:
            # Claim the sibling before any fallible install. A normal failure
            # removes it in the finally block; a hard process death leaves an
            # unmistakably usd-cli-owned staging directory, never a damaged
            # active runtime.
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            staged_provisioning.write_text(OVRTX_PIN, encoding="utf-8")
            _run_logged(
                [sys.executable, "-m", "venv", str(staged_venv)],
                what="ovrtx install (venv)",
            )
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            # The pylock carries exact artifact URLs and hashes for the whole
            # runtime (ovrtx, ovstage, Warp, numpy, pillow), so identical runs
            # provision the identical worker and no index resolution can
            # substitute a package. Install from captured bytes, not the live
            # lock path.
            lock_snapshot = staged_venv / "pylock.usd-cli-ovrtx-runtime.toml"
            lock_snapshot.write_bytes(runtime_lock_bytes)
            try:
                _run_logged(
                    [
                        *uv,
                        "pip",
                        "install",
                        "--python",
                        str(staged_py),
                        "--require-hashes",
                        "--no-deps",
                        "--no-config",
                        "--no-sources",
                        "-r",
                        str(lock_snapshot),
                    ],
                    what="ovrtx install (locked runtime)",
                )
                _validate_runtime_sibling(
                    staged_venv,
                    staged_identity,
                    parent_identity,
                )
                if _runtime_lock_digest(lock_snapshot) != runtime_lock_digest:
                    raise RuntimeError(
                        "the private ovrtx runtime lock snapshot changed "
                        "during provisioning; leaving the active venv "
                        "untouched"
                    )
            finally:
                _validate_runtime_sibling(
                    staged_venv,
                    staged_identity,
                    parent_identity,
                )
                lock_snapshot.unlink(missing_ok=True)
            # Record what is installed, not what the lock claimed. The marker
            # is the whole readiness contract, so prove every daemon import in
            # the staged interpreter before it can replace the active tree.
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            installed = _installed_ovrtx_version(staged_py)
            if installed != _pinned_ovrtx_version():
                raise RuntimeError(
                    f"ovrtx staging install for {venv_dir} did not provide "
                    f"the qualified runtime profile "
                    f"{_QUALIFIED_RUNTIME_PROFILE!r} ({runtime_lock} declared "
                    "that profile). Leaving the active runtime untouched."
                )
            current_lock_digest = _runtime_lock_digest(runtime_lock)
            if current_lock_digest != runtime_lock_digest:
                raise RuntimeError(
                    f"{runtime_lock} changed while the ovrtx runtime was "
                    f"being provisioned ({runtime_lock_digest} before "
                    f"install, {current_lock_digest or 'unreadable'} "
                    f"afterward). Leaving the active runtime untouched "
                    f"because the staged companions cannot be bound to the "
                    f"current lock."
                )
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            staged_marker.write_text(
                _readiness_marker_body(
                    runtime_lock, runtime_lock_digest=runtime_lock_digest
                ),
                encoding="utf-8",
            )
            _validate_runtime_sibling(
                staged_venv,
                staged_identity,
                parent_identity,
            )
            staged_provisioning.unlink(missing_ok=True)
            _activate_staged_venv(
                staged_venv,
                venv_dir,
                expected_target,
                parent_identity=parent_identity,
                staged_identity=staged_identity,
            )
            _validate_runtime_parent(runtime_parent, parent_identity)
            _recover_interrupted_venv_swap(venv_dir)
        finally:
            try:
                _validate_runtime_parent(runtime_parent, parent_identity)
            except RuntimeError:
                logger.warning(
                    "ovrtx auto-install: retaining the staged runtime because "
                    "its parent path changed during provisioning",
                    exc_info=True,
                )
            else:
                if staged_venv.exists():
                    try:
                        _validate_runtime_sibling(
                            staged_venv,
                            staged_identity,
                            parent_identity,
                        )
                        shutil.rmtree(staged_venv)
                    except (OSError, RuntimeError) as exc:
                        logger.warning(
                            "ovrtx auto-install: could not remove failed staging "
                            "runtime at %s: %s",
                            staged_venv,
                            exc,
                        )
                    else:
                        _release_runtime_sibling(staged_venv)
                else:
                    _release_runtime_sibling(staged_venv)
        logger.info("ovrtx auto-install: DONE in %.0fs — venv ready at %s",
                    time.monotonic() - t0, venv_dir)


def _run_logged(cmd: list[str], *, what: str, heartbeat_s: float = 30.0) -> None:
    """Run a provisioning command with its output streamed to the logger.

    The ovrtx auto-install used bare subprocess.run: a ~2.5 GB pip download
    produced ZERO output for ~10 minutes (daemon.log and the agent both saw a
    hang). Every line pip prints is logged as it appears, and a heartbeat line
    proves liveness through pip's silent download phase. Failure raises with
    the output tail — not just an exit code."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    tail: list[str] = []
    done = threading.Event()
    t0 = time.monotonic()

    def _beat() -> None:
        while not done.wait(heartbeat_s):
            logger.info("%s: still running (%.0fs elapsed)", what,
                        time.monotonic() - t0)

    beat = threading.Thread(target=_beat, name="ovrtx-install-heartbeat", daemon=True)
    beat.start()
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                logger.info("%s: %s", what, line)
                tail.append(line)
                del tail[:-15]
        code = proc.wait()
    finally:
        done.set()
        beat.join(timeout=1)
    if code != 0:
        raise RuntimeError(f"{what} failed (exit {code}) — last output:\n"
                           + "\n".join(tail[-10:]))


class _file_lock:
    """Cross-process lock for shared runtime provisioning."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: BinaryIO | None = None
        self._lock: Any = None
        self._parent_context: Any = None

    def __enter__(self) -> _file_lock:
        if self._path.is_symlink():
            raise RuntimeError(
                f"ovrtx provisioning lock may not be a symlink: {self._path}"
            )
        descriptor = -1
        try:
            if os.name == "nt":
                parent_context = open_confined_directory(self._path.parent)
                parent_descriptor = parent_context.__enter__()
                self._parent_context = parent_context
                descriptor = open_confined_regular_file_at(
                    parent_descriptor,
                    self._path.name,
                    readable=True,
                    writable=True,
                    create=True,
                )
            else:
                flags = os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                descriptor = os.open(self._path, flags, 0o600)
            opened = os.fstat(descriptor)
            named = self._path.stat(follow_symlinks=False)
            if (
                not stat.S_ISREG(opened.st_mode)
                or opened.st_nlink != 1
                or not stat.S_ISREG(named.st_mode)
                or named.st_nlink != 1
                or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
            ):
                raise RuntimeError(
                    f"ovrtx provisioning lock is not a single-link regular file: "
                    f"{self._path}"
                )
            if os.name == "nt":
                # msvcrt.locking requires the locked byte to exist. The held
                # parent and leaf handles deny delete/rename sharing and reject
                # reparses and hard links; POSIX mode bits are not authoritative.
                if opened.st_size == 0:
                    os.write(descriptor, b"\0")
                    os.fsync(descriptor)
                    os.lseek(descriptor, 0, os.SEEK_SET)
            else:
                os.fchmod(descriptor, 0o600)
            self._fh = os.fdopen(descriptor, "a+b")
            descriptor = -1
            self._lock = advisory_file_lock(self._fh.fileno())
            self._lock.__enter__()
        except BaseException:
            if descriptor >= 0:
                os.close(descriptor)
            if self._fh is not None:
                self._fh.close()
            self._fh = None
            self._lock = None
            if self._parent_context is not None:
                self._parent_context.__exit__(None, None, None)
                self._parent_context = None
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        if self._fh is not None:
            try:
                assert self._lock is not None
                self._lock.__exit__(*exc)
            finally:
                self._fh.close()
                self._fh = None
                self._lock = None
                if self._parent_context is not None:
                    self._parent_context.__exit__(*exc)
                    self._parent_context = None


# Runs *inside* the OVRTX venv (no pxr import here). ovstage owns scene population;
# loads and time updates are published at explicit ordinals before the attached renderer
# consumes them.  Warp kernels require a real source file (Warp intentionally rejects
# decorated kernels defined by ``python -c``), so _OvRTXDaemon writes this reviewed
# constant to a private temporary .py before spawning it.
_DAEMON_SCRIPT = r'''
import importlib.metadata
import ctypes, json, os, signal, sys

def _bind_to_parent_lifetime():
    """Do not outlive the usd-cli component daemon that owns this GPU worker."""
    if not sys.platform.startswith("linux"):
        return
    try:
        expected_parent = int(os.environ.get("USD_CLI_OVRTX_PARENT_PID", "0"))
    except ValueError:
        expected_parent = 0
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux PR_SET_PDEATHSIG. SIGKILL is intentional: the parent may disappear
    # while this process is blocked inside a GPU render and cannot read stdin.
    if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number))
    # Close the fork/exec race: the parent can die before prctl is installed.
    if expected_parent <= 1 or os.getppid() != expected_parent:
        os._exit(1)

_bind_to_parent_lifetime()

# Reserve the inherited stdout pipe exclusively for JSON before importing any
# native runtime. Keep fd 1 redirected to stderr for the daemon's lifetime so
# synchronous and background-thread C writes cannot corrupt the protocol.
_protocol_fd = os.dup(1)
os.dup2(2, 1)
_protocol_stream = os.fdopen(_protocol_fd, "w", encoding="utf-8", buffering=1)

import numpy as np
import ovrtx
import ovstage
import warp as wp
from PIL import Image

WORKER_PROTOCOL_VERSION = 3
MAX_SEMANTIC_IDS = 256
MAX_SENSOR_UPDATES = 1024
FLOAT_SENTINEL_LIMIT = 1.0e30
wp.init()
# Construct only after Warp has initialized, but still before the ready event.
# fd 1 is already redirected, so native constructor banners cannot corrupt the
# JSON-lines protocol.
_renderer = ovrtx.Renderer()

@wp.kernel
def _reduce_f32(values: wp.array(dtype=wp.float32), valid: wp.array(dtype=wp.int32),
                nonzero: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.float64),
                squares: wp.array(dtype=wp.float64), minimum: wp.array(dtype=wp.float64),
                maximum: wp.array(dtype=wp.float64)):
    index = wp.tid()
    value32 = values[index]
    value = wp.float64(value32)
    if wp.isfinite(value32) and wp.abs(value32) < FLOAT_SENTINEL_LIMIT:
        wp.atomic_add(valid, 0, 1)
        if wp.abs(value32) > 1.0e-12:
            wp.atomic_add(nonzero, 0, 1)
        wp.atomic_add(total, 0, value)
        wp.atomic_add(squares, 0, value * value)
        wp.atomic_min(minimum, 0, value)
        wp.atomic_max(maximum, 0, value)

@wp.kernel
def _reduce_f16(values: wp.array(dtype=wp.float16), valid: wp.array(dtype=wp.int32),
                nonzero: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.float64),
                squares: wp.array(dtype=wp.float64), minimum: wp.array(dtype=wp.float64),
                maximum: wp.array(dtype=wp.float64)):
    index = wp.tid()
    value32 = wp.float32(values[index])
    value = wp.float64(value32)
    if wp.isfinite(value32) and wp.abs(value32) < FLOAT_SENTINEL_LIMIT:
        wp.atomic_add(valid, 0, 1)
        if wp.abs(value32) > 1.0e-12:
            wp.atomic_add(nonzero, 0, 1)
        wp.atomic_add(total, 0, value)
        wp.atomic_add(squares, 0, value * value)
        wp.atomic_min(minimum, 0, value)
        wp.atomic_max(maximum, 0, value)

@wp.kernel
def _reduce_u8(values: wp.array(dtype=wp.uint8), valid: wp.array(dtype=wp.int32),
               nonzero: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.float64),
               squares: wp.array(dtype=wp.float64), minimum: wp.array(dtype=wp.float64),
               maximum: wp.array(dtype=wp.float64)):
    index = wp.tid()
    value = wp.float64(values[index])
    wp.atomic_add(valid, 0, 1)
    if values[index] != wp.uint8(0):
        wp.atomic_add(nonzero, 0, 1)
    wp.atomic_add(total, 0, value)
    wp.atomic_add(squares, 0, value * value)
    wp.atomic_min(minimum, 0, value)
    wp.atomic_max(maximum, 0, value)

@wp.kernel
def _reduce_u32(values: wp.array(dtype=wp.uint32), valid: wp.array(dtype=wp.int32),
                nonzero: wp.array(dtype=wp.int32), total: wp.array(dtype=wp.float64),
                squares: wp.array(dtype=wp.float64), minimum: wp.array(dtype=wp.float64),
                maximum: wp.array(dtype=wp.float64)):
    index = wp.tid()
    value = wp.float64(values[index])
    wp.atomic_add(valid, 0, 1)
    if values[index] != wp.uint32(0):
        wp.atomic_add(nonzero, 0, 1)
    wp.atomic_add(total, 0, value)
    wp.atomic_add(squares, 0, value * value)
    wp.atomic_min(minimum, 0, value)
    wp.atomic_max(maximum, 0, value)

@wp.kernel
def _count_semantic_ids(values: wp.array(dtype=wp.uint32),
                        semantic_ids: wp.array(dtype=wp.uint32),
                        counts: wp.array(dtype=wp.int32), semantic_id_count: int):
    value = values[wp.tid()]
    for semantic_index in range(semantic_id_count):
        if value == semantic_ids[semantic_index]:
            wp.atomic_add(counts, semantic_index, 1)
            break

def _emit(obj):
    _protocol_stream.write(json.dumps(obj) + "\n"); _protocol_stream.flush()

def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"

# Compat: ovrtx >=110.1.0.273788 changed map(device=) from str to enum. Map LdrColor on
# the CPU so np.from_dlpack copies to host memory (a GPU buffer would not).
def _cpu_device():
    return ovrtx.Device.CPU if hasattr(ovrtx, "Device") else "cpu"

def _cuda_device():
    return ovrtx.Device.CUDA if hasattr(ovrtx, "Device") else "cuda"

def _seconds_for_time_code(req, time_code):
    # OVRTX expects seconds, while usd-cli's render API accepts USD time codes.
    tcps = float(req.get("time_codes_per_second", 24.0))
    if not np.isfinite(tcps) or tcps <= 0.0:
        tcps = 24.0
    return float(time_code) / tcps

def _sensor_updates(req):
    value = req.get("num_sensor_updates", 64)
    if type(value) is not int or value < 1 or value > MAX_SENSOR_UPDATES:
        raise RuntimeError(
            "num_sensor_updates must be an integer between 1 and %d"
            % MAX_SENSOR_UPDATES
        )
    return value

def _decode_semantic_id_map(tensor):
    data = np.ascontiguousarray(tensor).view(np.uint8).reshape(-1)
    if data.size < 4:
        return {}
    entry_dtype = np.dtype(
        [("id", "<u4", (4,)), ("label_length", "<u4"), ("label_offset", "<u4")]
    )
    num_entries = int.from_bytes(data[-4:].tobytes(), byteorder="little")
    if num_entries < 0 or num_entries > MAX_SEMANTIC_IDS:
        raise RuntimeError("SemanticIdMap contains %d entries; limit is %d"
                           % (num_entries, MAX_SEMANTIC_IDS))
    entries_size = num_entries * entry_dtype.itemsize
    if entries_size > data.size - 4:
        raise RuntimeError("SemanticIdMap entry table is truncated")
    entries = data[:entries_size].view(entry_dtype).reshape(num_entries)
    labels = {}
    for entry in entries:
        semantic_id = int(entry["id"][0])
        label_offset = int(entry["label_offset"])
        label_length = int(entry["label_length"])
        label_end = label_offset + label_length
        if label_offset < entries_size or label_end > data.size - 4:
            raise RuntimeError("SemanticIdMap label range is invalid")
        labels[semantic_id] = data[label_offset:label_end].tobytes().decode(
            "utf-8", errors="strict"
        ).rstrip("\x00").rstrip()
    return labels

def _map_cpu_copy(output):
    mapped = output.map(device=_cpu_device())
    view = None
    try:
        view = np.from_dlpack(mapped)
        return view.copy()
    finally:
        if view is not None:
            del view
        mapped.unmap()

def _semantic_map(frame):
    output = frame.render_vars.get("SemanticIdMap")
    if output is None:
        return {}, None
    values = _map_cpu_copy(output)
    labels = _decode_semantic_id_map(values)
    summary = {
        "shape": [int(value) for value in values.shape],
        "dtype": str(values.dtype),
        "statistics": {
            "element_count": int(values.size),
            "entry_count": len(labels),
            "reduction": "cpu_semantic_metadata_decode_v1",
        },
    }
    return labels, summary

def _gpu_summary(output, stream, semantic_labels=None):
    mapped = output.map(device=_cuda_device(), sync_stream=stream.cuda_stream)
    values = flat = None
    try:
        values = wp.from_dlpack(mapped)
        flat = values.flatten()
        device = flat.device
        with wp.ScopedStream(stream):
            valid = wp.zeros(1, dtype=wp.int32, device=device)
            nonzero = wp.zeros(1, dtype=wp.int32, device=device)
            total = wp.zeros(1, dtype=wp.float64, device=device)
            squares = wp.zeros(1, dtype=wp.float64, device=device)
            minimum = wp.array([np.finfo(np.float64).max], dtype=wp.float64, device=device)
            maximum = wp.array([-np.finfo(np.float64).max], dtype=wp.float64, device=device)
            kernel_by_dtype = {
                wp.float32: _reduce_f32,
                wp.float16: _reduce_f16,
                wp.uint8: _reduce_u8,
                wp.uint32: _reduce_u32,
            }
            kernel = kernel_by_dtype.get(flat.dtype)
            if kernel is None:
                raise RuntimeError("unsupported OVRTX AOV DLPack dtype: %s" % flat.dtype)
            wp.launch(kernel, dim=flat.size,
                      inputs=[flat, valid, nonzero, total, squares, minimum, maximum],
                      stream=stream)
            semantic_counts = None
            ordered_semantic_ids = sorted(semantic_labels or {})
            if ordered_semantic_ids:
                if flat.dtype != wp.uint32:
                    raise RuntimeError("semantic histogram requires a uint32 segmentation AOV")
                semantic_ids = wp.array(
                    np.asarray(ordered_semantic_ids, dtype=np.uint32),
                    dtype=wp.uint32, device=device,
                )
                semantic_counts = wp.zeros(
                    len(ordered_semantic_ids), dtype=wp.int32, device=device
                )
                wp.launch(
                    _count_semantic_ids,
                    dim=flat.size,
                    inputs=[flat, semantic_ids, semantic_counts, len(ordered_semantic_ids)],
                    stream=stream,
                )
        wp.synchronize_stream(stream)
        count = int(flat.size)
        valid_count = int(valid.numpy()[0])
        nonzero_count = int(nonzero.numpy()[0])
        summed = float(total.numpy()[0])
        squared = float(squares.numpy()[0])
        stats = {
            "element_count": count,
            "valid_count": valid_count,
            "invalid_or_sentinel_count": count - valid_count,
            "nonzero_count": nonzero_count,
            "minimum": float(minimum.numpy()[0]) if valid_count else None,
            "maximum": float(maximum.numpy()[0]) if valid_count else None,
            "mean": summed / valid_count if valid_count else None,
            "rms": (squared / valid_count) ** 0.5 if valid_count else None,
            "reduction": "warp_cuda_dlpack_v1",
        }
        if semantic_counts is not None:
            count_values = semantic_counts.numpy().tolist()
            stats["semantic_labels"] = [
                {
                    "id": int(semantic_id),
                    "label": semantic_labels[semantic_id],
                    "pixels": int(pixel_count),
                }
                for semantic_id, pixel_count in zip(
                    ordered_semantic_ids, count_values, strict=True
                )
            ]
            stats["unmapped_pixels"] = count - sum(int(value) for value in count_values)
        return {
            "shape": [int(value) for value in values.shape],
            "dtype": str(flat.dtype).rsplit(".", 1)[-1].rstrip("'>"),
            "statistics": stats,
        }
    finally:
        if flat is not None:
            del flat
        if values is not None:
            del values
        mapped.unmap(stream=stream.cuda_stream)
        # ``unmap(stream=...)`` is asynchronous.  The next AOV mapping may reuse
        # renderer resources, so complete that hand-off before requesting it; this
        # also makes the DLPack lifetime boundary explicit and regression-testable.
        wp.synchronize_stream(stream)

def _save_artifact(output, name, path, semantic_labels):
    if name == "SemanticIdMap":
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(
                [{"id": int(key), "label": semantic_labels[key]}
                 for key in sorted(semantic_labels)],
                stream, sort_keys=True, separators=(",", ":"),
            )
            stream.write("\n")
        return
    values = _map_cpu_copy(output)
    if name in {"LdrColor", "DiffuseAlbedoSD"}:
        Image.fromarray(values).convert("RGB").save(path)
    else:
        np.save(path, values, allow_pickle=False)

def _step_until_ready(renderer, product, ordinal, updates, required_aovs):
    renderer.reset()
    rendered = None
    for _ in range(updates):
        rendered = renderer.step(
            render_products={product}, delta_time=0.0, ordinal=ordinal
        )
    frame = rendered[product].frames[0]
    extra = 0
    missing = [name for name in required_aovs if name not in frame.render_vars]
    while missing and extra < 8:
        rendered = renderer.step(
            render_products={product}, delta_time=0.0, ordinal=ordinal
        )
        frame = rendered[product].frames[0]
        extra += 1
        missing = [name for name in required_aovs if name not in frame.render_vars]
    if missing:
        raise RuntimeError(
            "render produced no %s output after %d samples (vars: %s)"
            % (missing, updates + extra, list(frame.render_vars.keys()))
        )
    return rendered, frame, extra

def _render_request(renderer, req, ordinal):
    manifest = []
    updates = _sensor_updates(req)
    for cam, product, out_path in zip(
        req["cameras"], req["product_paths"], req["out_paths"], strict=True
    ):
        rendered, frame, extra = _step_until_ready(
            renderer, product, ordinal, updates, ["LdrColor"]
        )
        _save_artifact(frame.render_vars["LdrColor"], "LdrColor", out_path, {})
        manifest.append({"camera": cam, "path": out_path, "extra_updates": extra})
        del frame, rendered
    return manifest

def _observe_request(renderer, req, ordinal):
    requested_aovs = list(req["requested_aovs"])
    artifact_paths = list(req.get("artifact_paths") or [{} for _ in req["cameras"]])
    if len(artifact_paths) != len(req["cameras"]):
        raise RuntimeError("artifact-path records and cameras disagree")
    updates = _sensor_updates(req)
    stream = wp.Stream(device=wp.get_device("cuda:0"))
    manifest = []
    for cam, product, paths in zip(
        req["cameras"], req["product_paths"], artifact_paths, strict=True
    ):
        rendered, frame, extra = _step_until_ready(
            renderer, product, ordinal, updates, requested_aovs
        )
        semantic_labels, semantic_map_summary = _semantic_map(frame)
        # Requested host artifacts are copied before CUDA reduction. In the pinned
        # OVRTX release SemanticIdMap is a variable-sized metadata resource; once it
        # has been CUDA-mapped, mapping a texture from the same frame can fail.
        written = {}
        for name, path in sorted(paths.items()):
            if name not in requested_aovs:
                raise RuntimeError("artifact requested for an unrendered AOV: %s" % name)
            _save_artifact(frame.render_vars[name], name, path, semantic_labels)
            written[name] = path
        summaries = {}
        for name in requested_aovs:
            if name == "SemanticIdMap":
                # This variable-sized table is compact metadata and OVRTX exposes it
                # as a host buffer even for a CUDA map request. Decode it once on CPU;
                # the full-resolution SemanticSegmentation image is still mapped and
                # counted through Warp/DLPack on CUDA.
                if semantic_map_summary is None:
                    raise RuntimeError("SemanticIdMap output disappeared before decode")
                summaries[name] = semantic_map_summary
            else:
                summaries[name] = _gpu_summary(
                    frame.render_vars[name],
                    stream,
                    semantic_labels if name == "SemanticSegmentation" else None,
                )
        manifest.append({
            "camera": cam,
            "render_product": product,
            "aovs": summaries,
            "artifacts": written,
            "extra_updates": extra,
        })
        del frame, rendered
    return manifest

def main():
    renderer = _renderer
    stage = ovstage.Stage("usd-cli.ovrtx-render-daemon")
    renderer.attach_ovstage(stage)
    ordinal = 0
    # Extra ready fields are backward-compatible with the original JSON-lines
    # protocol.  Evidence consumers use the ACTUAL isolated-worker distributions,
    # never the controller process's environment, for qualification.
    _emit({
        "status": "ready",
        "worker_protocol_version": WORKER_PROTOCOL_VERSION,
        "capabilities": [
            "rgb_render_v1",
            "ovstage_attached_v1",
            "camera_observation_v1",
            "warp_dlpack_reduction_v1",
            "semantic_overlay_v1",
        ],
        "stage_transport": "ovstage_attached_ordinals",
        "runtime_versions": {
            "ovrtx": _distribution_version("ovrtx"),
            "ovstage": _distribution_version("ovstage"),
            "warp": _distribution_version("warp-lang"),
        },
    })
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            req = json.loads(line)
            cmd = req.get("command")
            if cmd == "shutdown":
                break
            if cmd not in {"render", "observe"}:
                _emit({"status": "error", "error": "unknown command"}); continue
            try:
                ordinal += 1
                ovstage.population.open_usd(stage, req["usd_path"], ordinal=ordinal)
                stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                # A JSON null means USD Default time.  In that case open_usd's
                # default-valued population is already the exact world sampled by
                # SceneAnalysisIR and camera-definition evidence, so do not silently
                # replace it with numeric time code 0.  Numeric frames remain explicit.
                requested_time = req.get("usd_time")
                if requested_time is not None:
                    ordinal += 1
                    ovstage.population.update_from_usd_time_async(
                        stage,
                        ordinal=ordinal,
                        time_code=_seconds_for_time_code(req, requested_time),
                    ).wait()
                    stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
                if cmd == "render":
                    manifest = _render_request(renderer, req, ordinal)
                else:
                    manifest = _observe_request(renderer, req, ordinal)
                _emit({"status": "ok", "manifest": manifest})
            except Exception as exc:  # surface to the client, keep the daemon alive
                _emit({"status": "error", "error": repr(exc)})
    finally:
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()

main()
'''


# One live ovrtx child per (venv, log level) per PROCESS. The usd-cli daemon
# builds a NEW render backend for every command (deliberate — config hot-reload),
# which meant every local ovrtx render spawned a fresh child and paid full RTX
# init + shader compilation again (field report, 2026-07-15). Same medicine as
# the remote pool's _pool_state_for: constructions share module-level state.
_DAEMON_REGISTRY_LOCK = threading.Lock()
_DAEMON_REGISTRY: dict[tuple, "_OvRTXDaemon"] = {}


def _shared_daemon(venv_dir: Path, log_level: str, auto_install: bool,
                   block_install: bool) -> "_OvRTXDaemon":
    key = (str(Path(venv_dir).resolve()), log_level)
    with _DAEMON_REGISTRY_LOCK:
        d = _DAEMON_REGISTRY.get(key)
        if d is not None and d.alive():
            return d
        d = _OvRTXDaemon(venv_dir, log_level, auto_install, block_install)
        _DAEMON_REGISTRY[key] = d
        return d


def _evict_shared(daemon: "_OvRTXDaemon") -> None:
    with _DAEMON_REGISTRY_LOCK:
        for k, v in list(_DAEMON_REGISTRY.items()):
            if v is daemon:
                del _DAEMON_REGISTRY[k]


def close_shared_daemons() -> None:
    """Stop every process-local OVRTX worker during project-daemon shutdown."""
    with _DAEMON_REGISTRY_LOCK:
        daemons = list(dict.fromkeys(_DAEMON_REGISTRY.values()))
        _DAEMON_REGISTRY.clear()
    for daemon in daemons:
        daemon.close()


class _OvRTXDaemon:
    def __init__(self, venv_dir: Path, log_level: str = "warn",
                 auto_install: bool = False, block_install: bool = True):
        self._py = _ovrtx_python(venv_dir, auto_install, block=block_install)
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)  # keep the app's pxr bindings out of the ovrtx venv
        env.setdefault("DISPLAY", ":0")
        env["OVRTX_LOG_LEVEL"] = log_level
        env["USD_CLI_OVRTX_PARENT_PID"] = str(os.getpid())
        # stderr → a temp file, NOT a PIPE: OVRTX/Vulkan are very chatty and a full pipe
        # buffer would deadlock the daemon mid-render. We read the file on death for
        # diagnostics. (Drained implicitly by the OS; no blocking.)
        self._errfile = tempfile.NamedTemporaryFile(prefix="ovrtx_err_", suffix=".log",
                                                    delete=False)
        self._script_file = tempfile.NamedTemporaryFile(
            prefix="usd_cli_ovrtx_worker_", suffix=".py", mode="w",
            encoding="utf-8", delete=False,
        )
        self._script_file.write(_DAEMON_SCRIPT)
        self._script_file.flush()
        self._script_file.close()
        # -I matches the readiness probes' isolation: they vouch for an
        # isolated import environment, so the daemon must run in one too —
        # otherwise a CWD/user-site shadow invisible to the probes would
        # still be first on the daemon's sys.path. -I only isolates the
        # interpreter; OVRTX_LOG_LEVEL, USD_CLI_OVRTX_PARENT_PID, DISPLAY,
        # and LD_LIBRARY_PATH still reach the daemon through env.
        self._proc = subprocess.Popen(
            [self._py, "-I", self._script_file.name],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._errfile,
            text=True, env=env,
        )
        self._lock = threading.Lock()  # serialize the single-pipe protocol across callers
        # Pump stdout lines onto a queue so reads can enforce a real timeout (a hung GPU
        # otherwise blocks forever).
        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()
        try:
            ready = self._read_status(_START_TIMEOUT_S)  # skip non-JSON banners
            if ready.get("status") != "ready":
                raise RuntimeError(f"ovrtx daemon failed to start: {ready}")
            reported_versions = ready.get("runtime_versions")
            if not isinstance(reported_versions, dict) or any(
                reported_versions.get(package) != version
                for package, version in _QUALIFIED_WORKER_RUNTIME_VERSIONS.items()
            ):
                raise RuntimeError(
                    "ovrtx daemon runtime profile mismatch: expected "
                    f"{_QUALIFIED_WORKER_RUNTIME_VERSIONS!r}, reported "
                    f"{reported_versions!r}"
                )
        except Exception:
            self._kill()
            raise
        self._runtime_identity = {
            key: value for key, value in ready.items() if key != "status"
        }

    def alive(self) -> bool:
        return self._proc.poll() is None

    def runtime_identity(self) -> dict:
        """Actual distributions and capabilities reported by the isolated worker."""

        return dict(self._runtime_identity)

    def _pump(self) -> None:
        for line in self._proc.stdout:
            self._lines.put(line)
        self._lines.put(None)  # EOF sentinel

    def _stderr_tail(self, limit: int = 2000) -> str:
        try:
            return Path(self._errfile.name).read_text()[-limit:]
        except Exception:  # noqa: BLE001
            return ""

    def _read(self, timeout_s: float) -> str:
        try:
            line = self._lines.get(timeout=timeout_s)
        except queue.Empty:
            self._kill()
            raise RuntimeError(f"ovrtx daemon timed out after {timeout_s:.0f}s") from None
        if line is None:
            raise RuntimeError(f"ovrtx daemon died: {self._stderr_tail().strip()}")
        return line

    def _read_status(self, timeout_s: float) -> dict:
        """Read lines until one parses as a JSON protocol object, skipping library banners
        that OVRTX may print to stdout before/around the protocol messages."""
        deadline = time.monotonic() + timeout_s
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._kill()
                raise RuntimeError(f"ovrtx daemon timed out after {timeout_s:.0f}s")
            line = self._read(remaining).strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue  # a stray init/banner line on stdout — ignore
            if isinstance(obj, dict) and "status" in obj:
                return obj

    def _request(self, command: str, params: dict) -> list[dict]:
        with self._lock:  # one GPU operation at a time over the shared pipe
            if self._proc.poll() is not None:
                raise RuntimeError(f"ovrtx daemon is not running: {self._stderr_tail().strip()}")
            try:
                self._proc.stdin.write(json.dumps({"command": command, **params}) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                self._kill()
                raise RuntimeError(f"ovrtx daemon stdin closed: {exc}") from exc
            resp = self._read_status(_RENDER_TIMEOUT_S)
            if resp.get("status") == "error":
                raise RuntimeError(f"ovrtx {command} error: {resp.get('error')}")
            manifest = resp.get("manifest")
            if not isinstance(manifest, list):
                raise RuntimeError(f"ovrtx {command} returned a malformed manifest")
            return manifest

    def render(self, params: dict) -> list[dict]:
        return self._request("render", params)

    def observe(self, params: dict) -> list[dict]:
        return self._request("observe", params)

    def _kill(self) -> None:
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        try:
            Path(self._script_file.name).unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass

    def close(self) -> None:
        _evict_shared(self)  # a closed daemon must not be handed out again
        try:
            self._proc.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
            self._proc.stdin.flush()
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self._kill()
        finally:
            try:
                Path(self._script_file.name).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                pass


class OvRTXRenderBackend:
    name = "ovrtx"

    def __init__(self, venv_dir: Path | None = None, num_sensor_updates: int = 64,
                 render_mode: str = "", log_level: str = "warn",
                 auto_install: bool = False):
        self._venv_dir = Path(venv_dir) if venv_dir else DEFAULT_VENV
        self._num_sensor_updates = _validated_sensor_updates(num_sensor_updates)
        # "" => derive from the per-call fast/quality mode; else pin an explicit rt1/rt2/pt.
        self._render_mode = render_mode
        self._log_level = log_level
        # Whether a missing ovrtx venv may be pip-provisioned (~2.5 GB download) at
        # render time; render.ovrtx_auto_install, default false (see _ovrtx_python).
        self._auto_install = auto_install
        self._daemon: _OvRTXDaemon | None = None

    def _ensure_daemon(self, allow_install: bool | None = None,
                       block_install: bool = False) -> _OvRTXDaemon:
        # block_install=False on the render path: a synchronous 2.5 GB pip
        # download inside a command handler wedged the per-project daemon for
        # ~10 minutes; render-time provisioning now runs in the background and
        # the command fails fast with progress (see _ovrtx_python).
        if self._daemon is None or not self._daemon.alive():
            allow = self._auto_install if allow_install is None else allow_install
            # shared across backend constructions: shader compilation happens
            # once per project daemon, not once per render
            self._daemon = _shared_daemon(self._venv_dir, self._log_level,
                                          auto_install=allow,
                                          block_install=block_install)
        return self._daemon

    def runtime_identity(self) -> dict:
        """Return evidence-safe identity from the executing isolated process."""

        daemon = self._ensure_daemon()
        reporter = getattr(daemon, "runtime_identity", None)
        if not callable(reporter):
            raise RuntimeError("ovrtx worker does not report runtime identity")
        identity = reporter()
        if not isinstance(identity, dict):
            raise RuntimeError("ovrtx worker returned a malformed runtime identity")
        return dict(identity)

    def ensure_ready(self) -> None:
        """Boot the OVRTX daemon now (provisions the venv on first call). Used by the
        rendering service to warm the GPU off the request path — an explicit warm-up is
        the right moment for the one-time wheel download, so installation is always
        allowed here AND blocks until done (unlike at render time)."""
        self._ensure_daemon(allow_install=True, block_install=True)

    def restart_daemon(self) -> None:
        """Tear down the ovrtx daemon so the next render boots a fresh one."""
        daemon = self._daemon
        self._daemon = None
        if daemon is None:
            return
        _evict_shared(daemon)
        close = getattr(daemon, "close", None)
        if not callable(close):
            # Keep registry eviction usable with lightweight test/service doubles
            # that implement only the shared-daemon ``alive`` contract.
            kill = getattr(daemon, "_kill", None)
            if callable(kill):
                kill()
            return
        try:
            close()
        except Exception:  # noqa: BLE001 — last resort
            kill = getattr(daemon, "_kill", None)
            if callable(kill):
                kill()

    @property
    def daemon_running(self) -> bool:
        return self._daemon is not None and self._daemon._proc.poll() is None

    def _resolve_render_mode(self, mode: str) -> str:
        if self._render_mode:
            return self._render_mode
        return _MODE_TO_RENDER_MODE.get(mode, "rt1")

    def _resolve_updates(self, mode: str) -> int:
        """Accumulation passes for this mode. OVRTX time scales ~linearly with sample
        count, and `fast` previews don't need the full quality budget — using the whole
        64-sample count for every render made large-scene previews minutes-long. `fast`
        caps at a small budget; `quality` gets the configured max. Override the fast cap
        with OVRTX_FAST_SENSOR_UPDATES."""
        if mode == "quality":
            return self._num_sensor_updates
        import os
        fast = int(os.environ.get("OVRTX_FAST_SENSOR_UPDATES", "8"))
        return max(1, min(self._num_sensor_updates, fast))

    def _run_worker(self, command: str, params: dict) -> list[dict]:
        """Run one daemon command, preserving the existing one-retry policy."""

        if command not in {"render", "observe"}:
            raise ValueError(f"unknown OVRTX worker command: {command}")

        def invoke():
            daemon = self._ensure_daemon()
            operation = getattr(daemon, command, None)
            if not callable(operation):
                raise RuntimeError(
                    f"ovrtx worker does not support the {command} command"
                )
            return operation(params)

        try:
            return invoke()
        except RuntimeError as exc:
            msg = str(exc)
            # A dead subprocess needs replacement.  An application-level render
            # failure must stay on the live daemon: tearing Vulkan down after such
            # a failure can leave the device unavailable (NVML 999).
            daemon_dead = (
                "daemon died" in msg
                or "daemon is not running" in msg
                or "stdin closed" in msg
            )
            if daemon_dead:
                logger.warning(
                    "ovrtx daemon died (%s) — restarting and retrying once", exc
                )
                self.restart_daemon()
            else:
                logger.warning(
                    "ovrtx %s error (%s) — retrying once on the live daemon",
                    command,
                    exc,
                )
            return invoke()

    def render(self, stage, cameras, width, height, out_dir, mode: str = "quality",
               names: list[str] | None = None, frame: float | None = None) -> list[RenderResult]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        cameras = list(cameras)
        if not cameras:
            raise ValueError("OVRTX render requires at least one camera")
        if len(cameras) > 64:
            raise ValueError("OVRTX render accepts at most 64 cameras")
        if width < 1 or height < 1 or width * height > 67_108_864:
            raise ValueError("OVRTX render resolution is invalid or exceeds 64 megapixels")
        stems = _safe_output_stems(cameras, names)
        render_mode = self._resolve_render_mode(mode)
        num_sensor_updates = self._resolve_updates(mode)

        # Camera validation must precede stage export: instance-proxy rejection is
        # expected to fail closed without leaking the exported stage or overlays.
        _validate_ovrtx_cameras(stage, cameras)

        # Resolve and validate the complete default light rig before exporting
        # the stage or writing render-product overlays. A bad explicit HDRI must
        # fail without leaving workflow-owned temporary files behind.
        default_lights_usda: str | None = None
        if not _stage_has_lights(stage):
            default_lights_usda = _build_default_lights_usda(
                _default_hdri_intensity()
            )

        # Export the working stage by path (its USD libs open the file), preserving
        # payload/texture resolution (see prepare_render_input).
        usd_path, is_temp = prepare_render_input(stage, out_dir)

        # Author RenderProducts (+ default lights) as sublayer overlays into one combined
        # stage. The daemon opens only `_combined.usda`.
        products_usda, product_paths = _build_render_products_usda(
            cameras, width, height, render_mode
        )
        # Unique per-render token (PID alone collides under in-process concurrency).
        tok = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        products_path = out_dir / f".usd-cli_render_products_{tok}.usda"
        products_path.write_text(products_usda)
        # Preserve the legacy render command's square-pixel filmback behavior in a
        # disposable stronger layer.  The former implementation authored this opinion
        # directly onto the caller's live stage, making a nominal render mutate history.
        aspect_path = out_dir / f".usd-cli_render_camera_aspect_{tok}.usda"
        _write_camera_aspect_overlay(stage, cameras, width, height, aspect_path)

        sublayers = [str(products_path), str(aspect_path), str(usd_path)]  # earlier = stronger
        lights_path: Path | None = None
        if default_lights_usda is not None:
            lights_path = out_dir / f".usd-cli_render_lights_{tok}.usda"
            lights_path.write_text(default_lights_usda)
            sublayers.append(str(lights_path))

        combined_path = out_dir / f".usd-cli_render_combined_{tok}.usda"
        sublayer_list = ", ".join(f"@{p}@" for p in sublayers)
        # Time metadata must live on the COMBINED root layer: a root without
        # timeCodesPerSecond defaults to 24, and USD then time-SCALES sublayers
        # authored at a different rate (a tcps=30 recording sampled 0..30 maps
        # to 0..24) — animation renders sampled the wrong codes.
        root_metadata = _combined_stage_metadata(stage)
        combined_path.write_text(
            "#usda 1.0\n(\n"
            f"{root_metadata}    subLayers = [{sublayer_list}]\n"
            ")\n"
        )

        out_paths = [str(out_dir / f"{s}.png") for s in stems]
        params = {
            "usd_path": str(combined_path),
            "cameras": cameras,
            "product_paths": product_paths,
            "out_paths": out_paths,
            "num_sensor_updates": num_sensor_updates,
            # Preserve the established plain-render contract: omitting --frame
            # renders time code 0.  Observation/verification deliberately uses
            # USD Default time when its frame is omitted, because its camera and
            # scene digests are bound to Default rather than numeric zero.
            "usd_time": float(frame) if frame is not None else 0.0,
            "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
        }
        t0 = time.perf_counter()
        try:
            manifest = self._run_worker("render", params)
        finally:
            if is_temp:
                usd_path.unlink(missing_ok=True)
            products_path.unlink(missing_ok=True)
            aspect_path.unlink(missing_ok=True)
            combined_path.unlink(missing_ok=True)
            if lights_path is not None:
                lights_path.unlink(missing_ok=True)
        dt = time.perf_counter() - t0
        results = [RenderResult(path=m["path"], camera=m["camera"], width=width, height=height,
                                backend=self.name, render_time=dt,
                                ovrtx_render_mode=render_mode,
                                ovrtx_num_sensor_updates=num_sensor_updates,
                                active_aov="LdrColor") for m in manifest]
        # blank-frame sanity: flag featureless outputs (camera missed its subject) so the
        # caller sees `blank_suspect` + a warning instead of trusting a plausible gradient
        from usd_core.render.remote import _flag_blank_suspects

        _flag_blank_suspects(results)
        return results

    def observe(
        self,
        stage,
        cameras,
        width: int,
        height: int,
        out_dir,
        *,
        mode: str = "quality",
        aovs: tuple[str, ...] | list[str] | None = None,
        artifact_aovs: tuple[str, ...] | list[str] = ("LdrColor",),
        semantic_labels: dict[str, str] | None = None,
        names: list[str] | None = None,
        frame: float | None = None,
    ) -> list[OvRTXObservationResult]:
        """Render independent cameras and return GPU-reduced typed AOV summaries.

        AOV mappings are reduced inside the isolated worker on the exact CUDA stream
        synchronized by OVRTX.  Only ``artifact_aovs`` cross to CPU in full.  The
        optional semantic mapping is authored into a disposable stronger layer and is
        never applied to ``stage``.
        """

        requested_aovs = _validated_observation_aovs(aovs)
        requested_artifacts = tuple(str(name) for name in artifact_aovs)
        if len(set(requested_artifacts)) != len(requested_artifacts):
            raise ValueError("OVRTX observation artifact AOVs must be unique")
        unknown_artifacts = sorted(set(requested_artifacts) - set(requested_aovs))
        if unknown_artifacts:
            raise ValueError(
                "OVRTX observation artifact was not requested as an AOV: "
                f"{unknown_artifacts[0]}"
            )
        semantic_labels = dict(semantic_labels or {})
        if semantic_labels and "SemanticSegmentation" not in requested_aovs:
            raise ValueError(
                "OVRTX semantic labels require SemanticSegmentation and SemanticIdMap"
            )
        cameras = [str(camera) for camera in cameras]
        if not cameras:
            raise ValueError("OVRTX camera observation requires at least one camera")
        if len(cameras) > 64:
            raise ValueError("OVRTX camera observation accepts at most 64 cameras")
        if len(set(cameras)) != len(cameras):
            raise ValueError("OVRTX camera observation cameras must be unique")
        width, height = int(width), int(height)
        if width < 1 or height < 1 or width * height > 67_108_864:
            raise ValueError(
                "OVRTX camera observation resolution is invalid or exceeds 64 megapixels"
            )
        # One product is processed at a time, but all requested buffers coexist for
        # that product. Reject an unsafe lower-bound estimate before OVRTX allocates.
        bytes_per_pixel = sum(
            _IMAGE_AOV_CHANNELS.get(name, 0)
            * (2 if name == "HdrColor" else 4 if name not in {"LdrColor", "DiffuseAlbedoSD"} else 1)
            for name in requested_aovs
        )
        estimated_bytes = width * height * bytes_per_pixel
        if estimated_bytes > 2 * 1024 * 1024 * 1024:
            raise ValueError(
                "OVRTX camera observation AOVs need at least "
                f"{estimated_bytes:,} bytes per camera; limit is 2 GiB"
            )
        stems = _safe_output_stems(
            cameras,
            names
            if names is not None
            else [
                f"camera_{index:03d}"
                for index in range(1, len(cameras) + 1)
            ],
        )

        out_dir = Path(out_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        render_mode = self._resolve_render_mode(mode)
        num_sensor_updates = self._resolve_updates(mode)

        identity = self.runtime_identity()
        capabilities = identity.get("capabilities") or []
        required_capabilities = {
            CAMERA_OBSERVATION_CAPABILITY,
            WARP_DLPACK_REDUCTION_CAPABILITY,
        }
        missing_capabilities = sorted(required_capabilities - set(capabilities))
        if missing_capabilities:
            raise RuntimeError(
                "OVRTX worker does not support typed camera observations: "
                + ", ".join(missing_capabilities)
            )
        if semantic_labels and SEMANTIC_OVERLAY_CAPABILITY not in capabilities:
            raise RuntimeError("OVRTX worker does not support semantic overlays")

        default_lights_usda: str | None = None
        if not _stage_has_lights(stage):
            default_lights_usda = _build_default_lights_usda(
                _default_hdri_intensity()
            )
        usd_path, is_temp = prepare_render_input(stage, out_dir)
        token = f"{os.getpid()}_{uuid.uuid4().hex[:8]}"
        products_path = out_dir / f".usd-cli_observe_products_{token}.usda"
        combined_path = out_dir / f".usd-cli_observe_combined_{token}.usda"
        semantic_path: Path | None = None
        lights_path: Path | None = None
        cleanup_paths = [products_path, combined_path]
        semantic_assignments: list[dict[str, str]] = []
        try:
            products_usda, product_paths = _build_render_products_usda(
                cameras,
                width,
                height,
                render_mode,
                render_vars=requested_aovs,
            )
            products_path.write_text(products_usda)
            sublayers = [str(products_path)]
            if semantic_labels:
                semantic_path = out_dir / f".usd-cli_observe_semantics_{token}.usda"
                semantic_assignments = _write_semantic_overlay(
                    stage, semantic_labels, semantic_path
                )
                cleanup_paths.append(semantic_path)
                sublayers.append(str(semantic_path))
            sublayers.append(str(usd_path))
            if default_lights_usda is not None:
                lights_path = out_dir / f".usd-cli_observe_lights_{token}.usda"
                lights_path.write_text(default_lights_usda)
                cleanup_paths.append(lights_path)
                sublayers.append(str(lights_path))

            root_metadata = _combined_stage_metadata(stage)
            sublayer_list = ", ".join(f"@{path}@" for path in sublayers)
            combined_path.write_text(
                "#usda 1.0\n(\n"
                f"{root_metadata}    subLayers = [{sublayer_list}]\n"
                ")\n"
            )

            artifact_paths: list[dict[str, str]] = []
            for stem in stems:
                artifact_paths.append(
                    {
                        name: str(
                            out_dir
                            / f"{stem}__{name}{_observation_artifact_suffix(name)}"
                        )
                        for name in requested_artifacts
                    }
                )
            params = {
                "usd_path": str(combined_path),
                "cameras": cameras,
                "product_paths": product_paths,
                "requested_aovs": list(requested_aovs),
                "artifact_paths": artifact_paths,
                "num_sensor_updates": num_sensor_updates,
                # None is the explicit USD Default-time protocol value.  Numeric 0
                # is distinct on a stage with authored time samples.
                "usd_time": float(frame) if frame is not None else None,
                "time_codes_per_second": float(stage.GetTimeCodesPerSecond()),
            }
            manifest = self._run_worker("observe", params)
        finally:
            if is_temp:
                usd_path.unlink(missing_ok=True)
            for path in cleanup_paths:
                path.unlink(missing_ok=True)

        if len(manifest) != len(cameras):
            raise RuntimeError(
                f"OVRTX observation returned {len(manifest)} camera(s) for "
                f"{len(cameras)} requests"
            )
        records: dict[str, dict] = {}
        for record in manifest:
            if not isinstance(record, dict):
                raise RuntimeError("OVRTX observation returned a malformed camera record")
            camera = str(record.get("camera", ""))
            if camera not in cameras or camera in records:
                raise RuntimeError(
                    f"OVRTX observation returned unexpected or duplicate camera {camera!r}"
                )
            records[camera] = record

        results: list[OvRTXObservationResult] = []
        for index, camera in enumerate(cameras):
            record = records.get(camera)
            if record is None:
                raise RuntimeError(f"OVRTX observation omitted camera {camera}")
            aov_records = record.get("aovs")
            if not isinstance(aov_records, dict) or set(aov_records) != set(requested_aovs):
                raise RuntimeError(
                    f"OVRTX observation returned an incomplete AOV set for {camera}"
                )
            for name, aov_record in aov_records.items():
                if not isinstance(aov_record, dict):
                    raise RuntimeError(f"OVRTX {name} summary for {camera} is malformed")
                shape = aov_record.get("shape")
                if name != "SemanticIdMap":
                    expected_shape = [height, width, _IMAGE_AOV_CHANNELS[name]]
                    if shape != expected_shape:
                        raise RuntimeError(
                            f"OVRTX {name} shape for {camera} is {shape}, expected "
                            f"{expected_shape}"
                        )
                elif (
                    not isinstance(shape, list)
                    or not shape
                    or any(
                        not isinstance(dimension, int)
                        or isinstance(dimension, bool)
                        or dimension < 1
                        for dimension in shape
                    )
                ):
                    raise RuntimeError(f"OVRTX SemanticIdMap for {camera} has no shape")
                statistics = aov_record.get("statistics")
                expected_reduction = (
                    "cpu_semantic_metadata_decode_v1"
                    if name == "SemanticIdMap"
                    else "warp_cuda_dlpack_v1"
                )
                if (
                    not isinstance(statistics, dict)
                    or statistics.get("reduction") != expected_reduction
                    or not isinstance(statistics.get("element_count"), int)
                    or isinstance(statistics.get("element_count"), bool)
                    or statistics["element_count"] != math.prod(shape)
                ):
                    raise RuntimeError(
                        f"OVRTX {name} summary for {camera} used an unexpected reduction"
                    )
            returned_artifacts = record.get("artifacts")
            expected_artifacts = {
                name: str(
                    out_dir
                    / f"{stems[index]}__{name}{_observation_artifact_suffix(name)}"
                )
                for name in requested_artifacts
            }
            if returned_artifacts != expected_artifacts:
                raise RuntimeError(
                    f"OVRTX observation returned an unexpected artifact set for {camera}"
                )
            for name, artifact_path in expected_artifacts.items():
                path = Path(artifact_path)
                if not path.is_file() or path.stat().st_size < 1:
                    raise RuntimeError(
                        f"OVRTX {name} observation artifact is missing or empty: {path}"
                    )
            # Reject NaN/Infinity before the record can enter canonical evidence.
            try:
                json.dumps(aov_records, allow_nan=False)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"OVRTX observation for {camera} contains a non-JSON statistic"
                ) from exc
            results.append(
                OvRTXObservationResult(
                    camera=camera,
                    render_product=str(record.get("render_product", "")),
                    aovs=aov_records,
                    artifacts=expected_artifacts,
                    ovrtx_render_mode=render_mode,
                    ovrtx_num_sensor_updates=num_sensor_updates,
                    semantic_assignments=tuple(semantic_assignments),
                )
            )
        return results
