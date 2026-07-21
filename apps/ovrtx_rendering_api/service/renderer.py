# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""OVRTX rendering wrapper for the rendering API service.

Fetches USD from URL/data-URI, renders via OvRTXRenderingBackend,
and converts results to the V1 response format expected by the
material-agent-service client.
"""

from __future__ import annotations

import base64
import binascii
import io
import ipaddress
import logging
import os
import re
import socket
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import numpy as np
import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.poolmanager import PoolManager

from world_understanding.utils.archive import (
    ArchiveSizeLimitExceeded,
    copy_stream_limited,
)
from world_understanding.utils.image_blankness import analyze_image_blankness

logger = logging.getLogger(__name__)

# USD layer extensions recognized inside ZIP bundles. Mirrors the client
# bundling in world_understanding.functions.graphics.render_remote, which
# packages a .usda root plus MDL/texture assets when the scene references
# local files.
_USD_EXTENSIONS = (".usd", ".usda", ".usdc")

# Denial-of-service guards for untrusted ZIP bundles. The default keeps
# generic public-facing deployments conservative; internal large-scene
# deployments can raise the byte ceiling with OVRTX_ZIP_MAX_UNCOMPRESSED_BYTES.
_ZIP_MAX_FILES = 10_000
_ZIP_DEFAULT_MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB
_ZIP_MAX_UNCOMPRESSED_BYTES_ENV = "OVRTX_ZIP_MAX_UNCOMPRESSED_BYTES"

# Regex for S3 HTTPS URLs:
#   https://bucket.s3.region.amazonaws.com/key
#   https://s3.region.amazonaws.com/bucket/key
_S3_VHOST_RE = re.compile(
    r"^https?://(?P<bucket>[a-z0-9][a-z0-9.\-]+)\.s3[.\-](?P<region>[a-z0-9-]+)\.amazonaws\.com/(?P<key>.+)$"
)
_S3_PATH_RE = re.compile(
    r"^https?://s3[.\-](?P<region>[a-z0-9-]+)\.amazonaws\.com/(?P<bucket>[^/]+)/(?P<key>.+)$"
)

# Sensor name mapping: Kit API names -> OVRTX names
_SENSOR_KIT_TO_OVRTX: dict[str, str] = {
    "linear_depth": "depth",
    "depth": "depth",
}

# Sensors supported by OVRTX
_SUPPORTED_SENSORS = {"depth"}

_REMOTE_USD_SCHEMES = frozenset({"http", "https"})
_URL_SCHEME_SEPARATOR = "://"
_HTTP_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_MAX_HTTP_REDIRECTS = 5
# Keep service-side blank checks bounded; dataset guardrails can do deeper
# analysis later when deciding whether to fail a pipeline.
_SERVICE_BLANKNESS_MAX_ANALYSIS_PIXELS = 65_536

_RECOVERABLE_RENDER_ERROR_SNIPPETS = (
    "OvRTX daemon pipe failed",
    "OvRTX daemon died during render",
    "OvRTX daemon render timed out",
    "OvRTX daemon startup timed out",
    "OvRTX daemon unexpected response",
)
_RECOVERY_FAILURE_COOLDOWN_SECONDS = 5.0
_AWS_METADATA_IPV4 = str(ipaddress.ip_address(0xA9FEA9FE))
_BLOCKED_URL_HOSTS = frozenset(
    {
        "localhost",
        _AWS_METADATA_IPV4,
        "metadata.google.internal",
    }
)
_IPV4_PART_MAX_VALUES = (0xFF, 0xFF, 0xFF, 0xFF)
_LEGACY_IPV4_PART_MAX_VALUES = {
    1: (0xFFFFFFFF,),
    2: (0xFF, 0xFFFFFF),
    3: (0xFF, 0xFF, 0xFFFF),
    4: _IPV4_PART_MAX_VALUES,
}


def _parse_zip_max_uncompressed_bytes(raw_value: str | None) -> int:
    if raw_value is None or raw_value == "":
        return _ZIP_DEFAULT_MAX_UNCOMPRESSED_BYTES

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{_ZIP_MAX_UNCOMPRESSED_BYTES_ENV} must be an integer number of bytes"
        ) from exc

    if value <= 0:
        raise ValueError(f"{_ZIP_MAX_UNCOMPRESSED_BYTES_ENV} must be greater than zero")
    return value


_ZIP_MAX_UNCOMPRESSED_BYTES = _parse_zip_max_uncompressed_bytes(
    os.environ.get(_ZIP_MAX_UNCOMPRESSED_BYTES_ENV)
)


def _zip_max_uncompressed_bytes() -> int:
    return _ZIP_MAX_UNCOMPRESSED_BYTES


class Renderer:
    """OVRTX-based USD renderer.

    Wraps OvRTXRenderingBackend from world_understanding with methods
    for fetching USD and converting results to the Kit API V1 format.
    """

    def __init__(
        self,
        log_level: str = "warn",
        num_sensor_updates: int = 500,
        render_mode: str = "pt",
    ) -> None:
        from world_understanding.functions.graphics.rendering import (
            OvRTXRenderingBackend,
        )

        self._backend = OvRTXRenderingBackend(
            log_level=log_level,
            num_sensor_updates=num_sensor_updates,
            render_mode=render_mode,
        )
        # Constructing OvRTXRenderingBackend builds the daemon IPC handle but
        # the actual ovrtx subprocess (and the GPU) doesn't start until the
        # first render request — see rendering.py "The daemon starts lazily
        # on first render(); GPU init cost is paid once." Leave `_initialized`
        # False until warm_up() actually exercises the GPU so /health does
        # not lie about readiness.
        self._initialized = False
        # The OVRTX daemon crashes when hit with concurrent renders.
        # Serialize all render calls through a lock so requests queue
        # instead of overlapping.
        self._render_lock = threading.RLock()
        self._recovery_cooldown_until = 0.0
        logger.info(
            "OVRTX renderer constructed (num_sensor_updates=%d, render_mode=%s)",
            num_sensor_updates,
            render_mode,
        )

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def daemon_running(self) -> bool:
        daemon = getattr(self._backend, "_daemon", None)
        if daemon is None:
            return False
        is_running = getattr(daemon, "_is_running", None)
        if not callable(is_running):
            return False
        return bool(is_running())

    @property
    def daemon_lifecycle(self) -> dict[str, int | str | None]:
        """Return bounded-daemon diagnostics without blocking the render lock."""
        daemon = getattr(self._backend, "_daemon", None)
        snapshot = getattr(daemon, "lifecycle_snapshot", None)
        if not callable(snapshot):
            return {}
        payload = snapshot()
        return payload if isinstance(payload, dict) else {}

    @property
    def is_ready(self) -> bool:
        return self._initialized and self.daemon_running

    def warm_up(self) -> bool:
        """Force lazy GPU init by rendering a tiny programmatic scene once.

        Builds an in-memory USD stage (cube + camera + distant light) and
        drives it straight through the backend. No fixture file needed —
        earlier versions shipped ``tests/renders/smoke_cube.usda`` and
        resolved it via ``__file__``, which silently broke on the wheel
        install path because the wheel doesn't ship ``tests/``.

        Returns True on success and flips ``is_initialized``. Returns
        False and leaves the renderer in the not-initialized state on
        failure so /health correctly reports the GPU is not ready.
        """
        with self._render_lock:
            try:
                stage = _build_smoke_stage()
                result = self._backend.render(
                    stage=stage,
                    cameras=["/World/Camera"],
                    image_width=64,
                    image_height=64,
                    frames="0",
                    sensors=None,
                )
            except Exception:
                logger.exception("OVRTX warm-up render failed; GPU not initialized")
                return False
            if not result.get("results"):
                logger.error("OVRTX warm-up render returned no results: %s", result)
                return False
            self._initialized = True
            logger.info("OVRTX renderer warmed up — GPU is ready")
            return True

    def shutdown(self) -> None:
        """Shut down the OVRTX daemon.

        TODO: switch to a public OvRTXRenderingBackend.shutdown() once the
        backend exposes one — we reach into ``_daemon`` directly today.
        """
        try:
            if hasattr(self._backend, "_daemon"):
                self._backend._daemon.shutdown()
        finally:
            self._initialized = False

    def recover(self, *, force: bool = False) -> bool:
        """Restart the OVRTX daemon and re-run warm-up as a single-flight action.

        The daemon can die or wedge underneath an otherwise initialized service.
        Recovery owns the same service-level render lock used for renders so
        concurrent requests do not stampede daemon restart. Non-forced callers
        skip work if another request already recovered the renderer.
        """
        with self._render_lock:
            if not force and self.is_ready:
                logger.info("OVRTX daemon recovery skipped; renderer is already ready")
                return True

            now = time.monotonic()
            cooldown_until = getattr(self, "_recovery_cooldown_until", 0.0)
            if cooldown_until > now:
                logger.warning(
                    "Skipping OVRTX daemon recovery for %.2fs after recent failure",
                    cooldown_until - now,
                )
                return False

            logger.warning("Recovering OVRTX daemon")
            try:
                self.shutdown()
            except Exception:
                logger.exception("OVRTX daemon shutdown failed during recovery")
            recovered = self.warm_up()
            if recovered:
                self._recovery_cooldown_until = 0.0
                logger.info("OVRTX daemon recovery completed")
            else:
                self._recovery_cooldown_until = (
                    time.monotonic() + _RECOVERY_FAILURE_COOLDOWN_SECONDS
                )
                logger.error("OVRTX daemon recovery failed")
            return recovered

    def _render_backend_once(
        self,
        *,
        stage: Any,
        base_dir: str | Path | None,
        camera_paths: list[str],
        width: int,
        height: int,
        frames: str,
        ovrtx_sensors: list[str],
        num_sensor_updates: int | None,
        render_mode: str | None,
        material_target: str | None,
    ) -> dict[str, Any]:
        return self._backend.render(
            stage=stage,
            cameras=camera_paths,
            image_width=width,
            image_height=height,
            frames=frames,
            sensors=ovrtx_sensors or None,
            num_sensor_updates=num_sensor_updates,
            render_mode=render_mode,
            material_target=material_target,
            base_dir=base_dir,
        )

    def _render_backend_with_recovery(
        self,
        *,
        stage: Any,
        base_dir: str | Path | None,
        camera_paths: list[str],
        width: int,
        height: int,
        frames: str,
        ovrtx_sensors: list[str],
        num_sensor_updates: int | None,
        render_mode: str | None,
        material_target: str | None,
    ) -> dict[str, Any]:
        try:
            return self._render_backend_once(
                stage=stage,
                base_dir=base_dir,
                camera_paths=camera_paths,
                width=width,
                height=height,
                frames=frames,
                ovrtx_sensors=ovrtx_sensors,
                num_sensor_updates=num_sensor_updates,
                render_mode=render_mode,
                material_target=material_target,
            )
        except Exception as exc:
            if not _is_recoverable_render_error(exc):
                raise
            logger.warning(
                "OVRTX daemon render failed with a recoverable error; "
                "restarting daemon and retrying once",
                exc_info=True,
            )
            if not self.recover(force=True):
                raise RuntimeError("OVRTX daemon recovery failed") from exc

        return self._render_backend_once(
            stage=stage,
            base_dir=base_dir,
            camera_paths=camera_paths,
            width=width,
            height=height,
            frames=frames,
            ovrtx_sensors=ovrtx_sensors,
            num_sensor_updates=num_sensor_updates,
            render_mode=render_mode,
            material_target=material_target,
        )

    def render(
        self,
        url: str,
        camera_paths: list[str],
        frame_start: int,
        frame_end: int,
        width: int,
        height: int,
        sensors: list[str] | None = None,
        num_sensor_updates: int | None = None,
        render_mode: str | None = None,
        material_target: str | None = None,
    ) -> dict[str, Any]:
        """Render a USD file and return V1-format response.

        Args:
            url: USD file URL (http/https) or data URI.
            camera_paths: Camera prim paths to render.
            frame_start: First frame to render.
            frame_end: Last frame to render (inclusive).
            width: Output image width.
            height: Output image height.
            sensors: Optional sensor names (Kit API names).
            num_sensor_updates: Per-request number of progressive
                ``renderer.step(dt=0)`` iterations per frame. ``None``
                falls back to the instance default
                (``OVRTX_NUM_SENSOR_UPDATES``, default 500 — the
                convergence plateau on the kit golden scene).
            render_mode: ``rt1`` | ``rt2`` | ``pt``. ``None`` falls back
                to the instance default (``OVRTX_RENDER_MODE``, default
                ``pt`` — the only mode that reaches Kit-parity quality).

        Returns:
            V1 response dict: {status, error, images}.
        """
        from pxr import Usd

        # Map requested sensors to OVRTX names, track unsupported
        ovrtx_sensors: list[str] = []
        unsupported_sensors: list[str] = []
        for s in sensors or []:
            mapped = _SENSOR_KIT_TO_OVRTX.get(s)
            if mapped and mapped in _SUPPORTED_SENSORS:
                if mapped not in ovrtx_sensors:
                    ovrtx_sensors.append(mapped)
            else:
                unsupported_sensors.append(s)

        if unsupported_sensors:
            logger.warning(
                "Unsupported sensors (will be empty in response): %s",
                unsupported_sensors,
            )

        # Fetch USD to temp file. Use the extension-agnostic `.usd` name so
        # USD's format sniffing picks the right reader for the content —
        # `.usda` ASCII and `.usdc` binary crate both work. Hardcoding
        # `.usdc` broke ASCII payloads ("Sdf crate bootstrap section corrupt").
        tmp_dir = tempfile.mkdtemp(prefix="render_api_")
        usd_path = os.path.join(tmp_dir, "input.usd")

        try:
            _fetch_usd(url, usd_path)

            # ZIP payloads include generic render_nvcf bundles and USDZ
            # packages. Extract both forms so relative texture files are real
            # files on disk before render_ovrtx re-exports the stage into its
            # daemon IPC directory. Opening a USDZ package directly keeps
            # textures behind ArPackageResolver; they are then lost on export.
            if zipfile.is_zipfile(usd_path):
                usd_path = _extract_zip_bundle(
                    usd_path,
                    tmp_dir,
                    prefer_first_usd=_is_usdz_payload(url, usd_path),
                )

            stage = Usd.Stage.Open(usd_path)
            if not stage:
                return _error_response(f"Failed to open USD stage from: {url}")

            # Build frames string
            if frame_start == frame_end:
                frames = str(frame_start)
            else:
                frames = f"{frame_start}:{frame_end}"

            # Serialize render calls — the OVRTX daemon crashes under
            # concurrent renders. Uvicorn dispatches sync endpoints to a
            # thread pool, so without this lock multiple requests would
            # call _backend.render() in parallel and kill the daemon.
            lock_start = time.time()
            self._render_lock.acquire()
            lock_wait = time.time() - lock_start
            try:
                if lock_wait > 0.05:
                    logger.info(
                        "Acquired OVRTX render lock after %.2fs "
                        "(%d camera(s), frames %s)",
                        lock_wait,
                        len(camera_paths),
                        frames,
                    )
                render_start = time.time()
                result = self._render_backend_with_recovery(
                    stage=stage,
                    base_dir=Path(usd_path).parent,
                    camera_paths=camera_paths,
                    width=width,
                    height=height,
                    frames=frames,
                    num_sensor_updates=num_sensor_updates,
                    ovrtx_sensors=ovrtx_sensors,
                    render_mode=render_mode,
                    material_target=material_target,
                )
                render_elapsed = time.time() - render_start
                logger.info(
                    "OVRTX daemon render completed in %.2fs "
                    "(lock wait %.2fs, %d camera(s), frames %s)",
                    render_elapsed,
                    lock_wait,
                    len(camera_paths),
                    frames,
                )
            finally:
                self._render_lock.release()

            # Convert to V1 response format
            return _to_v1_response(result, sensors or [], ovrtx_sensors, frame_start)

        except Exception as e:
            logger.exception("Render failed")
            return _error_response(str(e))
        finally:
            # Clean up temp files
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)


def _build_smoke_stage() -> Any:
    """Build a tiny in-memory USD stage for warm-up rendering.

    Contains a single unit cube at the origin, a camera at (3,3,3)
    looking at the origin, and a distant key light. Kept deliberately
    minimal — the point is to exercise GPU init, not to render anything
    pretty.
    """
    from pxr import Gf, Usd, UsdGeom, UsdLux

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(stage.DefinePrim("/World", "Xform"))

    cube = UsdGeom.Cube.Define(stage, "/World/Cube")
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(0.6, 0.4, 0.2)])

    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(35.0)
    cam.CreateHorizontalApertureAttr(36.0)
    cam.CreateVerticalApertureAttr(36.0)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.1, 1000.0))
    cam_xform = UsdGeom.Xformable(cam)
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(3.0, 3.0, 3.0))
    cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(-30.0, 45.0, 0.0))

    light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
    light.CreateIntensityAttr(5000.0)
    UsdGeom.Xformable(light).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 30.0, 0.0))

    return stage


def _is_recoverable_render_error(exc: Exception) -> bool:
    """Return True for daemon/process failures worth one restart + retry."""
    if isinstance(exc, TimeoutError):
        return True
    message = str(exc)
    return any(snippet in message for snippet in _RECOVERABLE_RENDER_ERROR_SNIPPETS)


def _parse_legacy_ipv4_part(part: str) -> int | None:
    """Parse one inet_aton-style IPv4 part without accepting hostnames."""
    if not part:
        return None
    if part.lower().startswith("0x"):
        digits = part[2:]
        if not digits or any(ch not in "0123456789abcdefABCDEF" for ch in digits):
            return None
        return int(digits, 16)
    if len(part) > 1 and part.startswith("0"):
        if any(ch not in "01234567" for ch in part):
            return None
        return int(part, 8)
    if not part.isdigit():
        return None
    return int(part, 10)


def _parse_legacy_ipv4_literal(host: str) -> ipaddress.IPv4Address | None:
    """Normalize legacy decimal/octal/hex IPv4 literals accepted by URL stacks."""
    parts = host.split(".")
    if len(parts) not in _LEGACY_IPV4_PART_MAX_VALUES:
        return None
    numeric_parts: list[int] = []
    for part, max_value in zip(
        parts, _LEGACY_IPV4_PART_MAX_VALUES[len(parts)], strict=True
    ):
        value = _parse_legacy_ipv4_part(part)
        if value is None or value > max_value:
            return None
        numeric_parts.append(value)

    if len(numeric_parts) == 1:
        address = numeric_parts[0]
    elif len(numeric_parts) == 2:
        address = (numeric_parts[0] << 24) | numeric_parts[1]
    elif len(numeric_parts) == 3:
        address = (numeric_parts[0] << 24) | (numeric_parts[1] << 16) | numeric_parts[2]
    else:
        address = (
            (numeric_parts[0] << 24)
            | (numeric_parts[1] << 16)
            | (numeric_parts[2] << 8)
            | numeric_parts[3]
        )
    return ipaddress.IPv4Address(address)


def _parse_url_host_ip(
    host: str,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        return _parse_legacy_ipv4_literal(host)


def _normalize_ip_for_url_blocking(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> ipaddress.IPv4Address | ipaddress.IPv6Address:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        return ip.ipv4_mapped
    return ip


def _raise_if_blocked_url_ip(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
    url: str,
) -> None:
    normalized = _normalize_ip_for_url_blocking(ip)
    if (
        normalized.is_unspecified
        or normalized.is_private
        or normalized.is_loopback
        or normalized.is_link_local
    ):
        raise ValueError(f"URL blocked (non-public IP): {url[:80]}")


def _iter_resolved_host_ips(
    host: str,
) -> tuple[ipaddress.IPv4Address | ipaddress.IPv6Address, ...]:
    try:
        results = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return ()

    resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    seen: set[str] = set()
    for result in results:
        sockaddr = result[4]
        if not sockaddr:
            continue
        try:
            ip = ipaddress.ip_address(sockaddr[0])
        except ValueError:
            continue
        normalized = _normalize_ip_for_url_blocking(ip)
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        resolved_ips.append(normalized)
    return tuple(resolved_ips)


def _validate_url_target(url: str) -> None:
    """Block HTTP requests to cloud metadata, loopback, and private addresses."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").rstrip(".").lower()
    if host in _BLOCKED_URL_HOSTS:
        raise ValueError(f"URL blocked (metadata/loopback): {url[:80]}")
    ip = _parse_url_host_ip(host)
    if ip is not None:
        _raise_if_blocked_url_ip(ip, url)
        return

    for resolved_ip in _iter_resolved_host_ips(host):
        _raise_if_blocked_url_ip(resolved_ip, url)


class _PrivateAddressBlockingHTTPConnection(HTTPConnection):
    """HTTP connection that rejects private peers after DNS resolution."""

    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()
        _validate_connected_socket_peer(sock, _url_for_hint("http", self.host))
        return sock


class _PrivateAddressBlockingHTTPSConnection(HTTPSConnection):
    """HTTPS connection that rejects private peers after DNS resolution."""

    def _new_conn(self) -> socket.socket:
        sock = super()._new_conn()
        _validate_connected_socket_peer(sock, _url_for_hint("https", self.host))
        return sock


class _PrivateAddressBlockingHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PrivateAddressBlockingHTTPConnection


class _PrivateAddressBlockingHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PrivateAddressBlockingHTTPSConnection


class _PrivateAddressBlockingPoolManager(PoolManager):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pool_classes_by_scheme = {
            **self.pool_classes_by_scheme,
            "http": _PrivateAddressBlockingHTTPConnectionPool,
            "https": _PrivateAddressBlockingHTTPSConnectionPool,
        }


class _PrivateAddressBlockingAdapter(HTTPAdapter):
    """Requests adapter that checks the connected peer before sending bytes.

    The adapter is mounted on a fresh Session in _safe_requests_get for each
    URL fetch. Keeping the session scoped to one request avoids reusing pooled
    connections across independently preflighted URLs.
    """

    def init_poolmanager(
        self,
        connections: int,
        maxsize: int,
        block: bool = False,
        **pool_kwargs: Any,
    ) -> None:
        self.poolmanager = _PrivateAddressBlockingPoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            **pool_kwargs,
        )


def _validate_connected_socket_peer(sock: socket.socket, url_hint: str) -> None:
    try:
        peer_host = sock.getpeername()[0]
    except OSError:
        sock.close()
        raise

    try:
        ip = ipaddress.ip_address(peer_host)
    except ValueError:
        sock.close()
        raise

    try:
        _raise_if_blocked_url_ip(ip, url_hint)
    except ValueError:
        sock.close()
        raise


def _url_for_hint(scheme: str, host: str) -> str:
    return f"{scheme}{_URL_SCHEME_SEPARATOR}{host}"


def _safe_requests_get(url: str, *, timeout: float, allow_redirects: bool):
    # Scope the session to this single URL so every fetch gets a fresh DNS
    # preflight plus connected-peer validation before response bytes flow.
    # Mount both schemes because user-supplied HTTP URLs are allowed only after
    # SSRF preflight and connected-peer blocking.
    with requests.Session() as session:
        session.trust_env = False
        adapter = _PrivateAddressBlockingAdapter()
        session.mount(_url_for_hint("http", ""), adapter)
        session.mount(_url_for_hint("https", ""), adapter)
        return session.get(url, timeout=timeout, allow_redirects=allow_redirects)


def _safe_http_get(url: str, *, timeout: float):
    """Fetch http(s), manually validating every redirect target."""
    current_url = url
    for _ in range(_MAX_HTTP_REDIRECTS + 1):
        _validate_url_target(current_url)
        resp = _safe_requests_get(
            current_url,
            timeout=timeout,
            allow_redirects=False,
        )
        if resp.status_code not in _HTTP_REDIRECT_STATUSES:
            return resp

        location = resp.headers.get("Location")
        resp.close()
        if not location:
            raise ValueError(f"HTTP redirect missing Location header: {current_url}")
        next_url = urljoin(current_url, location)
        if urlparse(next_url).scheme.lower() not in _REMOTE_USD_SCHEMES:
            raise ValueError(f"Unsupported redirect URL scheme: {next_url[:50]}")
        current_url = next_url

    raise ValueError(f"Too many redirects while fetching USD: {url[:80]}")


def _is_usdz_payload(url: str, zip_path: str) -> bool:
    """Return True if the ZIP should open as a native ``.usdz`` package.

    Primary signal is the URL path: a ``.usdz`` suffix on http/https/s3
    URLs is the caller's declaration that they want package semantics
    (default prim, internal ``package.usdz[inner.usda]`` references)
    preserved. For data URIs, which carry no path, fall back to the
    structural signature of the Pixar USDZ spec — all entries stored
    uncompressed with a USD layer as the first member. That shape is
    narrow enough that it only matches archives produced by the USDZ
    tooling (``usdzip`` and equivalents), not generic bundles.

    This keeps render_nvcf client bundles on the extraction path (they
    upload ``.zip`` with ``ZIP_DEFLATED``) while letting real USDZ
    assets reach Usd.Stage.Open intact.
    """
    if url.startswith("data:"):
        return _zip_matches_usdz_structure(zip_path)
    return urlparse(url).path.lower().endswith(".usdz")


def _zip_matches_usdz_structure(zip_path: str) -> bool:
    """Narrow check for the Pixar USDZ archive layout (stored + USD first)."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            infos = zf.infolist()
    except zipfile.BadZipFile:
        return False
    if not infos:
        return False
    if any(info.compress_type != zipfile.ZIP_STORED for info in infos):
        return False
    first = infos[0].filename.lower()
    return any(first.endswith(ext) for ext in _USD_EXTENSIONS)


def _extract_zip_bundle(
    zip_path: str,
    tmp_dir: str,
    prefer_first_usd: bool = False,
) -> str:
    """Extract a USD+assets bundle and return the path to the main USD layer.

    Selection priority follows kit-gen-ai-service/common/file_handler.py so the
    two services share behavior: ``main.*`` → ``scene.*`` → ``stage.*`` (the
    root name used by render_remote._bundle_stage_with_local_assets) → first
    match alphabetically. For USDZ payloads, ``prefer_first_usd`` selects the
    first USD layer in archive order because the USDZ spec uses that as the
    package root. Raises ValueError if no USD layer is present.
    """
    extract_dir = os.path.join(tmp_dir, "bundle")
    os.makedirs(extract_dir, exist_ok=True)
    extract_root = Path(extract_dir).resolve()
    with zipfile.ZipFile(zip_path, "r") as zf:
        infos = zf.infolist()

        # ZIP-bomb guard: reject obvious oversized archives from central-directory
        # metadata, then enforce the same limit on actual streamed bytes below.
        if len(infos) > _ZIP_MAX_FILES:
            raise ValueError(
                f"ZIP bundle has too many entries: {len(infos)} "
                f"(limit {_ZIP_MAX_FILES})"
            )
        total_uncompressed = sum(info.file_size for info in infos)
        max_uncompressed = _zip_max_uncompressed_bytes()
        if total_uncompressed > max_uncompressed:
            raise ValueError(
                f"ZIP bundle uncompressed size too large: "
                f"{total_uncompressed} bytes (limit {max_uncompressed})"
            )

        # Python's zipfile already strips leading "/" and ".." components, but
        # this endpoint opens URLs/data URIs from external callers so we
        # belt-and-suspenders reject: (a) entries that resolve outside
        # extract_root after symlink expansion and (b) symlink entries that
        # extractall would otherwise materialize on disk (0xA000 = S_IFLNK).
        total_written = 0
        for info in infos:
            if (info.external_attr >> 16) & 0xF000 == 0xA000:
                raise ValueError(f"ZIP bundle contains symlink entry: {info.filename}")
            resolved = (extract_root / info.filename).resolve()
            if extract_root not in resolved.parents and resolved != extract_root:
                raise ValueError(
                    f"ZIP bundle contains unsafe entry path: {info.filename}"
                )
            if info.is_dir():
                resolved.mkdir(parents=True, exist_ok=True)
                continue

            resolved.parent.mkdir(parents=True, exist_ok=True)
            try:
                with zf.open(info) as src, resolved.open("wb") as dst:
                    total_written += copy_stream_limited(
                        src,
                        dst,
                        max_bytes=max_uncompressed - total_written,
                    )
            except ArchiveSizeLimitExceeded as exc:
                resolved.unlink(missing_ok=True)
                raise ValueError(
                    "ZIP bundle uncompressed size too large while extracting: "
                    f"limit {max_uncompressed} bytes"
                ) from exc

    # Walk everything once and filter by a lowercased suffix so producers
    # that mint entries like ``MAIN.USDA`` are still recognized on
    # case-sensitive filesystems (Linux). ``Path.rglob(f"*{ext}")`` would
    # otherwise be case-sensitive and drop those archives.
    usd_files = sorted(
        p
        for p in Path(extract_dir).rglob("*")
        if p.is_file() and not p.is_symlink() and p.suffix.lower() in _USD_EXTENSIONS
    )
    if not usd_files:
        raise ValueError(
            f"No USD layer found in ZIP bundle (expected one of {_USD_EXTENSIONS})"
        )

    if prefer_first_usd:
        for info in infos:
            if Path(info.filename).suffix.lower() not in _USD_EXTENSIONS:
                continue
            first_usd = Path(extract_dir) / info.filename
            if first_usd.is_file() and not first_usd.is_symlink():
                return str(first_usd)

    for preferred in ("main", "scene", "stage"):
        for p in usd_files:
            if p.stem.lower() == preferred:
                return str(p)
    return str(usd_files[0])


def _fetch_usd(url: str, dest_path: str) -> None:
    """Fetch USD file from URL, data URI, or S3 to a local path."""
    if url.startswith("data:"):
        # data:application/octet-stream;base64,<data>
        parts = url.split(",", 1)
        if len(parts) != 2 or not parts[1]:
            raise ValueError(
                f"Malformed data URI: missing comma or payload: {url[:60]}"
            )
        try:
            raw = base64.b64decode(parts[1], validate=True)
        except (ValueError, binascii.Error) as e:
            raise ValueError(f"Malformed data URI: invalid base64: {e}") from e
        with open(dest_path, "wb") as f:
            f.write(raw)
        return

    scheme = urlparse(url).scheme.lower()
    if scheme == "s3":
        _download_s3(url, dest_path)
        return
    # Accept http/https here because material-agent-service and internal
    # tools call this service over the in-cluster plain-text network (no
    # TLS terminator between the pods). External callers are expected to
    # use https or s3://.
    if scheme in _REMOTE_USD_SCHEMES:
        # Check if this is an S3 HTTPS URL (e.g. bucket.s3.region.amazonaws.com/key)
        s3_url = _https_to_s3(url)
        if s3_url:
            _download_s3(s3_url, dest_path)
            return
        resp = _safe_http_get(url, timeout=300)
        resp.raise_for_status()
        with open(dest_path, "wb") as f:
            f.write(resp.content)
        return

    raise ValueError(f"Unsupported URL scheme: {url[:50]}")


def _https_to_s3(url: str) -> str | None:
    """Convert an S3 HTTPS URL to s3:// URI, or None if not an S3 URL."""
    m = _S3_VHOST_RE.match(url) or _S3_PATH_RE.match(url)
    if m:
        return f"s3://{m.group('bucket')}/{m.group('key')}"
    return None


def _download_s3(s3_url: str, dest_path: str) -> None:
    """Download from s3://bucket/key to a local path."""
    import boto3
    from botocore.exceptions import ClientError, ProfileNotFound

    parts = s3_url[5:].split("/", 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ""
    # Try AWS_PROFILE env var, then default credential chain
    profile = os.environ.get("AWS_PROFILE")
    if profile:
        session = boto3.Session(profile_name=profile)
    else:
        # Fall back to the "default" profile if it exists; otherwise let
        # boto3 use its standard credential chain (env vars, instance role).
        session = None
        try:
            candidate = boto3.Session(profile_name="default")
        except ProfileNotFound as e:
            logger.debug("S3 profile 'default' not found: %s", e)
        else:
            try:
                candidate.client("s3").head_bucket(Bucket=bucket)
            except ClientError as e:
                logger.debug(
                    "S3 profile 'default' cannot access bucket %s: %s", bucket, e
                )
            else:
                logger.debug("S3 profile 'default' resolved for bucket %s", bucket)
                session = candidate
        if session is None:
            logger.debug(
                "No named S3 profile worked for bucket %s, using default "
                "credential chain",
                bucket,
            )
            session = boto3.Session()
    s3 = session.client("s3")
    logger.info("Downloading s3://%s/%s", bucket, key)
    s3.download_file(bucket, key, dest_path)


def _to_v1_response(
    result: dict[str, Any],
    requested_sensors: list[str],
    ovrtx_sensors: list[str],
    frame_start: int,
) -> dict[str, Any]:
    """Convert OvRTXRenderingBackend output to V1 response format.

    V1 format: images[frame_str][camera_path][sensor_name] = base64 string

    ``frame_start`` is added to the 0-based enumerate index so that the
    ``v1_images`` keys are the *absolute* frame numbers the client asked
    for (not 0,1,2,...). Sensor lookups are keyed on the same absolute
    frame number because OVRTX returns sensor dicts keyed on the real
    frame index. Getting this wrong is silent — the main RGB image is
    keyed 0-based so looks correct, but sensor data comes back empty for
    any render where frame_start != 0.
    """
    v1_images: dict[str, dict[str, dict[str, str]]] = {}
    blank_frames: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_blank_frames: set[tuple[str, int]] = set()
    seen_warnings: set[str] = set()
    top_level_blank_frames = result.get("blank_render_frames", [])

    def add_warning(message: str) -> None:
        if message in seen_warnings:
            return
        warnings.append(message)
        seen_warnings.add(message)

    def add_blank_frame(blank_frame: dict[str, Any]) -> None:
        key = (str(blank_frame["camera"]), int(blank_frame["frame"]))
        if key in seen_blank_frames:
            return
        blank_frames.append(blank_frame)
        seen_blank_frames.add(key)
        add_warning(_blank_frame_warning(blank_frame))

    for cam_result in result.get("results", []):
        camera = cam_result["camera"]
        images: list[Image.Image] = cam_result.get("images", [])
        sensor_data: dict[str, dict[int, np.ndarray]] = cam_result.get("sensors", {})
        image_frames = _image_frames_for_response(cam_result, len(images), frame_start)
        upstream_blank_frames = _blank_frames_by_frame(
            cam_result.get("blank_render_frames", []),
            default_camera=camera,
            valid_frames=set(image_frames),
        )
        if not upstream_blank_frames:
            upstream_blank_frames = _blank_frames_by_frame(
                top_level_blank_frames,
                default_camera=camera,
                valid_frames=set(image_frames),
            )
        if upstream_blank_frames:
            for blank_frame in upstream_blank_frames.values():
                add_blank_frame(blank_frame)
            for warning in _string_list(cam_result.get("warnings", [])):
                add_warning(warning)

        for frame_idx, img in enumerate(images):
            actual_frame = image_frames[frame_idx]
            frame_str = str(actual_frame)
            if frame_str not in v1_images:
                v1_images[frame_str] = {}

            camera_data: dict[str, str] = {}

            if not upstream_blank_frames:
                stats = analyze_image_blankness(
                    img,
                    max_analysis_pixels=_SERVICE_BLANKNESS_MAX_ANALYSIS_PIXELS,
                )
                if stats.blank:
                    add_blank_frame(
                        {
                            "frame": actual_frame,
                            "camera": camera,
                            "stats": stats.to_dict(),
                        }
                    )

            # Main RGB image -> base64 PNG
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            camera_data["images"] = base64.b64encode(png_bytes).decode()

            # Sensor data -> base64 raw bytes
            for req_sensor in requested_sensors:
                ovrtx_name = _SENSOR_KIT_TO_OVRTX.get(req_sensor)
                if ovrtx_name and ovrtx_name in sensor_data:
                    arr = sensor_data[ovrtx_name].get(actual_frame)
                    if arr is not None:
                        camera_data[req_sensor] = base64.b64encode(
                            arr.tobytes()
                        ).decode()
                    else:
                        camera_data[req_sensor] = ""
                else:
                    # Unsupported sensor -- return empty string
                    camera_data[req_sensor] = ""

            v1_images[frame_str][camera] = camera_data

    response: dict[str, Any] = {
        "status": "success",
        "error": None,
        "images": v1_images,
    }
    if warnings:
        response["warnings"] = warnings
        response["blank_render_frames"] = blank_frames
    return response


def _image_frames_for_response(
    cam_result: dict[str, Any],
    image_count: int,
    frame_start: int,
) -> list[int]:
    raw_frames = cam_result.get("image_frames", [])
    if not isinstance(raw_frames, list) or len(raw_frames) < image_count:
        return [frame_start + index for index in range(image_count)]

    frame_numbers: list[int] = []
    for index, raw_frame in enumerate(raw_frames[:image_count]):
        if isinstance(raw_frame, int) and raw_frame >= 0:
            frame_numbers.append(raw_frame)
        else:
            frame_numbers.append(frame_start + index)
    return frame_numbers


def _blank_frames_by_frame(
    raw_frames: Any,
    *,
    default_camera: str,
    valid_frames: set[int] | None = None,
) -> dict[int, dict[str, Any]]:
    if not isinstance(raw_frames, list):
        return {}

    frames: dict[int, dict[str, Any]] = {}
    for raw_frame in raw_frames:
        blank_frame = _normalize_blank_frame(raw_frame, default_camera=default_camera)
        if blank_frame is None:
            continue
        if str(blank_frame["camera"]) != str(default_camera):
            continue
        frame = int(blank_frame["frame"])
        if valid_frames is not None and frame not in valid_frames:
            continue
        frames[frame] = blank_frame
    return frames


def _normalize_blank_frame(
    raw_frame: Any,
    *,
    default_camera: str,
) -> dict[str, Any] | None:
    if not isinstance(raw_frame, dict):
        return None

    frame = raw_frame.get("frame")
    if not isinstance(frame, int) or frame < 0:
        return None

    camera = raw_frame.get("camera", default_camera)
    if not isinstance(camera, str):
        camera = default_camera

    stats = raw_frame.get("stats")
    if not isinstance(stats, dict):
        stats = {"blank": True, "reason": "remote_blank_render"}

    normalized = {
        "frame": frame,
        "camera": camera,
        "stats": stats,
    }
    image_file = raw_frame.get("image_file")
    if isinstance(image_file, str):
        normalized["image_file"] = image_file
    return normalized


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _blank_frame_warning(blank_frame: dict[str, Any]) -> str:
    stats = blank_frame.get("stats", {})
    dominant_color_ratio = float(stats.get("dominant_color_ratio", 0.0) or 0.0)
    luma_std = float(stats.get("luma_std", 0.0) or 0.0)
    return (
        "Blank or near-blank render detected "
        f"for frame {blank_frame['frame']} camera {blank_frame['camera']}: "
        f"{stats.get('reason')} "
        f"(unique_colors={stats.get('unique_colors')}, "
        f"dominant_color_ratio={dominant_color_ratio:.3f}, "
        f"luma_std={luma_std:.3f})"
    )


def _error_response(message: str) -> dict[str, Any]:
    """Build an error response dict."""
    return {
        "status": "exception",
        "error": message,
        "images": {},
    }
