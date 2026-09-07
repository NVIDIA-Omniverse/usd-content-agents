# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote render backend — REST client to an OVRTX render service.

Same interface as the local OVRTX daemon (the render-execution decision): the only
difference is the daemon lives at `remote_url` instead of localhost. Targets the
`apps/ovrtx_rendering_api` contract.

The wire payload is a self-contained **USDZ** bundle (geometry + textures), so a SimReady
asset's material maps render remotely instead of arriving as dangling file references.
Transport preference: against a backend advertising the `cas` feature, the bundle
travels as a **content-addressed manifest** (its zip entries hashed per file —
negotiate what's missing, PUT only those blobs, render by manifest; see
`_dispatch_cas` and usd_core.render.remote_cas) so a re-render after a small edit
uploads KBs instead of the scene; otherwise `POST /render/upload` — a streamed
multipart upload of the (optionally gzip-compressed) binary USDZ with progress
reporting — with the legacy base64-in-JSON `POST /render` as the final fallback
for older services. Payload sizes are logged and included
in errors, and `render.remote_max_upload_mb` (config default 512; 0 = unlimited) bounds
the upload client-side — both as a pre-packaging estimate (fail fast before minutes of
bundling) and on the final bytes before any of them leave the machine. Backend errors
carry the response body (truncated), and every packaging/upload stage logs a progress
line to the daemon log.

**Multi-backend pool.** `[[render.backends]]` in config (see
`usd_core.config.resolve_render_backends`) gives the client a fleet of render services
instead of one. Single renders go to the healthy backend with the fewest in-flight
requests (round-robin tiebreak); `render_frames` with N>1 frames shards the frame list
into contiguous chunks — one per healthy backend — packages the scene ONCE, uploads the
same bundle to each backend in parallel threads (uploads are I/O-bound), and reassembles
the results in frame order. A backend that fails its version handshake (or is
unreachable) is marked unhealthy with one warning and skipped; the pool errors only when
every backend is out. With a one-entry pool every path is exactly the single-backend
behavior. Multi-camera `render` calls (e.g. an orbit chunk) stay one remote call — the
service ingests the scene once for all cameras in the request — and are not sharded.
Pool scheduling state (slot health, in-flight counts, the round-robin cursor)
persists process-wide across backend constructions via a registry keyed by the
normalized pool config, so per-command clients share one scheduler instead of every
render starting over at backend zero (see _pool_state_for).

**Scheduling policy (throughput/failure-aware).** In-flight counts alone let a
degraded node keep winning jobs (benchmark: a slot that took 9m12s for an 18 MB
upload — a sibling did the same payload in 4.5s — still received two of three jobs).
Each slot therefore keeps a rolling record of its last `_THROUGHPUT_WINDOW`
successful upload throughputs (bytes/s) plus a recent-failure flag (set on any
upload/backend error or stall, cleared by the next successful upload). Selection:
slots with a recent failure (within `_FAILURE_COOLDOWN_S` of it), or whose average
recent throughput is below `_DEMOTE_BPS_FRACTION` (10%) of the best among slots
WITHOUT a recent failure (a fast-but-failing node must not set the baseline that
demotes its healthy, slower peers), are *demoted* — skipped while any non-demoted
slot exists, but still used when nothing better is available (never starve).
Failure demotion is not permanent: after the cooldown the slot re-enters selection
as a canary (a success clears the flag, another failure re-demotes it), and even
inside the cooldown, once `_CANARY_EVERY` consecutive picks have gone to
non-demoted slots one render is routed to the best demoted slot so it can prove
recovery — without this, a transient failure starved a slot forever (it could
never get the success needed to clear the flag). Among the remaining slots the
pick is least-busy first, then highest recent throughput (slots with no samples
yet rank as best, so new slots get measured), then round-robin. Frame sharding
(`_execute_sharded`) fans out only across non-demoted healthy slots for the same
reason — a degraded node would be the long pole of the whole batch.

**Bundle cache.** Packaging + gzip of a large scene takes minutes; one benchmark
night packaged an identical 1.38 GB bundle four times within 15 minutes. Packaged
bundles are therefore cached under the private work root when a launcher pins one,
or the per-user staging root otherwise, keyed by a sha256
fingerprint of everything the bundle depends on: each used layer's serialized
content when dirty/anonymous, else its realPath+size+mtime_ns — plus, for clean
on-disk layers up to `_CACHE_CONTENT_HASH_MAX_BYTES`, the file's content sha (a
same-size, same-mtime byte change must not serve a stale bundle; for larger clean
layers the stat key stands and that collision remains a documented residual
risk) — every resolved external asset dependency (textures, payloads) by
path+size+mtime_ns (editing a texture in place invalidates the bundle; when the
dependency walk fails the stage is simply uncacheable), and the packaging options.
A hit hard-links (or copies) the cached usdz *and* its gzip into the render's work
directory, skipping packaging and compression while giving the upload an artifact
lifetime independent of cache eviction; the cache keeps the newest
`_BUNDLE_CACHE_KEEP` bundles per scene stem (LRU by directory mtime) and drops
entries older than 24h. Uploads are still per-request — the remote service ingests
per job — but re-*packaging* is gone.
Disable with `render.remote_bundle_cache = false`. Trust: the cache lives only
inside a private 0700 root — when that root is unavailable (squatted, shared-tmp
fallback) caching is disabled outright, the cache root must be an owned,
mode-0700, non-symlink directory, and hits are served only from regular, owned,
non-symlink artifact files (verified via O_NOFOLLOW).
"""

from __future__ import annotations

import base64
import contextvars
import gzip
import json
import logging
import os
import re
import shutil
import stat as stat_mod
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from usd_core.render.base import RenderResult, prepare_render_input

logger = logging.getLogger(__name__)

_MB = 1024 * 1024
# config.py's render.remote_max_upload_mb default — the only value
# _effective_cap_mb treats as "unset, adopt the backend's advertised limit"
_DEFAULT_MAX_UPLOAD_MB = 512.0


def _zstd_module():
    """The zstandard module, or None — zstd is optional on both ends and is used
    only when the local library exists AND
    every verified backend advertises the feature; gzip stays the fallback."""
    try:
        import zstandard
        return zstandard
    except Exception:  # noqa: BLE001 — any import problem = feature absent
        return None


# Upload-dedup measurement:
# cumulative re-upload waste. Identity is (realpath, size, mtime_ns) — exact for
# cache-hit re-uploads (the case dedup targets: the cached bundle file is stable
# across renders); same-content-different-packaging is already visible as a
# bundle-cache miss. Bounded; never fails an upload.
_UPLOAD_LEDGER_LOCK = threading.Lock()
_UPLOAD_LEDGER: dict[tuple, int] = {}
_UPLOAD_RESHIPPED = [0]  # duplicate bytes shipped by this process
_ERROR_BODY_LIMIT = 500  # chars of backend response body carried into errors/logs
_STAGING_PREFIX = "usd_cli_remote_render_"  # staging tmpdirs: usd_cli_remote_render_<pid>_<rand>
#: pre-rename staging prefixes — the reaper still recognizes leftovers from older builds
_LEGACY_STAGING_PREFIXES = ("ov_remote_render_", "dsc_remote_render_")
_STAGING_MAX_AGE_S = 24 * 3600  # reap orphaned staging dirs older than this
REMOTE_RENDER_STAGING_ROOT_ENV = "USD_CLI_REMOTE_RENDER_STAGING_ROOT"

# pool scheduling (see the module docstring's "Scheduling policy" section)
_THROUGHPUT_WINDOW = 5  # rolling record: last N successful upload throughputs per slot
_DEMOTE_BPS_FRACTION = 0.10  # under 10% of the pool's best recent throughput → demoted
_FAILURE_COOLDOWN_S = 120.0  # failure demotion expires; the slot re-enters as a canary
_CANARY_EVERY = 10  # after N consecutive non-demoted picks, try the best demoted slot
# A stalled upload (no socket write progressing for this long) fails instead of
# holding the render for the full request timeout — the failure then demotes the
# slot like any other error (benchmark: a mid-transfer stall never hit the failure
# path and the stalled node kept winning the throughput tiebreak).
_UPLOAD_WRITE_STALL_S = 60.0
# A backend marked unusable (unreachable/protocol mismatch) gets ONE re-probe
# after this cooldown — permanent session-scoped blindness outlived transient
# network blips in round 7.
_UNHEALTHY_REPROBE_S = 120.0
# Rolling minimum upload throughput: a crawling-but-not-stalled transfer (round 7:
# the same 18 MB took 42s on one node and ~6 min on another) aborts and fails
# over to a sibling instead of monopolizing the render for minutes.
_UPLOAD_MIN_BPS = 50 * 1024
_UPLOAD_MIN_BPS_WINDOW_S = 45.0
# Slot health survives daemon restarts: persisted per pool config, entries expire
# after this long (a restart was the agent's natural response to a stall, and it
# wiped the demotion state that would have avoided the slow backend).
_POOL_STATE_TTL_S = 3600.0
_POOL_STATE_VERSION = 1

# packaged-bundle cache (see the module docstring's "Bundle cache" section)
_BUNDLE_CACHE_DIRNAME = "dsc3_bundle_cache"
_BUNDLE_CACHE_KEEP = 4  # newest bundles kept per scene stem (LRU by dir mtime)
_ESTIMATE_MAX_DEPS = 10_000  # cap the pre-packaging dependency stat walk (speed)
_CACHE_CONTENT_HASH_MAX_BYTES = 8 * _MB  # clean layers up to this size get a content sha

# usdUtils asset localization re-warns about its OWN package-internal remaps
# ("@0/tex.png@" — numbered archive dirs it just created); those assets ARE bundled
_PKG_INTERNAL_REMAP_RE = re.compile(r"@\d+/[^@]*@")

# C0 control chars (incl. ESC) + DEL become visible \xNN escapes: backend response
# bodies are untrusted, and a compromised backend must not be able to inject terminal
# escape sequences or forge log lines via the error/log surfaces that carry its text.
_CTRL_ESCAPES = {i: f"\\x{i:02x}" for i in (*range(0x20), 0x7F)}


def _sanitize_text(text: str) -> str:
    """Untrusted (backend-supplied) text made terminal/log-safe — see _CTRL_ESCAPES."""
    return text.translate(_CTRL_ESCAPES)


def _monotonic() -> float:
    """time.monotonic, indirected so tests can simulate the failure-cooldown clock."""
    return time.monotonic()


# Transient render cameras the tool authored (usd_core.camera tags them; the second
# key is the pre-rename tag still carried by stages saved by 3dsc builds).
_MANAGED_CAM_KEYS = ("usdManagedCamera", "ovManagedCamera", "dscManagedCamera")
# Any usd-cli-authored camera (incl. user `camera create` ones) is fully described by a
# camera_def, so a render can strip THE CAMERAS IT WAS ASKED TO RENDER from the
# bundle and keep the bundle viewpoint-independent. Only requested cameras are
# stripped — other authored cameras stay scene content.
_AUTHORED_CAM_KEYS = _MANAGED_CAM_KEYS + ("usdAuthoredCamera", "ovAuthoredCamera")

# Hook the daemon sets around a render dispatch: called once, right after the
# stage is fully packaged, to release the per-root write lock early. Everything
# after packaging is network + local file IO — holding the root exclusively
# through a multi-minute upload/render blocked every read-only audit session
# on the same root (benchmark round 5). Same-thread contract: dispatch runs the
# render synchronously, so a plain ContextVar reaches this module.
STAGE_RELEASE_HOOK: "contextvars.ContextVar[Callable[[], None] | None]" = (
    contextvars.ContextVar("ov_stage_release_hook", default=None))


# Prim paths of THIS render's requested cameras that travel as camera_defs —
# packaging and the cache fingerprint strip exactly these (plus always-managed
# cameras). Same-thread contract as STAGE_RELEASE_HOOK.
_CAMERA_STRIP_PATHS: "contextvars.ContextVar[frozenset]" = (
    contextvars.ContextVar("usd_cli_camera_strip_paths", default=frozenset()))


def _signal_stage_released() -> None:
    """Invoke (once) the daemon's early root-lock release, if any. Errors in the
    hook must never fail a render that is otherwise proceeding."""
    hook = STAGE_RELEASE_HOOK.get()
    if hook is None:
        return
    STAGE_RELEASE_HOOK.set(None)
    try:
        hook()
    except Exception:  # noqa: BLE001 — lock bookkeeping must not kill the render
        logger.debug("stage-release hook failed", exc_info=True)


def _is_managed_camera_prim(prim) -> bool:
    """True for a live Usd.Prim that is a tool-authored transient render camera."""
    try:
        if not prim or not prim.IsValid() or prim.GetTypeName() != "Camera":
            return False
        data = prim.GetCustomData() or {}
        return any(data.get(k) for k in _AUTHORED_CAM_KEYS)
    except Exception:  # noqa: BLE001 — a broken prim is not a managed camera
        return False


def _managed_camera_defs(stage, cameras: list[str]) -> list[dict]:
    """Wire-format specs for the requested cameras that are tool-authored.

    These cameras are STRIPPED from the packaged bundle (so the bundle content —
    and its cache fingerprint — no longer changes with every viewpoint) and
    re-authored by the render service from these specs (protocol v2). User-created
    cameras are real scene content: they stay in the bundle and are rendered by
    path exactly as before, so they never appear here.
    """
    from pxr import Usd, UsdGeom

    if stage is None or not hasattr(stage, "GetPrimAtPath"):
        return []  # stubbed/absent stage (tests): nothing to extract
    defs: list[dict] = []
    for path in cameras:
        prim = stage.GetPrimAtPath(str(path))
        if not _is_managed_camera_prim(prim):
            continue
        cam = UsdGeom.Camera(prim)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        spec: dict = {
            "path": str(path),
            # row-major 16 floats — Gf.Matrix4d iterates rows
            "matrix": [float(v) for row in m for v in row],
        }
        for key, attr in (("focal_length", cam.GetFocalLengthAttr()),
                          ("horizontal_aperture", cam.GetHorizontalApertureAttr()),
                          ("vertical_aperture", cam.GetVerticalApertureAttr())):
            val = attr.Get()
            if val is not None:
                spec[key] = float(val)
        clip = cam.GetClippingRangeAttr().Get()
        if clip is not None:
            spec["clipping_range"] = [float(clip[0]), float(clip[1])]
        proj = cam.GetProjectionAttr().Get()
        if proj:
            spec["projection"] = str(proj)
        defs.append(spec)
    return defs


def _strip_managed_cameras_spec(layer, extra_paths: set | None = None) -> int:
    """Remove tool-authored camera prim SPECS from an Sdf layer (recursively);
    returns how many were removed. Managed (transient) cameras are always
    stripped; `extra_paths` names usd-cli-AUTHORED cameras this render was asked to
    render (they travel as camera_defs), which are stripped too. Mirrors the
    strip `save` applies to deliverables — here it keeps viewpoints out of the
    render bundle so the bundle fingerprint stays camera-independent."""
    removed = 0
    extra = extra_paths or set()

    def _walk(prim_spec):
        nonlocal removed
        for child in list(prim_spec.nameChildren):
            data = child.customData or {}
            if child.typeName == "Camera" and (
                    any(data.get(k) for k in _MANAGED_CAM_KEYS)
                    or str(child.path) in extra):
                del prim_spec.nameChildren[child.name]
                removed += 1
            else:
                _walk(child)

    _walk(layer.pseudoRoot)
    return removed


def _session_contains_only_managed_cameras(stage) -> bool:
    """Whether the session is empty after removing transmitted render cameras.

    A USDZ root can be forwarded byte-for-byte only when no semantic session
    opinions need composing. Protocol-v2 cameras travel separately as
    ``camera_defs``, so their specs (and empty ancestor overs) do not count.
    """
    from pxr import Sdf

    session = stage.GetSessionLayer()
    if session is None or session.empty:
        return True
    copy = Sdf.Layer.CreateAnonymous(".usda")
    copy.TransferContent(session)
    removed = _strip_managed_cameras_spec(
        copy,
        extra_paths=set(_CAMERA_STRIP_PATHS.get()),
    )

    def _prune_empty_overs(prim_spec) -> None:
        for child in list(prim_spec.nameChildren):
            _prune_empty_overs(child)
            if (
                child.specifier == Sdf.SpecifierOver
                and not child.nameChildren
                and not child.properties
                and set(child.ListInfoKeys()) <= {"specifier"}
            ):
                del prim_spec.nameChildren[child.name]

    # Defining a session camera below an existing prim authors ancestor `over`
    # specs (for example `/World`). After removing that camera those empty
    # ancestors are not semantic session opinions.
    _prune_empty_overs(copy.pseudoRoot)
    return removed > 0 and copy.empty


def _layer_text_without_managed_cams(layer) -> str:
    """`layer.ExportToString()` with tool-authored cameras removed — the
    fingerprint text for dirty layers, so a moved render camera does not
    invalidate an otherwise identical bundle. The layer itself is never touched:
    stripping happens on an anonymous copy, and only when the serialized text
    actually mentions a managed-camera tag."""
    from pxr import Sdf

    text = layer.ExportToString()
    if not any(k in text for k in _AUTHORED_CAM_KEYS):
        return text
    copy = Sdf.Layer.CreateAnonymous(".usda")
    copy.TransferContent(layer)
    _strip_managed_cameras_spec(copy, extra_paths=set(_CAMERA_STRIP_PATHS.get()))
    return copy.ExportToString()


def _scene_stem(stage) -> str:
    """Short scene identifier for logs and errors (root-layer stem, or a placeholder)."""
    try:
        ident = stage.GetRootLayer().identifier or ""
    except Exception:  # noqa: BLE001 — logging context only, never fail a render on it
        return "<unknown>"
    # anonymous layer identifiers look like "anon:0x...:scene.usda"
    stem = Path(ident.rsplit(":", 1)[-1]).stem
    return stem or "<in-memory>"


def _stage_asset_attribute_paths(stage) -> list[str]:
    """Resolved asset-valued attribute paths authored/composed on the stage.

    Anonymous/in-memory root layers have no on-disk file for
    `UsdUtils.ComputeAllDependencies` to walk, so a tiny anonymous layer over a
    huge texture/payload estimated as (0, 0) and sailed past the fail-fast tier.
    Scanning asset-valued attributes directly is the cheap correct option there
    (no export of the whole root layer just to estimate). Capped at
    `_ESTIMATE_MAX_DEPS` entries; instance proxies are traversed so prototype
    contents count too.
    """
    from pxr import Sdf, Usd

    out: list[str] = []
    seen: set[str] = set()
    asset_types = (Sdf.ValueTypeNames.Asset, Sdf.ValueTypeNames.AssetArray)
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        for attr in prim.GetAttributes():
            if attr.GetTypeName() not in asset_types:
                continue
            value = attr.Get()
            if value is None:
                continue
            items = [value] if isinstance(value, Sdf.AssetPath) else list(value)
            for ap in items:
                p = (str(getattr(ap, "resolvedPath", "") or "")
                     or str(getattr(ap, "path", "") or ""))
                if p and p not in seen:
                    seen.add(p)
                    out.append(p)
                    if len(out) >= _ESTIMATE_MAX_DEPS:
                        return out
    return out


# id()s of stages whose last dependency walk hit the cap (see below): the bundle
# cache checks membership and skips caching for them. Weak-value semantics are not
# needed — entries are only consulted within the same render call.
_CAPPED_SENTINEL: set[int] = set()


class _CasUnsupportedError(RuntimeError):
    """The backend advertised the CAS feature but lacks its endpoints (404/405 on
    negotiate) — a deploy skew, not a render failure: fall back to the bundle
    upload instead of demoting the slot."""


def _retryable_render_error(exc: Exception) -> bool:
    """Only failures a DIFFERENT backend could plausibly serve justify a sibling
    retry (and the demotion that follows): transport errors, stalls, and HTTP 5xx.
    Deterministic client errors (HTTP 4xx — bad params, payload too large) and
    local packaging failures would fail identically everywhere and must not
    demote a healthy second backend."""
    try:
        import httpx
        if isinstance(exc, httpx.HTTPError):
            return True
    except ImportError:  # pragma: no cover
        pass
    text = str(exc)
    if "HTTP 4" in text:
        return False
    return ("HTTP 5" in text or "incomplete result set" in text
            or "timed out" in text or "stall" in text.lower())


def _walk_asset_dependencies(stage) -> list[str] | None:
    """Every resolved file dependency packaging would pull, beyond the used layers.

    On-disk roots: dependency layers + external assets (textures, engdata
    payloads) from `UsdUtils.ComputeAllDependencies`. Anonymous/in-memory roots:
    a direct asset-valued attribute scan (see _stage_asset_attribute_paths).
    Returns None when the walk itself fails — the estimator then falls back to
    the used-layer sum, and the bundle cache treats the stage as *uncacheable*
    (an unknown dependency set must never serve a maybe-stale bundle). Capped at
    `_ESTIMATE_MAX_DEPS` entries.
    """
    try:
        root_real = getattr(stage.GetRootLayer(), "realPath", "") or ""
        if not root_real:
            return _stage_asset_attribute_paths(stage)
        from pxr import UsdUtils

        with _StderrCapture() as capture:  # the walk re-warns per broken ref
            dep_layers, assets, _unresolved = UsdUtils.ComputeAllDependencies(
                root_real)
        logger.debug("dependency-walk warnings:\n%s", capture.text)
        out = [p for lyr in dep_layers if (p := getattr(lyr, "realPath", "") or "")]
        out.extend(str(asset) for asset in assets)
        if len(out) > _ESTIMATE_MAX_DEPS:
            # a TRUNCATED dependency list is fine for the size estimate but not
            # for the bundle cache: dependency 10,001 changing must not serve a
            # stale bundle — the caller treats an over-cap stage as uncacheable
            _CAPPED_SENTINEL.add(id(stage))
            return out[:_ESTIMATE_MAX_DEPS]
        return out
    except Exception:  # noqa: BLE001 — degraded estimate / uncacheable, never fatal
        logger.debug("asset-dependency walk failed", exc_info=True)
        return None


def _estimate_stage_bytes(stage, deps: list[str] | None = None) -> tuple[int, int]:
    """(estimated on-disk bytes of everything packaging would bundle, file count).

    Sums the stage's used-layer file sizes PLUS the resolved external asset
    dependencies from `_walk_asset_dependencies` — USDZ packaging pulls those
    assets too, and the layer-size sum alone undercounted a real 1,377 MB bundle
    as 103.8 MB, so the >2× fail-fast tier fired only *after* the ~2.5-minute
    packaging+gzip. `deps` accepts a precomputed walk result so callers who
    already ran it (the bundle-cache key shares the same walk) don't pay twice;
    with `deps=None` the walk runs here, and on walk failure the plain layer sum
    still stands. Still an estimate (gzip may later shrink the wire bytes), so it
    is only used to reject scenes that are hopeless *before* minutes of
    packaging — the exact check on the final bytes (`_check_limit`) still runs
    afterwards. Advisory: a stage that can't enumerate layers (or a test stub)
    estimates as (0, 0), never an error.
    """
    sizes: dict[str, int] = {}

    def add(path_str: str) -> None:
        if not path_str or path_str in sizes or len(sizes) >= _ESTIMATE_MAX_DEPS:
            return
        try:
            sizes[path_str] = Path(path_str).stat().st_size
        except OSError:
            return  # unresolved/package-internal path — packaging decides its fate

    try:
        layers = list(stage.GetUsedLayers(includeClipLayers=True))
    except Exception:  # noqa: BLE001 — estimate only; packaging is the real gate
        return 0, 0
    for layer in layers:
        add(getattr(layer, "realPath", "") or "")
    if deps is None:
        deps = _walk_asset_dependencies(stage)
    for dep in deps or []:
        add(dep)
    return sum(sizes.values()), len(sizes)


def _flag_blank_suspects(results: list[RenderResult]) -> list[str]:
    """Tag results whose image looks featureless (blank-frame sanity, Issue: renders
    that miss their subject produce a plausible-looking gradient the agent only catches
    by viewing). Sets `blank_suspect` on each result and logs one warning listing the
    suspect paths. Never fails a successful render — the heuristic is advisory.
    """
    from usd_core.imaging import blank_suspects

    try:
        suspects = blank_suspects([r.path for r in results])
    except Exception:  # noqa: BLE001 — advisory only; PIL hiccups must not kill a render
        logger.debug("blank-frame heuristic failed", exc_info=True)
        return []
    for r in results:
        r.blank_suspect = r.path in suspects
    if suspects:
        logger.warning(
            "render sanity: %d of %d image(s) look blank/featureless — the camera "
            "may have missed its subject. blank_suspect: %s",
            len(suspects), len(results), suspects)
    return suspects


@dataclass
class _BakePlan:
    """What packaging must rewrite so a prototype-resolving renderer matches pxr.

    `bakes` are (gprim_or_subset_path, material_path, material_purpose) direct
    bindings to author on the throwaway packaging copy (purpose "" = all-purpose).
    `neutralize_rels` are property paths of prototype-stored collection-binding
    relationships that would outrank those baked direct bindings once their subtree
    is de-instanced — they get blocked (explicit empty targets) in the copy.
    """

    bakes: list[tuple[str, str, str]] = field(default_factory=list)
    neutralize_rels: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.bakes)

    def __bool__(self) -> bool:
        return bool(self.bakes or self.neutralize_rels)


class _BackendSlot:
    """One remote service in the pool: its address plus scheduling/health state.

    `in_flight` is the number of requests currently running against it (least-busy
    scheduling); `healthy` flips False — once, with one warning — when the version
    handshake fails or the service is unreachable; `verified` caches a passed
    handshake so each backend is probed at most once per pool state. `probe_lock`
    single-flights the handshake and locks health transitions: concurrent commands
    sharing a slot must not race healthy/verified/reason writes or probe the same
    backend twice.

    Degradation tracking (module docstring, "Scheduling policy"): `recent_bps`
    holds the throughput (bytes/s) of the last `_THROUGHPUT_WINDOW` successful
    uploads; `recent_failure` is set on an upload/backend error or stall (with
    `failure_at` stamping when, for the `_FAILURE_COOLDOWN_S` expiry) and cleared
    by the next successful upload. All are read/written under the pool lock
    (see _PoolState.record_upload / record_failure).
    """

    def __init__(self, url: str, api_key: str | None = None):
        self.url = url.rstrip("/")
        self.api_key = api_key or None
        self.in_flight = 0
        self.healthy = True
        self.verified = False
        self.info: dict = {}  # the /live payload (max_body_bytes, features, ...)
        self.reason = ""  # why it was marked unhealthy (for the all-out error)
        self.probe_lock = threading.Lock()
        self.recent_bps: deque[float] = deque(maxlen=_THROUGHPUT_WINDOW)
        self.recent_failure = False
        self.failure_at = float("-inf")  # _monotonic() of the last recorded failure
        self.unhealthy_at = float("-inf")  # _monotonic() when healthy flipped False
        # observability: set while the slot is logged as demoted, cleared (with a
        # re-promotion line) when it leaves the demoted set — demotion decisions
        # were previously invisible in the daemon log (benchmark round 5)
        self.demote_logged = False

    @property
    def avg_bps(self) -> float:
        """Mean recent upload throughput; 0.0 = no samples yet (unknown, not slow)."""
        return sum(self.recent_bps) / len(self.recent_bps) if self.recent_bps else 0.0


class _PoolState:
    """Scheduler state for one normalized pool configuration, shared across every
    `RenderBackend` built from that configuration (see _POOL_REGISTRY).

    Health outlives the process: throughput samples and failure stamps are
    persisted (see _pool_state_path) and reloaded on the first construction after
    a daemon restart — restarting the daemon was the agent's natural response to
    a stalled render, and it silently amnestied the very backend that stalled.
    Persisted entries expire after `_POOL_STATE_TTL_S`.
    """

    def __init__(self, slots: list[_BackendSlot], persist_path: Path | None = None):
        self.slots = slots
        self.lock = threading.Lock()
        self.rr = 0  # round-robin cursor for least-busy ties
        # consecutive _acquire_slot picks that went to non-demoted slots; at
        # _CANARY_EVERY one pick is routed to the best demoted slot (recovery)
        self.non_demoted_streak = 0
        self.persist_path = persist_path
        self._persist_lock = threading.Lock()
        self._load_persisted()

    def record_upload(self, slot: _BackendSlot, nbytes: int, seconds: float) -> None:
        """A successful upload: add a throughput sample and clear the failure flag."""
        with self.lock:
            if nbytes > 0 and seconds > 0:
                slot.recent_bps.append(nbytes / seconds)
            slot.recent_failure = False
        self._save_persisted()

    def record_failure(self, slot: _BackendSlot) -> None:
        """An upload/backend error or stall: demote the slot for the cooldown
        window (a later success clears the flag; another failure re-stamps it)."""
        with self.lock:
            already = slot.recent_failure
            slot.recent_failure = True
            slot.failure_at = _monotonic()
        if not already:
            logger.warning(
                "remote render: backend %s marked degraded after a failed/stalled "
                "request — deprioritized until a successful upload or a %.0fs "
                "cooldown", slot.url, _FAILURE_COOLDOWN_S)
        self._save_persisted()

    def _load_persisted(self) -> None:
        """Apply persisted slot health, best-effort: a missing/expired/garbled
        file just means a cold start. Failure stamps are stored as wall-clock
        (monotonic does not survive the process) and mapped back on load."""
        if self.persist_path is None:
            return
        try:
            st = os.lstat(self.persist_path)
            import stat as _stat
            if not _stat.S_ISREG(st.st_mode):
                return  # a symlink/dir planted at the persistence path: ignore
            if hasattr(os, "getuid") and st.st_uid != os.getuid():
                return  # not our file: never trust foreign scheduling state
            raw = json.loads(self.persist_path.read_text())
            if raw.get("v") != _POOL_STATE_VERSION:
                return
            age = time.time() - float(raw.get("saved_at", 0))
            if not 0 <= age < _POOL_STATE_TTL_S:
                return
            by_url = raw.get("slots", {})
            now_wall, now_mono = time.time(), _monotonic()
            restored = []
            with self.lock:
                for slot in self.slots:
                    entry = by_url.get(slot.url)
                    if not isinstance(entry, dict):
                        continue
                    bps = [float(b) for b in entry.get("bps", [])
                           if isinstance(b, (int, float)) and b > 0]
                    slot.recent_bps.extend(bps[-_THROUGHPUT_WINDOW:])
                    fail_wall = entry.get("failure_wall")
                    if isinstance(fail_wall, (int, float)):
                        elapsed = now_wall - float(fail_wall)
                        if 0 <= elapsed < _FAILURE_COOLDOWN_S:
                            slot.recent_failure = True
                            slot.failure_at = now_mono - elapsed
                    if bps or slot.recent_failure:
                        restored.append(slot.url)
            if restored:
                logger.info(
                    "remote render: restored persisted pool health for %d backend(s) "
                    "(%.0fs old): %s", len(restored), age, ", ".join(restored))
        except Exception:  # noqa: BLE001 — persistence is advisory, never a gate
            logger.debug("remote render: could not load persisted pool state",
                         exc_info=True)

    def _save_persisted(self) -> None:
        """Atomically snapshot slot health next to the bundle cache; best-effort.

        Serialized by `_persist_lock` (concurrent shard completions raced one
        predictable temp file) and written through a UNIQUE O_EXCL 0600 temp so
        a planted symlink or a sibling process can neither redirect nor
        interleave the write; os.replace never follows the destination."""
        if self.persist_path is None:
            return
        try:
            with self._persist_lock:
                now_wall, now_mono = time.time(), _monotonic()
                with self.lock:
                    slots = {
                        s.url: {
                            "bps": list(s.recent_bps),
                            "failure_wall": (now_wall - (now_mono - s.failure_at)
                                             if s.recent_failure else None),
                        }
                        for s in self.slots
                    }
                payload = json.dumps(
                    {"v": _POOL_STATE_VERSION, "saved_at": now_wall, "slots": slots})
                tmp = self.persist_path.with_name(
                    f"{self.persist_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                try:
                    with os.fdopen(fd, "w") as fh:
                        fh.write(payload)
                    os.replace(tmp, self.persist_path)
                finally:
                    if os.path.exists(tmp):
                        os.unlink(tmp)
        except Exception:  # noqa: BLE001 — persistence is advisory, never a gate
            logger.debug("remote render: could not persist pool state", exc_info=True)


# The daemon constructs a fresh RemoteRenderBackend per command; without shared state
# every command reset round-robin/health/in_flight and all renders started at backend
# zero. Multi-backend pool state is therefore cached here, keyed by the normalized
# pool config ((url, api_key) pairs + the verify flag), so repeated constructions
# share slots. Single-entry pools skip the registry — they have no scheduling state
# worth persisting (health there raises directly, preserving legacy behavior).
_POOL_REGISTRY: dict[tuple, _PoolState] = {}
_POOL_REGISTRY_LOCK = threading.Lock()


def _pool_state_path(key: tuple) -> Path | None:
    """Persistence file for one pool config, in the private per-user staging root
    (same trust rules as the bundle cache); None disables persistence. The name
    never matches the `usd_cli_remote_render_*` reaper glob, and the key digest is
    one-way (config api keys never appear on disk)."""
    import hashlib

    if (os.environ.get("USD_CLI_POOL_STATE_DISABLE")
            or os.environ.get("OV_POOL_STATE_DISABLE")):
        return None  # tests: cross-run health restoration breaks scheduling isolation
    staging, private = _staging_root_info()
    if not private:
        return None
    digest = hashlib.sha256(repr(key).encode()).hexdigest()[:16]
    return staging / f"usd_cli_pool_state_{digest}.json"


def _pool_state_for(slots: list[_BackendSlot], verify_version: bool) -> _PoolState:
    if len(slots) <= 1:
        return _PoolState(slots)
    key = (tuple((s.url, s.api_key or "") for s in slots), bool(verify_version))
    with _POOL_REGISTRY_LOCK:
        state = _POOL_REGISTRY.get(key)
        if state is None:
            state = _PoolState(slots, persist_path=_pool_state_path(key))
            _POOL_REGISTRY[key] = state
        return state


def _shard_frames(frames: list[float], k: int) -> list[list[float]]:
    """Split `frames` into at most k contiguous, non-empty chunks preserving order.

    Concatenating the chunks reproduces `frames` exactly, so per-chunk results
    reassemble in frame order. Chunk sizes differ by at most one.
    """
    k = max(1, min(int(k), len(frames)))
    base, extra = divmod(len(frames), k)
    chunks, start = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        chunks.append(frames[start:start + size])
        start += size
    return chunks


def _staging_pid(dirname: str) -> int | None:
    """PID encoded in a staging dir name (usd_cli_remote_render_<pid>_<rand>), or None.

    Also parses the pre-rename `ov/dsc_remote_render_<pid>_<rand>` forms so the reaper
    judges 3dsc-era leftovers by owner liveness, not just age.
    """
    for prefix in (_STAGING_PREFIX, *_LEGACY_STAGING_PREFIXES):
        if dirname.startswith(prefix):
            pid = dirname[len(prefix):].split("_", 1)[0]
            return int(pid) if pid.isdigit() else None
    return None


def _pid_alive(pid: int) -> bool:
    """True when `pid` is verifiably a live process. The pid comes from an untrusted
    /tmp dir name, so unrepresentable values (0, or huge ints that overflow os.kill's
    C pid_t — any local user can create `ov_remote_render_<huge-int>_x`) read as
    "not alive" instead of raising and breaking every subsequent render."""
    if pid <= 0:
        return False  # 0/negative signal process groups, never a single live owner
    try:
        os.kill(pid, 0)
    except (OverflowError, ValueError):
        return False  # not a representable pid — cannot be a live process
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        return True  # exists but owned elsewhere — treat as alive
    return True


def _secure_staging_root(
        root: Path, *, fallback: Path | None) -> tuple[Path, bool]:
    """Create or reuse an owned POSIX 0700 directory, optionally falling back."""
    uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        try:
            root.mkdir(mode=0o700)
        except FileExistsError:
            pass
        st = root.lstat()
        if not stat_mod.S_ISDIR(st.st_mode):  # a symlink/file squatting the name
            if fallback is not None:
                return fallback, False
            raise RuntimeError(f"remote render staging root is not a directory: {root}")
        if uid is not None and st.st_uid != uid:
            if fallback is not None:
                return fallback, False  # someone else's dir — never stage inside it
            raise RuntimeError(
                f"remote render staging root is owned by uid {st.st_uid}, "
                f"not us ({uid}): {root}"
            )
        mode = stat_mod.S_IMODE(st.st_mode)
        access_mode = mode & 0o777
        if uid is not None and access_mode != 0o700:
            if fallback is not None:
                return fallback, False
            raise RuntimeError(
                "remote render staging root must already be a dedicated "
                f"0700 directory, not access mode {access_mode:#05o} "
                f"(full mode {mode:#06o}): {root}"
            )
    except OSError as exc:
        if fallback is not None:
            return fallback, False
        raise RuntimeError(
            f"cannot secure remote render staging root {root}: {exc}"
        ) from exc
    return root, True


def _staging_root_info() -> tuple[Path, bool]:
    """(durable root, private?) — private=False means shared temp fallback.

    Staging in the shared world-writable temp dir let any local user place
    booby-trapped `ov_remote_render_*` names in the reaper's path; new staging
    happens under `<tmp>/dsc3-<uid>` instead, owned by us and mode 0700. Falls back
    to the shared temp dir when the private root can't be created or is squatted
    (symlink, foreign owner) — callers that must not trust a shared directory
    (the bundle cache) check the `private` flag and disable themselves.

    This stable root owns pool-state data and, without an explicit work-root pin,
    the bundle cache. A launcher's pin redirects transient work and the cache so
    both outlive launcher-private TMPDIR cleanup; it does not redirect pool state.
    """
    base = Path(tempfile.gettempdir())
    uid = os.getuid() if hasattr(os, "getuid") else None
    root = base / (f"dsc3-{uid}" if uid is not None else "dsc3")
    return _secure_staging_root(root, fallback=base)


def _work_staging_root_info() -> tuple[Path, bool]:
    """Return the ephemeral per-render work root and whether it is private.

    A launcher can pin ``REMOTE_RENDER_STAGING_ROOT_ENV`` to a directory whose
    lifecycle outlives its process temp directory. The variable's presence is an
    explicit confinement contract, including an empty value, so invalid or unsafe
    values fail closed instead of falling back elsewhere.
    """
    configured = os.environ.get(REMOTE_RENDER_STAGING_ROOT_ENV)
    if configured is None:
        return _staging_root_info()
    root = Path(configured)
    if not root.is_absolute():
        raise RuntimeError(
            f"{REMOTE_RENDER_STAGING_ROOT_ENV} must be an absolute path"
        )
    return _secure_staging_root(root, fallback=None)


def _staging_root() -> Path:
    """Durable private per-user root for cache and pool state."""
    return _staging_root_info()[0]


def _work_staging_root() -> Path:
    """Ephemeral per-render bundle root (see _work_staging_root_info)."""
    return _work_staging_root_info()[0]


def _reap_stale_staging(base: Path | None = None) -> None:
    """Best-effort removal of orphaned staging dirs from cancelled/killed renders.

    The per-render try/finally cleans up every failure path *within* the process, but a
    SIGKILL mid-packaging (a cancelled render, a crashed daemon) leaves the whole staging
    dir behind — 2.5 GB of /tmp/ov_remote_render_* observed after one benchmark night.
    On each new render, sweep siblings whose owning process (PID baked into the dir
    name) is verifiably gone, or that are older than 24h with no parseable owner — a
    dir whose owner is provably alive is never deleted, whatever its age. With no
    explicit `base`, the ephemeral work root, durable private per-user root, and
    legacy shared temp dir (pre-existing leftovers) are swept. Never raises:
    reaping is hygiene, not a gate.
    """
    if base is not None:
        bases = [base]
    else:
        bases = []
        for resolve_root in (
                _work_staging_root,
                _staging_root,
                lambda: Path(tempfile.gettempdir())):
            try:
                bases.append(resolve_root())
            except (OSError, RuntimeError):
                # An invalid explicit work root should fail the render itself, not
                # this best-effort hygiene pass. Other resolvable roots are still
                # safe to sweep.
                continue
    now = time.time()
    for b in dict.fromkeys(bases):  # dedup, keep order
        try:
            # ov/dsc_remote_render_* are the pre-rename prefixes — leftovers from
            # older builds must still be reaped (same name format, PID slot).
            candidates = list(b.glob(f"{_STAGING_PREFIX}*"))
            for legacy in _LEGACY_STAGING_PREFIXES:
                candidates += list(b.glob(f"{legacy}*"))
        except OSError:
            continue
        for d in candidates:
            try:
                if not d.is_dir():
                    continue
                pid = _staging_pid(d.name)
                if pid == os.getpid():
                    continue  # our own in-flight render (concurrent thread)
                if pid is not None and _pid_alive(pid):
                    continue  # verifiably live owner — never delete, even past 24h
                dead_owner = pid is not None
                stale_age = now - d.stat().st_mtime > _STAGING_MAX_AGE_S
                if dead_owner or stale_age:
                    logger.info("remote render: reaping orphaned staging dir %s (%s)",
                                d, "dead owner" if dead_owner else "older than 24h")
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                continue


# ── packaged-bundle cache (module docstring, "Bundle cache") ──────────────────────


_CACHE_DISABLE_WARNED: set[str] = set()  # log each disable reason once per process


def _cache_disabled(reason: str) -> None:
    if reason not in _CACHE_DISABLE_WARNED:
        _CACHE_DISABLE_WARNED.add(reason)
        logger.warning("remote render: bundle cache disabled — %s", reason)


def _bundle_cache_root() -> Path | None:
    """Cache root inside the private launcher-pinned or per-user root (0700).

    A launcher-pinned work root must own the cache as well as transient packaging:
    a cached USDZ is the artifact subsequently opened by manifest generation and
    upload, so leaving it under the launcher's ``TMPDIR`` would preserve the same
    mid-render deletion race that pinning transient work is meant to remove. With
    no explicit pin, the durable per-user staging root remains the fallback.

    Returns None when no trustworthy root exists (the cache is then disabled for
    this render).

    Named so it never matches the `ov_remote_render_*` glob — the staging reaper
    must not sweep cache entries; the cache has its own LRU/age bound
    (see _reap_bundle_cache).

    Trust requirements: staging must be the *private* root (never the shared-tmp
    fallback — a squatted staging root must not move the cache into a directory
    other users control), and the cache dir itself must be a non-symlink directory
    owned by us with no group/other access. A looser-but-owned pre-existing dir is
    tightened to 0700; anything else disables caching (with one log line saying
    why) rather than trusting it."""
    try:
        staging, private = _work_staging_root_info()
    except RuntimeError as exc:
        # Cache lookup/eviction is optional hygiene. The mandatory work-root
        # creation still fails the render closed at its normal call site, but an
        # invalidated pin must not make _reap_bundle_cache violate its no-raise
        # contract first.
        _cache_disabled(f"the pinned remote-render root is invalid ({exc})")
        return None
    if not private:
        _cache_disabled(f"the private remote-render staging root is unavailable "
                        f"(fell back to shared {staging})")
        return None
    root = staging / _BUNDLE_CACHE_DIRNAME
    uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        root.mkdir(mode=0o700, exist_ok=True)
        st = root.lstat()
        if not stat_mod.S_ISDIR(st.st_mode):  # a symlink/file squatting the name
            _cache_disabled(f"{root} is not a real directory (symlink/file squat)")
            return None
        if uid is not None and st.st_uid != uid:
            _cache_disabled(f"{root} is owned by uid {st.st_uid}, not us ({uid})")
            return None
        if st.st_mode & 0o077:
            os.chmod(root, 0o700)  # tighten a pre-existing looser root
    except OSError as exc:
        _cache_disabled(f"cannot secure {root} ({exc})")
        return None
    return root


def _bundle_cache_entry_dir(key: str, scene: str) -> Path | None:
    """Cache entry directory for (fingerprint, scene stem), or None when the cache
    is disabled (no trustworthy root — see _bundle_cache_root). The stem prefix
    groups entries for the per-scene LRU bound; the key suffix actually matches."""
    root = _bundle_cache_root()
    if root is None:
        return None
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in scene)[:60]
    return root / f"{safe or 'scene'}--{key[:32]}"


def _trusted_cache_file(path: Path) -> bool:
    """True only for a cache artifact safe to serve: opened O_NOFOLLOW (a symlink
    final component fails the open, so the checked inode IS the served file), a
    regular file, and owned by us — a symlink or foreign file planted in the cache
    must never be uploaded on the victim's behalf or poison a render."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(str(path), flags)
    except OSError:
        return False
    try:
        st = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat_mod.S_ISREG(st.st_mode):
        return False
    uid = os.getuid() if hasattr(os, "getuid") else None
    return uid is None or st.st_uid == uid


def _bundle_cache_get(entry: Path) -> tuple[Path, Path, str] | None:
    """(usdz_path, file_to_send, compression) for a cache hit, or None.

    A gzip sibling in the entry means compression helped when the bundle was first
    packaged; its absence (with compression enabled — part of the cache key) means
    it did not. The entry's mtime is bumped so LRU eviction sees the reuse.
    The entry directory must be a non-symlink directory owned by us, and each
    artifact must pass _trusted_cache_file; a planted gzip sibling that fails the
    check poisons the whole entry (no silent fallback to the raw usdz)."""
    usdz = entry / "scene_bundle.usdz"
    uid = os.getuid() if hasattr(os, "getuid") else None
    try:
        st = entry.lstat()
        if not stat_mod.S_ISDIR(st.st_mode):
            return None
        if uid is not None and st.st_uid != uid:
            return None
        if not _trusted_cache_file(usdz):
            return None
        now = time.time()
        os.utime(entry, (now, now))
        # sibling format is whatever the fleet supported at packaging time; a
        # gzip sibling on a now-zstd fleet is still perfectly acceptable wire
        for name, fmt in (("scene_bundle.usdz.zst", "zstd"),
                          ("scene_bundle.usdz.gz", "gzip")):
            sib = entry / name
            try:
                os.lstat(sib)  # present (even as a symlink)? then it must verify
            except OSError:
                continue
            if not _trusted_cache_file(sib):
                return None  # tampered sibling — distrust the whole entry
            return usdz, sib, fmt
        return usdz, usdz, "none"
    except OSError:
        return None


def _link_or_copy_cache_artifact(source: Path, destination: Path) -> None:
    """Create a cheap independent path to a cache artifact, copying as fallback."""

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _materialize_bundle_cache_hit(
    hit: tuple[Path, Path, str], work_dir: Path
) -> tuple[Path, Path, str] | None:
    """Pin a cache hit under this render's work dir for the whole upload.

    Cache eviction is cross-process and may unlink an entry while another daemon
    negotiates or uploads it. A hard link (or copy when links are unavailable)
    gives the in-flight render its own lifetime without holding a global cache
    lock. Best-effort: failure simply turns the hit into a normal repackage.
    """

    cached_usdz, cached_send, compression = hit
    created: list[Path] = []
    try:
        work_dir.mkdir(parents=True, exist_ok=True)
        usdz_path = work_dir / cached_usdz.name
        _link_or_copy_cache_artifact(cached_usdz, usdz_path)
        created.append(usdz_path)
        if cached_send == cached_usdz:
            send_path = usdz_path
        else:
            send_path = work_dir / cached_send.name
            _link_or_copy_cache_artifact(cached_send, send_path)
            created.append(send_path)
        return usdz_path, send_path, compression
    except OSError:
        logger.debug("bundle cache: could not pin cache hit", exc_info=True)
        for path in created:
            try:
                path.unlink()
            except OSError:
                pass
        return None


def _bundle_cache_put(entry: Path, usdz_path: Path,
                      send_path: Path) -> tuple[Path, Path] | None:
    """Publish freshly packaged artifacts into the cache; returns their new
    (usdz_path, send_path) or None (caller keeps its work_dir paths).

    The entry is built in a temp sibling and renamed into place so readers never
    see a half-copied bundle; when a concurrent packaging of the same content wins
    the rename race, its (identical) entry is reused. Best-effort throughout —
    caching failures must never fail a render."""
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        existing = _bundle_cache_get(entry)
        if existing is None:
            tmp = entry.with_name(f"{entry.name}.tmp{os.getpid()}")
            tmp.mkdir()
            sibling = send_path if send_path != usdz_path else None
            _link_or_copy_cache_artifact(usdz_path, tmp / "scene_bundle.usdz")
            if sibling is not None:
                # keep the compression-specific name (.usdz.gz / .usdz.zst)
                _link_or_copy_cache_artifact(sibling, tmp / sibling.name)
            try:
                tmp.rename(entry)
            except OSError:  # a concurrent put won the race — use its entry
                shutil.rmtree(tmp, ignore_errors=True)
            existing = _bundle_cache_get(entry)
        if existing is None:
            return None
        cached_usdz, cached_send, _compression = existing
        _reap_bundle_cache(keep_entry=entry)
        return cached_usdz, cached_send
    except OSError:
        logger.debug("bundle cache: could not publish %s", entry, exc_info=True)
        return None


def _reap_bundle_cache(keep_entry: Path | None = None) -> None:
    """Bound the cache: newest `_BUNDLE_CACHE_KEEP` entries per scene stem (LRU by
    dir mtime), nothing older than `_STAGING_MAX_AGE_S`, crashed-put temp dirs
    age-gated the same way. `keep_entry` (the entry just used) always survives.
    Never raises: eviction is hygiene, not a gate."""
    root = _bundle_cache_root()
    if root is None:
        return  # cache disabled (no trustworthy root) — nothing of ours to bound
    now = time.time()
    groups: dict[str, list[tuple[float, Path]]] = {}
    try:
        candidates = list(root.iterdir())
    except OSError:
        return
    for d in candidates:
        try:
            if not d.is_dir() or d == keep_entry:
                continue
            mtime = d.stat().st_mtime
            if ".tmp" in d.name or now - mtime > _STAGING_MAX_AGE_S:
                if now - mtime > _STAGING_MAX_AGE_S:
                    shutil.rmtree(d, ignore_errors=True)
                continue  # fresh temp dirs belong to an in-flight put
            groups.setdefault(d.name.rsplit("--", 1)[0], []).append((mtime, d))
        except OSError:
            continue
    for entries in groups.values():
        entries.sort(reverse=True)  # newest first
        keep = _BUNDLE_CACHE_KEEP - (1 if keep_entry is not None else 0)
        for _mtime, d in entries[max(0, keep):]:
            logger.info("remote render: evicting cached bundle %s (LRU bound)", d.name)
            shutil.rmtree(d, ignore_errors=True)


# Swapping fd 2 is process-wide state: named sessions can package concurrently, and
# two overlapping captures restoring fd 2 out of order leave the daemon's stderr
# pointed at an unlinked temp file forever. One lock held for the entire capture
# window serializes the (already expensive) capture+package sections across threads.
_STDERR_SWAP_LOCK = threading.Lock()


class _StderrCapture:
    """Capture the process-level stderr flood C++ USD emits during packaging.

    usdUtils asset localization TF_WARNs once per unresolved reference — thousands of
    lines on a big composed scene (benchmark task-14), drowning the daemon log with no
    progress signal. Those warnings go straight to fd 2 from C++ (not through Python
    logging), so the only way to summarize them is to swap the fd for the duration of
    the packaging call and re-emit a digest afterwards (see _log_packaging_warnings).
    Process-wide, so the whole capture window holds _STDERR_SWAP_LOCK — concurrent
    sessions serialize here rather than corrupt fd 2. No-op (but still serialized)
    when fd 2 can't be duplicated.
    """

    def __init__(self) -> None:
        self.text = ""
        self._saved: int | None = None
        self._tmp = None

    def __enter__(self) -> "_StderrCapture":
        import tempfile

        _STDERR_SWAP_LOCK.acquire()
        try:
            self._tmp = tempfile.TemporaryFile()
            sys.stderr.flush()
            self._saved = os.dup(2)
            os.dup2(self._tmp.fileno(), 2)
        except (OSError, ValueError):  # no real fd 2 (embedded interpreter) — skip
            if self._tmp is not None:
                self._tmp.close()
                self._tmp = None
            self._saved = None
        return self

    def __exit__(self, *exc) -> None:
        try:
            if self._saved is None:
                return
            try:
                sys.stderr.flush()
                os.dup2(self._saved, 2)
                os.close(self._saved)
                self._tmp.seek(0)
                self.text = self._tmp.read().decode("utf-8", "replace")
            finally:
                self._tmp.close()
        finally:
            _STDERR_SWAP_LOCK.release()


def _log_packaging_warnings(captured: str) -> None:
    """One summary line for a captured warning flood; the full list only at DEBUG.

    "Failed to resolve reference @N/…@" lines are counted separately: usdUtils
    asset localization emits them for its OWN package-internal remaps (numbered
    archive dirs like `0/tex.png` it just created inside the USDZ) — the assets ARE
    bundled, so reporting them as unresolved references sent agents chasing
    phantom missing textures (benchmark task-05's `@0/t_rubber…@` flood).
    """
    lines = [ln for ln in captured.splitlines() if ln.strip()]
    if not lines:
        return
    unresolved: list[str] = []
    remap_noise: list[str] = []
    for ln in lines:
        if "resolve" not in ln.lower():
            continue
        (remap_noise if _PKG_INTERNAL_REMAP_RE.search(ln) else unresolved).append(ln)
    first = (unresolved or remap_noise or lines)[0]
    logger.warning(
        "remote render packaging: suppressed %d USD warning line(s) (%d look like "
        "unresolved asset references; %d are usdUtils package-internal remap noise "
        "(@N/…@) — those assets ARE in the bundle). First: %s "
        "[full list at DEBUG level]",
        len(lines), len(unresolved), len(remap_noise), first[:200])
    logger.debug("remote render packaging warnings:\n%s", captured)


class _ProgressReader:
    """File-like wrapper that logs upload progress as httpx streams the body.

    Also times the streaming window (first read → EOF) so the pool scheduler can
    record per-slot upload throughput; `upload_seconds` is 0.0 when the body was
    never streamed (callers fall back to overall request time)."""

    def __init__(self, path: Path, label: str):
        self._fh = path.open("rb")
        self._total = path.stat().st_size
        self._sent = 0
        self._next_pct = 50  # 50/100 only — 25% steps were per-upload log spam
        self._label = label
        self._started: float | None = None
        self._finished: float | None = None

    def read(self, n: int = -1) -> bytes:
        if self._started is None:
            self._started = time.perf_counter()
        chunk = self._fh.read(n)
        if not chunk and self._finished is None:
            self._finished = time.perf_counter()
        self._sent += len(chunk)
        if self._total and self._sent * 100 >= self._next_pct * self._total:
            logger.info("remote render upload (%s): %d%% of %.1f MB",
                        self._label, min(100, self._sent * 100 // self._total),
                        self._total / _MB)
            self._next_pct += 50
        # Crawling-but-alive transfers never trip the write-stall timeout: abort
        # when the ROLLING average falls under the floor after a grace window —
        # the raised error is retryable, so the render fails over to a sibling
        # and the crawl demotes this slot (round 7: 18 MB in ~6 min at ~50 KB/s).
        if chunk and self._started is not None:
            elapsed = time.perf_counter() - self._started
            if (elapsed > _UPLOAD_MIN_BPS_WINDOW_S
                    and self._sent / elapsed < _UPLOAD_MIN_BPS
                    and self._sent < self._total):
                raise RuntimeError(
                    f"remote render upload stall: {self._label} at "
                    f"{self._sent / elapsed / 1024:.0f} KB/s after {elapsed:.0f}s "
                    f"({self._sent}/{self._total} bytes) — aborting to fail over")
        return chunk

    @property
    def upload_seconds(self) -> float:
        if self._started is None or self._finished is None:
            return 0.0
        return max(0.0, self._finished - self._started)

    def close(self) -> None:
        self._fh.close()


class RemoteRenderBackend:
    name = "remote"

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 300.0,
                 compress: bool = True, max_upload_mb: float = 0,
                 verify_version: bool = True, backends: list[dict] | None = None,
                 bundle_cache: bool = True, cas: bool = True):
        # `backends` (from [[render.backends]], via resolve_render_backends) is the
        # pool; without it the legacy base_url/api_key pair is a one-entry pool.
        # pool[0] is the primary and backs the legacy single-backend attributes.
        specs = [b for b in (backends or []) if str(b.get("url", "") or "").strip()]
        if not specs:
            if not base_url:
                raise ValueError("renderer = 'remote' requires render.remote_url in config")
            specs = [{"url": base_url, "api_key": api_key or ""}]
        slots = [_BackendSlot(str(s["url"]), str(s.get("api_key", "") or "") or None)
                 for s in specs]
        # multi-entry pools share scheduler state (slots/health/round-robin) across
        # constructions — the daemon builds a new backend per command, and per-command
        # state meant every render started at backend zero with reset health
        self._pool_state = _pool_state_for(slots, verify_version)
        self._pool = self._pool_state.slots
        self._pool_lock = self._pool_state.lock
        self._base_url = self._pool[0].url
        self._api_key = self._pool[0].api_key
        self._timeout = timeout
        self._compress = compress
        self._max_upload_mb = float(max_upload_mb or 0)  # 0 = unlimited
        self._verify_version = verify_version
        self._protocol_verified = False
        self._backend_info: dict = {}  # primary's /live payload (see _ensure_protocol)
        self._bundle_cache = bool(bundle_cache)
        # content-addressed manifest transport (render.remote_cas; plan Phase 2) —
        # used per-slot when the backend advertises the "cas" feature on /live
        self._cas = bool(cas)
        # upload-outcome hooks: no-ops on a standalone client; `_child` wires them to
        # the owning pool slot so the scheduler learns throughput/failures (they fire
        # only around the actual backend interaction in `_dispatch`, so a local
        # packaging error can never demote a slot)
        self._note_upload: Callable[[int, float], None] = lambda nbytes, seconds: None
        self._note_failure: Callable[[], None] = lambda: None
        self._last_upload_seconds = 0.0  # streamed-body time of the last multipart POST
        self._last_legacy_bytes = 0  # body size of the last base64-JSON legacy POST

    @staticmethod
    def _enclosing_prototype_root(prim):
        """Root path of the innermost prototype holding this prim's real spec, or None.

        An instance proxy under nested instancing maps through one prototype per
        level; following GetPrimInPrototype until the prim is real lands in the
        prototype a scene-delegate renderer actually resolves for it. Proxy-ness of a
        binding's owner alone is not enough: a binding on a nested instance root is an
        outer-prototype proxy yet still lies outside the inner mesh's prototype, so
        the renderer drops it — only comparing prototype roots catches that."""
        p = prim
        while p and p.IsInstanceProxy():
            nxt = p.GetPrimInPrototype()
            if not nxt or nxt == p:
                break
            p = nxt
        if not p or not p.IsInPrototype():
            return None
        return p.GetPath().GetPrefixes()[0]

    @classmethod
    def _collect_instance_binding_bakes(cls, stage) -> "_BakePlan":
        """The rewrites packaging must apply so a prototype-resolving renderer agrees
        with pxr composition.

        OVRTX (and Fabric-style scene delegates generally) resolve material bindings per
        *prototype*: opinions authored outside an instance's prototype — a binding
        override on the instance root (what usd-cli's own `material bind` authors for
        instanced prims) or one inherited from an ancestor — can be dropped or unified
        across instances, so those parts render unbound-gray even though pxr composes
        them correctly (benchmark task-03: per-instance Component_Blue and
        ancestor-inherited PCB_Default_Dark lost on a 1,285-instance PCB).

        Collects, for **every material purpose** (all-purpose, preview, full), every
        gprim/GeomSubset inside (or at the root of) an instance whose winning binding
        spec lives outside the prim's own (innermost) prototype, with its composed
        bound material. Prototype-stored collection-binding relationships along a
        baked prim's ancestor chain are marked for neutralization — once the subtree
        is de-instanced they become real opinions that can outrank the baked direct
        binding — and any prim whose winning binding *was* such a collection rel is
        baked too, so blocking the rel cannot strip its material. Read-only.
        """
        from pxr import Usd, UsdGeom, UsdShade

        purposes = (UsdShade.Tokens.allPurpose, UsdShade.Tokens.preview,
                    UsdShade.Tokens.full)
        baked: dict[tuple[str, str], str] = {}  # (gpath, purpose) -> material path
        neutralize: set[str] = set()  # collection-binding rel property paths to block
        harvested: set[str] = set()  # prim paths whose ancestor chain was scanned
        # (prim, purpose, material, rel-path) whose winner lives in the prim's own
        # prototype — renderer-safe unless that rel ends up neutralized (see below)
        proto_safe: list[tuple[object, str, str, str]] = []
        queue: list[tuple[object, str, str]] = []  # (prim, purpose, material) to bake

        def harvest_collection_rels(prim) -> None:
            # prototype-stored collection bindings anywhere on `prim`'s ancestor chain
            # (incl. composed onto the chain's instance roots) become real, competing
            # opinions once the chain is de-instanced for a bake — collect them
            key = str(prim.GetPath())
            if key in harvested:
                return
            harvested.add(key)
            p = prim
            while p and not p.IsPseudoRoot():
                if p.IsInstanceProxy() or p.IsInstance():
                    for rel in p.GetRelationships():
                        if rel.GetName().startswith("material:binding:collection"):
                            neutralize.add(str(rel.GetPath()))
                p = p.GetParent()

        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            if not (prim.IsInstanceProxy() or prim.IsInstance()):
                continue
            if not (prim.IsA(UsdGeom.Gprim) or prim.IsA(UsdGeom.Subset)):
                continue
            geom_root = cls._enclosing_prototype_root(prim)
            api = UsdShade.MaterialBindingAPI(prim)
            winners: dict = {}
            for purpose in purposes:
                mat, rel = api.ComputeBoundMaterial(materialPurpose=purpose)
                if mat and rel:
                    winners[purpose] = (str(mat.GetPath()), rel)
            all_win = winners.get(UsdShade.Tokens.allPurpose)
            for purpose, (mpath, rel) in winners.items():
                if (purpose != UsdShade.Tokens.allPurpose and all_win is not None
                        and mpath == all_win[0]
                        and rel.GetPath() == all_win[1].GetPath()):
                    continue  # the all-purpose fallback re-reported; one bake covers it
                owner_root = cls._enclosing_prototype_root(rel.GetPrim())
                if owner_root is not None and owner_root == geom_root:
                    proto_safe.append((prim, str(purpose), mpath, str(rel.GetPath())))
                    continue  # lives inside the prim's own prototype — renderers see it
                queue.append((prim, str(purpose), mpath))

        # fixpoint: baking a prim neutralizes prototype-stored collection rels on its
        # chain; a prim whose winning binding WAS such a rel loses it in the packaging
        # copy, so it must be baked (with its composed material) as well
        while queue:
            prim, purpose, mpath = queue.pop()
            key = (str(prim.GetPath()), purpose)
            if key in baked:
                continue
            baked[key] = mpath
            harvest_collection_rels(prim)
            still: list[tuple[object, str, str, str]] = []
            for entry in proto_safe:
                p2, purpose2, mpath2, rel_path2 = entry
                if rel_path2 in neutralize:
                    queue.append((p2, purpose2, mpath2))
                else:
                    still.append(entry)
            proto_safe = still

        return _BakePlan(
            bakes=[(gpath, mpath, purpose) for (gpath, purpose), mpath in baked.items()],
            neutralize_rels=sorted(neutralize))

    @staticmethod
    def _deinstance_until_real(stage, path):
        """The prim at `path`, de-instancing its enclosing instance roots (outermost
        first) until it is a real (editable) prim. Nested instances open one level per
        pass; the loop is bounded by progress — every pass must de-instance a root it
        has not seen before (a fixed level cap left valid deeper nesting unbaked)."""
        prim = stage.GetPrimAtPath(path)
        opened: set = set()
        while prim and prim.IsInstanceProxy():
            anc = prim.GetParent()
            while anc and anc.IsInstanceProxy():
                anc = anc.GetParent()
            if not anc or not anc.IsInstance():
                return prim
            root_path = anc.GetPath()
            if root_path in opened:
                raise RuntimeError(
                    f"de-instancing made no progress at {root_path} while making "
                    f"{path} editable in the packaging copy")
            opened.add(root_path)
            anc.SetInstanceable(False)
            prim = stage.GetPrimAtPath(path)
        return prim

    @classmethod
    def _bake_instance_bindings(cls, stage, plan: "_BakePlan") -> int:
        """Author the collected bindings directly on the gprims of a *throwaway* stage.

        De-instances each affected instance root (only those — untouched instances keep
        their prototype sharing) so the proxy paths become editable, then binds the
        composed material directly on every planned gprim, preserving each binding's
        material purpose. Competing prototype-local collection-binding relationships
        are blocked (explicit empty targets) so they cannot outrank the baked direct
        bindings — the packaging copy is throwaway, so deleting opinions there is fine.
        The result no longer depends on instance-external binding resolution, which any
        renderer honors. Must never run on the caller's live stage — only on the
        exported/flattened packaging copy.
        """
        from pxr import Sdf, UsdShade

        baked = 0
        for gpath, mpath, purpose in plan.bakes:
            prim = cls._deinstance_until_real(stage, gpath)
            # a material inside an instance (the redirect pattern binds an instance root
            # to its own prototype's Looks) becomes a real, targetable prim the same way
            mat_prim = cls._deinstance_until_real(stage, mpath)
            if (not prim or prim.IsInstanceProxy()
                    or not mat_prim or mat_prim.IsInstanceProxy()):
                logger.warning("remote render: could not bake material binding for %s "
                               "(material %s) into the packaged scene", gpath, mpath)
                continue
            if prim.IsInstance():
                prim.SetInstanceable(False)  # a gprim that is itself an instance root
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material(mat_prim), materialPurpose=purpose)
            baked += 1
        for rel_path in plan.neutralize_rels:
            path = Sdf.Path(rel_path)
            prim = cls._deinstance_until_real(stage, path.GetPrimPath())
            if not prim or prim.IsInstanceProxy():
                continue  # its bake already warned; don't fail packaging on hygiene
            rel = prim.GetRelationship(path.name)
            if rel:
                rel.SetTargets([])  # explicit-empty blocks the referenced opinion
        return baked

    @classmethod
    def _strip_cameras_from_prepared(cls, prepared, is_temp: bool) -> None:
        """Remove tool-authored render cameras from the prepared (exported) copy
        before it is zipped: their specs travel as `camera_defs` in the request
        params (protocol v2) and the service re-authors them, so the bundle —
        and its cache fingerprint — is viewpoint-independent. Only the throwaway
        export is ever edited; the original asset (a USDZ handed through as-is,
        is_temp=False with a .usdz suffix) is immutable and left alone — its
        managed cameras exist only in memory and were never in the file anyway.
        Best-effort: a strip failure just means a camera-fat bundle (correct,
        cache-unfriendly), never a failed render.
        """
        from pxr import Sdf

        p = Path(prepared)
        if p.suffix.lower() == ".usdz" and not is_temp:
            return
        try:
            layer = Sdf.Layer.FindOrOpen(str(p))
            if layer and _strip_managed_cameras_spec(
                    layer, extra_paths=set(_CAMERA_STRIP_PATHS.get())):
                layer.Save()
        except Exception:  # noqa: BLE001 — advisory optimization, never a gate
            logger.debug("remote render: could not strip managed cameras from %s",
                         p, exc_info=True)

    @classmethod
    def _package_flattened(cls, stage, work_dir: Path, usdz_path: Path) -> None:
        """Fallback packaging: flatten, drop unresolvable asset refs, then zip.

        `CreateNewUsdzPackage` fails hard when *any* dependency is missing on disk — even a
        texture of a material the scene never binds (a stray broken path in a shared
        material library killed every remote render of scenes referencing it). Flattening
        pulls all composition into one throwaway layer we are allowed to rewrite, so the
        broken paths can be stripped and everything that does exist still ships.

        Flattening preserves instancing, so instance-external bindings are re-baked
        onto the flattened copy exactly as on the normal path (see
        _collect_instance_binding_bakes).
        """
        from pxr import Sdf, Usd, UsdUtils

        flat = work_dir / "_flattened.usdc"
        dropped: list[str] = []
        plan = cls._collect_instance_binding_bakes(stage)
        baked = 0
        capture = _StderrCapture()
        try:
            with capture:  # Flatten + the dependency walks re-warn per broken reference
                stage.Flatten().Export(str(flat))
                layer = Sdf.Layer.FindOrOpen(str(flat))
                _strip_managed_cameras_spec(  # travel as camera_defs instead
                    layer, extra_paths=set(_CAMERA_STRIP_PATHS.get()))
                if plan:
                    baked = cls._bake_instance_bindings(Usd.Stage.Open(layer), plan)
                anchor = Path(flat).parent

                def _keep(_layer_path: str, asset_path: str) -> str:
                    if "://" in asset_path or "[" in asset_path:  # URI / usdz-internal
                        # `pkg.usdz[textures/t.png]`-style paths resolve inside the
                        # package via the resolver, never on the plain filesystem
                        return asset_path
                    p = Path(asset_path)
                    resolved = p if p.is_absolute() else anchor / p
                    if not resolved.exists():
                        dropped.append(asset_path)
                        return ""
                    return asset_path

                UsdUtils.ModifyAssetPaths(layer, lambda ap: _keep(str(flat), ap))
                layer.Save()
                packaged = UsdUtils.CreateNewUsdzPackage(str(flat), str(usdz_path))
        finally:
            _log_packaging_warnings(capture.text)
        if plan:
            logger.info("remote render: baked %d of %d instance-external material "
                        "binding(s) into the flattened scene", baked, len(plan))
        if dropped:
            logger.warning("remote render: dropped %d unresolvable asset reference(s) "
                           "during packaging (first: %s)", len(dropped), dropped[0])
        if not packaged:
            raise RuntimeError("failed to package the scene as a USDZ bundle "
                               "(flattened fallback)")

    @classmethod
    def _package_usdz(cls, stage, work_dir: Path) -> Path:
        """Zip the scene (geometry + textures) into a USDZ on disk; return its path.

        `prepare_render_input` writes a renderer-openable USD — carrying in-memory edits like
        the managed camera — next to the original asset where its relative payload/texture
        paths still resolve; `CreateNewUsdzPackage` then walks those dependencies and bundles
        them. For a purely in-memory stage (no on-disk original), textures referenced by
        relative path may still not resolve — the geometry renders regardless.

        When packaging fails because a dependency is missing on disk, retry from a
        flattened copy with the broken references stripped (see _package_flattened).

        Before zipping, material bindings that only compose through instance-external
        opinions (instance-root overrides, ancestor-inherited bindings) are baked as
        direct bindings into the exported copy — prototype-resolving renderers such as
        OVRTX drop them otherwise (see _collect_instance_binding_bakes).
        """
        from pxr import Usd, UsdUtils

        plan = cls._collect_instance_binding_bakes(stage)
        root_real = getattr(stage.GetRootLayer(), "realPath", "") or ""
        usdz_path = work_dir / "scene_bundle.usdz"
        if (
            not plan
            and root_real.lower().endswith(".usdz")
            and _session_contains_only_managed_cameras(stage)
        ):
            # The package already contains its dependency closure. Running
            # CreateNewUsdzPackage on it creates a USDZ whose only member is the
            # original USDZ; OVRTX cannot reliably resolve that nested package
            # or its image assets. Managed cameras are transmitted separately,
            # so forwarding the exact original bytes preserves both geometry
            # and textures without dropping any semantic session opinion.
            source_package = Path(root_real)
            if source_package.resolve() != usdz_path.resolve():
                shutil.copyfile(source_package, usdz_path)
            logger.info(
                "remote render: forwarding self-contained USDZ without nesting "
                "(%.1f MB)",
                usdz_path.stat().st_size / _MB,
            )
            return usdz_path
        prepared, prepared_is_temp = prepare_render_input(stage, work_dir)
        logger.info("remote render: scene exported for packaging (%s, %.1f MB)",
                    Path(prepared).name, Path(prepared).stat().st_size / _MB)
        if plan and Path(prepared).suffix.lower() == ".usdz":
            # A USDZ root layer is immutable, so the bake cannot happen in place.
            # Route the scene through the flattened-copy path (writable) and bake
            # there; if even that fails, FAIL the render — knowingly shipping wrong
            # per-instance materials read as a "successful" render before.
            logger.info(
                "remote render: %d instance-external material binding(s) on a "
                "USDZ-rooted stage — packaging from a flattened copy to bake them",
                len(plan))
            if prepared_is_temp and Path(prepared) != usdz_path:
                Path(prepared).unlink(missing_ok=True)
            try:
                cls._package_flattened(stage, work_dir, usdz_path)
            except Exception as exc:
                raise RuntimeError(
                    f"remote render: the scene needs {len(plan)} instance-external "
                    "material binding(s) baked for the renderer, but its root layer "
                    "is an immutable USDZ and packaging a flattened copy failed "
                    f"({exc}) — refusing to render with knowingly wrong materials"
                ) from exc
            logger.info("remote render: USDZ bundle ready (%.1f MB)",
                        usdz_path.stat().st_size / _MB)
            return usdz_path
        if plan:
            baked = 0
            capture = _StderrCapture()
            try:
                with capture:  # composing the exported copy re-warns per broken ref
                    tmp_stage = Usd.Stage.Open(str(prepared))
                    baked = cls._bake_instance_bindings(tmp_stage, plan)
                    if baked or plan.neutralize_rels:
                        tmp_stage.GetRootLayer().Save()
                    del tmp_stage
            finally:
                _log_packaging_warnings(capture.text)
            logger.info("remote render: baked %d of %d instance-external material "
                        "binding(s) into the packaged scene", baked, len(plan))
        cls._strip_cameras_from_prepared(prepared, prepared_is_temp)
        try:
            capture = _StderrCapture()
            try:
                with capture:  # the dependency walk warns once per unresolved reference
                    packaged = UsdUtils.CreateNewUsdzPackage(str(prepared), str(usdz_path))
            finally:
                _log_packaging_warnings(capture.text)
            if not packaged:
                raise RuntimeError("failed to package the scene as a USDZ bundle")
        except Exception as exc:  # noqa: BLE001 — missing deps: strip and retry
            logger.warning("USDZ packaging failed (%s) — retrying with a flattened copy, "
                           "dropping unresolvable references", exc)
            usdz_path.unlink(missing_ok=True)
            cls._package_flattened(stage, work_dir, usdz_path)
        finally:
            # Only remove the exported temp copy. `prepared` can also be the *original*
            # asset (a USDZ-rooted stage: is_temp=False) — deleting that loses user data;
            # the in-memory-flatten case lands inside work_dir, which _execute removes.
            if prepared_is_temp and Path(prepared) != usdz_path:
                Path(prepared).unlink(missing_ok=True)
        logger.info("remote render: USDZ bundle ready (%.1f MB)",
                    usdz_path.stat().st_size / _MB)
        return usdz_path

    def _known_backend_infos(self) -> list[dict]:
        """Every /live payload this client has seen (verified pool slots + the
        primary's handshake result), non-empty entries only."""
        infos = [s.info for s in self._pool if s.verified and isinstance(s.info, dict)]
        if isinstance(self._backend_info, dict) and self._backend_info:
            infos.append(self._backend_info)
        return [i for i in infos if i]

    def _pool_supports_feature(self, feature: str) -> bool:
        """True when every backend we have /live info for advertises `feature` (and
        we have info for at least one). Unknown fleet composition = feature off."""
        infos = self._known_backend_infos()
        return bool(infos) and all(feature in (i.get("features") or [])
                                   for i in infos)

    def _pool_supports_zstd(self) -> bool:
        return self._pool_supports_feature("zstd")

    def _slot_supports_cas(self) -> bool:
        """True when THIS client's bound backend advertises the CAS manifest
        transport (per-slot decision — a mixed fleet uses CAS where available)."""
        return (self._cas and isinstance(self._backend_info, dict)
                and "cas" in (self._backend_info.get("features") or []))

    def _fleet_max_scene_bytes(self) -> int | None:
        """Smallest advertised `max_scene_bytes` across the known fleet (the CAS
        staged-scene ceiling), or None when nothing advertises it."""
        vals = [i.get("max_scene_bytes") for i in self._known_backend_infos()]
        vals = [int(v) for v in vals if isinstance(v, (int, float)) and v > 0]
        return min(vals) if vals else None

    def _prepare_upload(self, usdz_path: Path) -> tuple[Path, str]:
        """(file_to_send, compression): compress the bundle when that actually
        shrinks it — zstd when both ends support it (plan Phase 0: faster and
        ~20-30% smaller than gzip on USD), gzip otherwise.

        Uniform policy, decided here and nowhere else: compress whenever it saves
        at least 5%, raw otherwise — and the decision is LOGGED either way, so a
        raw upload in a trace is visibly a policy outcome (incompressible bundle,
        or compression disabled by config), never a silently skipped step.
        """
        raw = usdz_path.stat().st_size
        logger.info("remote render payload: %.1f MB USDZ", raw / _MB)
        if not self._compress:
            logger.info("remote render payload: compression disabled by config "
                        "(render.remote_compress = false) — uploading the raw USDZ")
            return usdz_path, "none"
        zs = _zstd_module()
        fmt = "zstd" if (zs and self._pool_supports_zstd()) else "gzip"
        out_path = usdz_path.with_suffix(f".usdz.{'zst' if fmt == 'zstd' else 'gz'}")
        with usdz_path.open("rb") as src:
            if fmt == "zstd":
                with out_path.open("wb") as dst:
                    zs.ZstdCompressor(level=10).copy_stream(src, dst)
            else:
                with gzip.open(out_path, "wb", compresslevel=6) as dst:
                    while chunk := src.read(1 << 20):
                        dst.write(chunk)
        size = out_path.stat().st_size
        if size < raw * 0.95:
            logger.info("remote render payload compressed: %.1f MB %s (%.0f%%)",
                        size / _MB, fmt, 100.0 * size / raw)
            return out_path, fmt
        out_path.unlink(missing_ok=True)
        logger.info("remote render payload: %s saves only %.1f%% (< 5%% — the bundle "
                    "is already compressed) — uploading the raw %.1f MB USDZ",
                    fmt, 100.0 * (1.0 - size / raw) if raw else 0.0, raw / _MB)
        return usdz_path, "none"

    def _effective_cap_mb(self) -> tuple[float, str]:
        """(upload cap in MB, its provenance for error messages).

        The config DEFAULT (512) is a guess; the backend knows its actual body
        limit and newer services advertise it on /live. When the user left the
        cap at the default and the fleet advertises a bigger limit, adopt the
        smallest advertised value — round 8: the benchmark's own L40 idled for
        33 min of CPU renders because the default cap fail-fasted the flagship
        scene while the backend would have taken it. An explicitly configured
        cap (any non-default value, or 0 = unlimited) always wins."""
        cap = self._max_upload_mb
        if cap != _DEFAULT_MAX_UPLOAD_MB:
            return cap, "render.remote_max_upload_mb"
        advertised = [s.info.get("max_body_bytes") for s in self._pool
                      if s.verified and isinstance(s.info, dict)]
        if isinstance(self._backend_info, dict):
            advertised.append(self._backend_info.get("max_body_bytes"))
        vals = [float(a) / _MB for a in advertised
                if isinstance(a, (int, float)) and a > 0]
        if vals and min(vals) > cap:
            return min(vals), "the backend's advertised body limit"
        return cap, "render.remote_max_upload_mb"

    @staticmethod
    def _gzip_from_zstd(send_path: Path) -> Path:
        """Transcode a .usdz.zst sibling to .usdz.gz for a backend without zstd.
        Idempotent + rename-atomic (concurrent dispatches to two such slots)."""
        gz = send_path.with_suffix(".gz")  # scene_bundle.usdz.zst -> scene_bundle.usdz.gz
        if gz.exists():
            return gz
        zs = _zstd_module()
        if zs is None:  # we produced the .zst, so this cannot happen in practice
            raise RuntimeError("zstandard module vanished mid-render — cannot "
                               "transcode the bundle for a non-zstd backend")
        tmp = gz.with_name(f"{gz.name}.tmp{os.getpid()}")
        with send_path.open("rb") as src, gzip.open(tmp, "wb", compresslevel=6) as dst:
            zs.ZstdDecompressor().copy_stream(src, dst)
        os.replace(tmp, gz)
        return gz

    @staticmethod
    def _log_upload_ledger(send_path: Path, nbytes: int, seconds: float) -> None:
        """Plan-Phase-0 counters: per-upload throughput + re-upload waste."""
        try:
            st = os.stat(send_path)
            key = (os.path.realpath(send_path), st.st_size, st.st_mtime_ns)
        except OSError:
            key = None
        n = 1
        reshipped = 0
        if key is not None:
            with _UPLOAD_LEDGER_LOCK:
                if len(_UPLOAD_LEDGER) > 256 and key not in _UPLOAD_LEDGER:
                    _UPLOAD_LEDGER.clear()  # bounded; identity resets are harmless
                n = _UPLOAD_LEDGER.get(key, 0) + 1
                _UPLOAD_LEDGER[key] = n
                if n > 1:
                    _UPLOAD_RESHIPPED[0] += nbytes
                reshipped = _UPLOAD_RESHIPPED[0]
        mbps = (nbytes / _MB) / seconds if seconds > 0 else 0.0
        extra = ""
        if n > 1:
            extra = (f" — SAME content uploaded {n}x "
                     f"({(n - 1) * nbytes / _MB:.1f} MB re-shipped; process total "
                     f"{reshipped / _MB:.1f} MB — the dedup target)")
        logger.info("upload: %.1f MB in %.1fs (%.2f MB/s)%s",
                    nbytes / _MB, seconds, mbps, extra)

    def _check_limit(self, send_path: Path) -> None:
        size = send_path.stat().st_size
        cap, source = self._effective_cap_mb()
        if cap and size > cap * _MB:
            raise RuntimeError(
                f"remote render upload is {size / _MB:.1f} MB, over the {cap:g} MB "
                f"cap ({source}). Reduce the scene "
                "(exclude prims, smaller textures) or raise the limit on both ends "
                "(client config + the service's OVRTX_MAX_BODY_BYTES).")

    def _check_estimate(self, est_bytes: int, n_files: int, scene: str) -> None:
        """Two-tier pre-packaging check against the configured upload cap.

        Raw on-disk bytes are not wire bytes: gzip routinely halves a USD upload,
        so an estimate just over the cap can still compress under it. Only an
        estimate over 2× the cap fails fast (packaging a big stage takes minutes
        with no realistic chance the result fits); between 1× and 2× we warn and
        proceed — the exact post-gzip `_check_limit` on the final bytes still gates
        the upload. The estimate covers layers AND resolved asset dependencies
        (see _estimate_stage_bytes), so it is truthful enough for the 2× tier to
        fire *before* the minutes-long packaging+gzip. With
        render.remote_max_upload_mb = 0 (unlimited) this is a no-op; the config
        default is 512 MB (an unbounded default let a 1,377 MB bundle monopolize
        the render service for minutes).
        """
        cap, source = self._effective_cap_mb()
        if not (cap and est_bytes > cap * _MB):
            return
        if est_bytes <= 2 * cap * _MB:
            logger.warning(
                "remote render (%s): the scene's %d referenced file(s) (layers + "
                "assets) total ~%.1f MB on disk, over the %g MB cap (%s) "
                "— proceeding anyway because compression may still fit the "
                "upload under the cap; the exact check runs on the final "
                "(compressed) bytes",
                scene, n_files, est_bytes / _MB, cap, source)
            return
        raise RuntimeError(
            f"remote render ({scene}): the scene's {n_files} referenced file(s) "
            f"(layers + assets) total ~{est_bytes / _MB:.1f} MB on disk, already over "
            f"2× the {cap:g} MB cap ({source}) before packaging "
            "even starts — failing fast instead of spending minutes bundling an upload "
            "the limit would reject. Reduce the scene (exclude prims, smaller textures), "
            "raise the limit on both ends (client config + the service's "
            "OVRTX_MAX_BODY_BYTES), or set it to 0 for unlimited.")

    def _check_estimate_cas(self, est_bytes: int, n_files: int, scene: str) -> None:
        """Pre-packaging fail-fast for a CAS-ready fleet: size against the smallest
        advertised staged-scene ceiling (`max_scene_bytes`), not the per-body cap —
        a manifest render has no single big body, so the bundle cap would wrongly
        reject scenes whose warm re-render uploads KBs. Same two-tier shape as
        `_check_estimate`: >2× fails before minutes of packaging, 1–2× warns (the
        estimate over-counts files packaging may drop)."""
        limit = self._fleet_max_scene_bytes()
        if not (limit and est_bytes > limit):
            return
        if est_bytes <= 2 * limit:
            logger.warning(
                "remote render (%s): the scene's %d referenced file(s) total "
                "~%.1f MB on disk, over the fleet's %.1f MB staged-scene limit — "
                "proceeding; the service enforces the exact staged size",
                scene, n_files, est_bytes / _MB, limit / _MB)
            return
        raise RuntimeError(
            f"remote render ({scene}): the scene's {n_files} referenced file(s) "
            f"total ~{est_bytes / _MB:.1f} MB on disk, over 2× the fleet's "
            f"{limit / _MB:.1f} MB staged-scene limit (max_scene_bytes on /live) — "
            "failing fast instead of packaging an upload the service would reject. "
            "Reduce the scene (exclude prims, smaller textures) or raise "
            "OVRTX_MAX_BODY_BYTES on the service.")

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _ensure_protocol(self, client) -> None:
        """Refuse to render against a backend built from a different checkout.

        Runs before any packaging/upload work (fail fast — bundling a large scene can
        take minutes) and only once per backend instance."""
        if not self._verify_version or self._protocol_verified:
            return
        from usd_core.remote_protocol import check_remote_protocol

        self._backend_info = (
            check_remote_protocol(
                self._base_url,
                client=client,
                api_key=self._api_key,
            )
            or {}
        )
        self._protocol_verified = True
        if self._pool and self._pool[0].url == self._base_url:
            self._pool[0].verified = True  # keep the pool's view in sync
            self._pool[0].info = self._backend_info

    def _post_multipart(self, client, send_path: Path, compression: str, params: dict):
        """POST /render/upload (streamed multipart). Returns the response, or None when
        the service predates the endpoint (fall back to base64 JSON)."""
        import httpx  # lazy

        # The write component turns a mid-transfer stall (no socket write making
        # progress for _UPLOAD_WRITE_STALL_S) into a prompt failure that demotes
        # the slot, instead of a silent wait until the full request timeout.
        # Read/connect/pool keep the configured request timeout.
        try:
            # never LOOSEN a stricter configured budget: the stall ceiling only
            # tightens the write phase (remote_timeout=10 must not allow a 60s
            # blocked write)
            stall = _UPLOAD_WRITE_STALL_S
            if isinstance(self._timeout, (int, float)) and self._timeout:
                stall = min(float(self._timeout), stall)
            upload_timeout = httpx.Timeout(self._timeout, write=stall)
        except Exception:  # noqa: BLE001 — an exotic timeout object: use it as-is
            upload_timeout = self._timeout
        reader = _ProgressReader(send_path, send_path.name)
        try:
            resp = client.post(
                f"{self._base_url}/render/upload",
                files={"file": (send_path.name, reader, "application/octet-stream")},
                data={"params": json.dumps({**params, "compression": compression})},
                headers=self._headers(), timeout=upload_timeout)
        finally:
            self._last_upload_seconds = reader.upload_seconds
            reader.close()
        if resp.status_code in (404, 405):  # older service — no multipart endpoint
            logger.info("remote service has no /render/upload — falling back to base64 JSON")
            return None
        return resp

    def _post_legacy(self, client, usdz_path: Path, params: dict):
        usdz_b64 = base64.b64encode(usdz_path.read_bytes()).decode("ascii")
        self._last_legacy_bytes = len(usdz_b64)  # what this transport actually sends
        logger.info("remote render payload: %.1f MB base64 JSON (legacy transport)",
                    len(usdz_b64) / _MB)
        return client.post(f"{self._base_url}/render",
                           json={"usdz_base64": usdz_b64, **params},
                           headers=self._headers(), timeout=self._timeout)

    def _raise_for_status(self, resp, sent_bytes: int, scene: str = "<unknown>") -> None:
        """Surface a non-2xx backend response with its body, and log request context.

        `httpx.raise_for_status()` alone yields only "Server error '500 ...' for url
        '...'" — the agent then has to read source and openapi.json to learn what the
        backend actually complained about (benchmark task-02). Carry the body
        (truncated) in the error and put the request context in the daemon log.
        """
        if resp.is_success:
            return
        try:
            url = str(resp.request.url)
        except Exception:  # noqa: BLE001 — a response fabricated without a request
            url = self._base_url
        try:
            body = (resp.text or "").strip()
        except Exception:  # noqa: BLE001 — undecodable body: diagnostics only
            body = ""
        # the body is attacker-controlled (a compromised backend): make it terminal/
        # log-safe before it reaches the daemon log or the raised error text
        body = _sanitize_text(body)
        if len(body) > _ERROR_BODY_LIMIT:
            body = body[:_ERROR_BODY_LIMIT] + "… [truncated]"
        logger.error(
            "remote render failed: HTTP %d from %s (scene %s, %.1f MB sent). "
            "Response body: %s",
            resp.status_code, url, scene, sent_bytes / _MB, body or "<empty>")
        if resp.status_code == 413:
            raise RuntimeError(
                f"remote render service rejected the upload as too large "
                f"({sent_bytes / _MB:.1f} MB sent; HTTP 413). Raise OVRTX_MAX_BODY_BYTES "
                "on the service, or shrink the scene (exclude prims, smaller textures)."
                + (f" Service said: {body}" if body else ""))
        raise RuntimeError(
            f"remote render failed: HTTP {resp.status_code} from {url} "
            f"(scene {scene}, {sent_bytes / _MB:.1f} MB sent) — "
            f"{body or '<empty response body>'}")

    def _bundle_cache_key(self, stage, deps: list[str] | None = None) -> str | None:
        """sha256 fingerprint of everything the packaged bundle depends on, or None
        when the stage can't be fingerprinted (that render simply isn't cached).

        Dirty/anonymous/package-internal layers (in-memory edits: the managed
        camera, prim excludes) are hashed by their full serialized content. Clean
        on-disk layers are keyed by (realPath, size, mtime_ns) — and, for files up
        to `_CACHE_CONTENT_HASH_MAX_BYTES`, also by their content sha, so a byte
        change with a preserved size+mtime cannot serve a stale bundle. Larger
        clean layers keep the stat-only key: the residual risk of a same-size
        same-mtime_ns byte change in a >8 MB layer is accepted — hashing multi-GB
        layers every render would cost more than the cache saves. External asset
        dependencies (textures, payloads — `deps`, precomputed by the caller or
        walked here via _walk_asset_dependencies) are keyed by (path, size,
        mtime_ns) each, so editing a texture in place invalidates the bundle;
        when the dependency walk fails the stage is UNCACHEABLE (an unknown
        dependency set must never produce a hit). Packaging options that change
        the artifact (the compress flag) are folded in.
        """
        import hashlib

        try:
            if deps is None:
                deps = _walk_asset_dependencies(stage)
            if deps is None:
                logger.debug("bundle cache: dependency walk failed — uncacheable")
                return None
            h = hashlib.sha256()
            # v3: dirty layers are fingerprinted WITHOUT tool-authored cameras
            # (they are also stripped from the bundle and sent as camera_defs),
            # so changing the viewpoint no longer misses an identical bundle.
            h.update(f"v3;compress={int(self._compress)};".encode())
            layers = sorted(stage.GetUsedLayers(includeClipLayers=True),
                            key=lambda lyr: lyr.identifier or "")
            for layer in layers:
                real = getattr(layer, "realPath", "") or ""
                fingerprint: bytes | None = None
                if real and not layer.dirty and not layer.anonymous:
                    try:
                        st = os.stat(real)
                        content = ""
                        if st.st_size <= _CACHE_CONTENT_HASH_MAX_BYTES:
                            content = hashlib.sha256(
                                Path(real).read_bytes()).hexdigest()
                        fingerprint = (f"disk:{real}:{st.st_size}:"
                                       f"{st.st_mtime_ns}:{content};").encode()
                    except OSError:  # usdz-internal path etc. — hash content instead
                        fingerprint = None
                if fingerprint is None:
                    text = _layer_text_without_managed_cams(layer)
                    fingerprint = f"mem:{len(text)}:".encode() + text.encode()
                h.update(fingerprint)
            for dep in sorted(set(deps)):
                try:
                    st = os.stat(dep)
                    h.update(f"dep:{dep}:{st.st_size}:{st.st_mtime_ns};".encode())
                except OSError:  # unresolved/package-internal — key its absence
                    h.update(f"dep:{dep}:absent;".encode())
            return h.hexdigest()
        except Exception:  # noqa: BLE001 — no fingerprint, no caching; never fail
            logger.debug("bundle cache: could not fingerprint the stage",
                         exc_info=True)
            return None

    def _package_payload(self, stage, work_dir: Path, scene: str) -> tuple[Path, Path, str]:
        """Estimate-check, package, and compress the stage once — or reuse the
        cached bundle for identical content (see the module docstring's
        "Bundle cache" section).

        Returns (usdz_path, file_to_send, compression) pinned inside `work_dir`
        for this render's whole upload lifetime. Cache hits are hard-linked (or
        copied) there so concurrent cache eviction cannot invalidate an in-flight
        manifest or upload. The client-side upload limit is already enforced on
        the final bytes. The dependency walk runs ONCE and feeds both the cache
        key and the size estimate.
        """
        deps = _walk_asset_dependencies(stage)
        entry: Path | None = None
        if id(stage) in _CAPPED_SENTINEL:
            _CAPPED_SENTINEL.discard(id(stage))
            deps_cacheable = False
        else:
            deps_cacheable = deps is not None
        if self._bundle_cache and deps_cacheable:
            key = self._bundle_cache_key(stage, deps)
            if key is not None:
                entry = _bundle_cache_entry_dir(key, scene)  # None: cache disabled
        # A fleet that is CAS-ready end-to-end never ships the whole bundle: the
        # total-size fail-fast and bundle-level compression are replaced by the
        # staged-scene ceiling and per-blob checks at upload time. Mixed/unknown
        # fleets keep the bundle-transport behavior (CAS still applies per slot).
        cas_ready = self._cas and self._pool_supports_feature("cas")
        if entry is not None:
            hit = _bundle_cache_get(entry)
            if hit is not None:
                materialized = _materialize_bundle_cache_hit(hit, work_dir)
                if materialized is not None:
                    usdz_path, send_path, compression = materialized
                    logger.info(
                        "remote render (%s): reusing cached bundle (%.1f MB %s, "
                        "packaging skipped) — %s", scene,
                        send_path.stat().st_size / _MB,
                        "gzip" if compression == "gzip" else "raw usdz",
                        entry.name)
                    if not cas_ready:
                        self._check_limit(send_path)
                    return usdz_path, send_path, compression
        est_bytes, n_files = _estimate_stage_bytes(stage, deps)
        if cas_ready:
            self._check_estimate_cas(est_bytes, n_files, scene)
        else:
            self._check_estimate(est_bytes, n_files, scene)
        logger.info("remote render (%s): packaging scene for upload "
                    "(%d referenced files, ~%.1f MB on disk)",
                    scene, n_files, est_bytes / _MB)
        usdz_path = self._package_usdz(stage, work_dir)
        if cas_ready:
            # per-blob wire compression happens at upload; compressing the whole
            # bundle here would be pure wasted wall-clock on the warm path
            logger.info("remote render (%s): CAS-ready fleet — deferring "
                        "compression and size checks to per-blob uploads", scene)
            send_path, compression = usdz_path, "none"
        else:
            send_path, compression = self._prepare_upload(usdz_path)
            self._check_limit(send_path)
        if entry is not None:
            # Publishing uses hard links/copies and leaves these work-dir paths
            # intact, so this render never uploads directly from an evictable
            # cache entry.
            _bundle_cache_put(entry, usdz_path, send_path)
        return usdz_path, send_path, compression

    def _dispatch(self, client, usdz_path: Path, send_path: Path, compression: str,
                  params: dict, scene: str) -> list[dict]:
        """Upload an already-packaged bundle to this backend and return result items.

        Prefers the CAS manifest transport when this backend advertises it
        (negotiate → upload only missing blobs → render by manifest — see
        `_dispatch_cas`); the streamed bundle upload is the fallback for
        non-CAS backends and for a service that unexpectedly lacks the endpoints.

        Reports the outcome to the pool scheduler via the `_note_upload` /
        `_note_failure` hooks (no-ops outside a multi-backend pool): a success
        records the slot's upload throughput, any error or stall demotes it.
        """
        t0 = time.perf_counter()
        try:
            items = upload_bytes = upload_seconds = None
            if self._slot_supports_cas():
                try:
                    items, upload_bytes, upload_seconds = self._dispatch_cas(
                        client, usdz_path, params, scene)
                except _CasUnsupportedError as exc:
                    logger.warning("remote render (%s): %s — falling back to the "
                                   "bundle upload", scene, exc)
            if items is None:
                sent = send_path.stat().st_size
                # the CAS-skip packaging path never sized the bundle against the
                # cap; a fallback upload must not bypass it
                self._check_limit(send_path)
                logger.info("remote render (%s): uploading %.1f MB to %s",
                            scene, sent / _MB, self._base_url)
                resp = self._post_multipart(client, send_path, compression, params)
                # prefer the streamed-body window (pure upload time); fall back to
                # the whole request when the body was never streamed (stubbed tests)
                upload_bytes = sent
                upload_seconds = (self._last_upload_seconds
                                  or (time.perf_counter() - t0))
                if resp is None:
                    self._check_limit(usdz_path)
                    # the multipart attempt was REJECTED (404/405): its duration
                    # must not rank this node — time and size the transport that
                    # succeeds
                    legacy_t0 = time.perf_counter()
                    resp = self._post_legacy(client, usdz_path, params)
                    upload_seconds = time.perf_counter() - legacy_t0
                    upload_bytes = self._last_legacy_bytes
                self._raise_for_status(resp, upload_bytes, scene)
                items = resp.json().get("results")
                if not isinstance(items, list):
                    raise RuntimeError(
                        "remote renderer returned an incomplete result set")
                self._log_upload_ledger(send_path, upload_bytes, upload_seconds)
        except Exception:
            self._note_failure()
            raise
        self._note_upload(upload_bytes, upload_seconds)
        return items

    def _dispatch_cas(self, client, usdz_path: Path, params: dict,
                      scene: str) -> tuple[list[dict], int, float]:
        """Content-addressed dispatch (plan Phase 2): negotiate the bundle's
        per-entry manifest, upload only the blobs this backend is missing, then
        render by manifest. Returns (result items, uploaded bytes, seconds).

        A blob evicted between negotiate and render answers 409 with the missing
        list — re-upload exactly those and retry once (the service never renders a
        partially-materialized scene)."""
        from usd_core.render import remote_cas

        manifest = remote_cas.manifest_for_usdz(usdz_path)
        files, root = manifest["files"], manifest["root"]
        t0 = time.perf_counter()
        resp = client.post(f"{self._base_url}/render/negotiate",
                           json={"files": files}, headers=self._headers(),
                           timeout=self._timeout)
        if resp.status_code in (404, 405):
            raise _CasUnsupportedError(
                f"{self._base_url} advertises 'cas' but has no /render/negotiate")
        self._raise_for_status(resp, 0, scene)
        try:
            missing = {str(s) for s in (resp.json().get("missing") or [])}
        except ValueError as exc:
            raise RuntimeError("remote render: negotiate returned a non-JSON "
                               "response") from exc
        uploaded, upload_seconds = self._upload_missing_blobs(
            client, usdz_path, files, missing, scene)
        total_bytes = sum(f["size"] for f in files)
        reused = [f for f in files if f["sha256"] not in missing]
        logger.info(
            "remote render (%s): CAS — %d of %d file(s) already on %s "
            "(%.1f of %.1f MB reused); uploaded %.1f MB",
            scene, len(reused), len(files), self._base_url,
            sum(f["size"] for f in reused) / _MB, total_bytes / _MB,
            uploaded / _MB)
        body = {**params, "files": files, "root": root}
        resp = client.post(f"{self._base_url}/render/manifest", json=body,
                           headers=self._headers(), timeout=self._timeout)
        if resp.status_code == 409:
            evicted = self._evicted_blobs_from_409(resp, files)
            logger.warning(
                "remote render (%s): %s evicted %d blob(s) mid-flight — "
                "re-uploading and retrying once", scene, self._base_url,
                len(evicted))
            more, more_seconds = self._upload_missing_blobs(
                client, usdz_path, files, evicted, scene)
            uploaded += more
            upload_seconds += more_seconds
            resp = client.post(f"{self._base_url}/render/manifest", json=body,
                               headers=self._headers(), timeout=self._timeout)
        self._raise_for_status(resp, uploaded, scene)
        items = resp.json().get("results")
        if not isinstance(items, list):
            raise RuntimeError("remote renderer returned an incomplete result set")
        return items, uploaded, upload_seconds or (time.perf_counter() - t0)

    @staticmethod
    def _evicted_blobs_from_409(resp, files: list[dict]) -> set[str]:
        """The sha set a 409 asks to re-upload; an unparseable body re-uploads
        everything in the manifest (correct, just not minimal)."""
        try:
            detail = resp.json().get("detail")
            evicted = {str(s) for s in detail.get("missing")}
            if evicted:
                return evicted
        except Exception:  # noqa: BLE001 — fall through to the conservative set
            pass
        return {f["sha256"] for f in files}

    def _upload_missing_blobs(self, client, usdz_path: Path, files: list[dict],
                              missing: set, scene: str) -> tuple[int, float]:
        """PUT the manifest entries whose sha is in `missing` (deduped); returns
        (bytes actually sent, seconds spent streaming). Each blob is individually
        wire-compressed when that pays and individually subject to the upload cap
        and the stall/min-throughput protection of `_ProgressReader`."""
        if not missing:
            return 0, 0.0
        import zipfile

        import httpx  # lazy, matches _post_multipart

        from usd_core.render import remote_cas

        codec = None
        if self._compress:
            zs = _zstd_module()
            slot_features = (self._backend_info or {}).get("features") or []
            codec = "zstd" if (zs and "zstd" in slot_features) else "gzip"
        cap, cap_source = self._effective_cap_mb()
        try:
            stall = _UPLOAD_WRITE_STALL_S
            if isinstance(self._timeout, (int, float)) and self._timeout:
                stall = min(float(self._timeout), stall)
            upload_timeout = httpx.Timeout(self._timeout, write=stall)
        except Exception:  # noqa: BLE001 — an exotic timeout object: use it as-is
            upload_timeout = self._timeout
        uploaded, seconds = 0, 0.0
        seen: set[str] = set()
        with zipfile.ZipFile(usdz_path) as zf, tempfile.TemporaryDirectory(
                prefix=f"{_STAGING_PREFIX}{os.getpid()}_blob_",
                dir=str(_work_staging_root())) as td:
            for f in files:
                sha = f["sha256"]
                if sha not in missing or sha in seen:
                    continue
                seen.add(sha)
                blob_path, nbytes, blob_comp = remote_cas.stage_blob(
                    zf, f["path"], Path(td), codec)
                if cap and nbytes > cap * _MB:
                    raise RuntimeError(
                        f"remote render ({scene}): bundle file {f['path']} is "
                        f"{nbytes / _MB:.1f} MB ({blob_comp}) — over the {cap:g} MB "
                        f"per-upload cap ({cap_source}). Reduce the scene or raise "
                        "the limit on both ends (client config + the service's "
                        "OVRTX_MAX_BODY_BYTES).")
                reader = _ProgressReader(blob_path, f["path"])
                try:
                    resp = client.put(
                        f"{self._base_url}/blobs/{sha}",
                        params={"compression": blob_comp},
                        content=iter(lambda r=reader: r.read(1 << 20), b""),
                        headers={**self._headers(),
                                 "Content-Length": str(nbytes)},
                        timeout=upload_timeout)
                finally:
                    seconds += reader.upload_seconds
                    reader.close()
                self._raise_for_status(resp, nbytes, scene)
                uploaded += nbytes
        return uploaded, seconds

    def _execute(self, stage, params: dict) -> tuple[list[dict], float]:
        """Package the stage once, upload it, and return (result items, elapsed seconds)."""
        import httpx  # lazy

        _reap_stale_staging()  # sweep leftovers from cancelled/killed renders
        scene = _scene_stem(stage)
        t0 = time.perf_counter()
        # Bundle work files live in a secured staging root (see
        # _work_staging_root), not the user's output dir: a killed process must
        # never leave a scene_bundle.usdz where an image was expected (that read
        # as a "successful" render with no PNG). The PID in the name lets the
        # reaper above identify dirs whose owner died mid-render.
        work_dir = Path(tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{os.getpid()}_",
                                         dir=str(_work_staging_root())))
        try:
            with httpx.Client() as client:
                self._ensure_protocol(client)
                usdz_path, send_path, compression = self._package_payload(
                    stage, work_dir, scene)
                _signal_stage_released()  # stage no longer needed: unblock readers
                items = self._dispatch(client, usdz_path, send_path, compression,
                                       params, scene)
                self._bind_backend_identity(
                    items,
                    endpoint=self._base_url,
                    info=self._backend_info,
                )
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        return items, time.perf_counter() - t0

    @staticmethod
    def _bind_backend_identity(
            items: list[dict], *, endpoint: str, info: dict) -> None:
        """Attach the exact remote service identity to every returned image."""
        identity = {
            "endpoint": endpoint.rstrip("/"),
            "engine": info.get("engine"),
            "protocol_version": info.get("protocol_version"),
            "status": info.get("status"),
        }
        for item in items:
            if isinstance(item, dict):
                item["_render_backend_identity"] = dict(identity)

    # ── multi-backend pool: scheduling, health, sharding ──────────────────────────

    def _child(self, slot: _BackendSlot) -> "RemoteRenderBackend":
        """A single-backend client bound to `slot` (same knobs, slot's address).

        Keeps every per-request code path (`_execute`, `_dispatch`, transports) written
        against one `_base_url`/`_api_key` pair; the pool layer only picks slots.
        """
        child = RemoteRenderBackend(slot.url, api_key=slot.api_key, timeout=self._timeout,
                                    compress=self._compress,
                                    max_upload_mb=self._max_upload_mb,
                                    verify_version=self._verify_version,
                                    bundle_cache=self._bundle_cache, cas=self._cas)
        child._protocol_verified = slot.verified  # don't re-probe a verified backend
        # the slot's /live payload, so per-slot decisions (CAS, zstd) see it
        child._backend_info = slot.info if isinstance(slot.info, dict) else {}
        # feed upload outcomes back to the shared scheduler state (throughput
        # samples + failure demotion — see _acquire_slot)
        state = self._pool_state
        child._note_upload = (
            lambda nbytes, seconds, _s=slot: state.record_upload(_s, nbytes, seconds))
        child._note_failure = lambda _s=slot: state.record_failure(_s)
        return child

    def _probe_slot(self, slot: _BackendSlot) -> bool:
        """Lazily verify `slot`'s version handshake; mark it unhealthy (one warning)
        on mismatch or unreachability. Only used when the pool has >1 entries — a
        one-entry pool keeps the single-backend behavior of raising directly.
        Health transitions happen under the slot's probe lock: the pool state is
        shared across concurrent commands, so the handshake is single-flight per
        slot and healthy/verified/reason never race."""
        if not slot.healthy:
            # An unreachable backend is not unreachable FOREVER: round 7 saw a
            # transient network blip permanently blind a session to a backend
            # that recovered a minute later. After the cooldown, one probe gets
            # to prove recovery (failure re-stamps the cooldown).
            if _monotonic() - slot.unhealthy_at < _UNHEALTHY_REPROBE_S:
                return False
            with slot.probe_lock:
                if _monotonic() - slot.unhealthy_at >= _UNHEALTHY_REPROBE_S:
                    logger.info("remote render: re-probing backend %s after its "
                                "unusable cooldown", slot.url)
                    slot.healthy = True
                    slot.verified = False
        if slot.verified or not self._verify_version:
            return True
        with slot.probe_lock:
            if not slot.healthy:  # re-check: another thread probed while we waited
                return False
            if slot.verified:
                return True
            from usd_core.remote_protocol import check_remote_protocol

            try:
                slot.info = (
                    check_remote_protocol(slot.url, api_key=slot.api_key) or {}
                )
                slot.verified = True
                return True
            except Exception as exc:  # noqa: BLE001 — mismatch/unreachable: skip it
                slot.healthy = False
                slot.unhealthy_at = _monotonic()
                slot.reason = _sanitize_text(str(exc))
                logger.warning(
                    "remote render: backend %s is unusable and will be skipped for "
                    "this session (%s)", slot.url, slot.reason)
                return False

    def _healthy_slots(self) -> list[_BackendSlot]:
        """Every healthy (handshake-verified) slot; raises when the whole pool is out."""
        healthy = [s for s in self._pool if self._probe_slot(s)]
        if not healthy:
            detail = "; ".join(
                f"{s.url}: {_sanitize_text(s.reason) or 'unhealthy'}" for s in self._pool)
            raise RuntimeError(
                f"all {len(self._pool)} configured remote render backend(s) are "
                f"unusable — {detail}")
        return healthy

    @staticmethod
    def _demoted(slot: _BackendSlot, peers: list[_BackendSlot]) -> bool:
        """Deprioritize a degraded slot (module docstring, "Scheduling policy"):
        a recent failure/stall within its `_FAILURE_COOLDOWN_S` window, or recent
        upload throughput under `_DEMOTE_BPS_FRACTION` of the pool's best. The
        baseline "best" only counts peers WITHOUT a recent failure — a
        historically fast but currently failing node must not demote every
        healthy slower peer (which would instantly nullify its own demotion).
        After the cooldown a failed slot re-enters selection as a canary: one
        success clears the flag, another failure re-stamps the cooldown. A slot
        with no samples yet is never demoted on throughput (unknown is not slow).
        Callers hold the pool lock. Demoted slots are still used when nothing
        better exists — never starve a render because every backend is degraded."""
        if slot.recent_failure and _monotonic() - slot.failure_at < _FAILURE_COOLDOWN_S:
            return True
        if not slot.recent_bps:
            return False
        best = max((s.avg_bps for s in peers if not s.recent_failure), default=0.0)
        return best > 0 and slot.avg_bps < best * _DEMOTE_BPS_FRACTION

    @staticmethod
    def _demotion_transitions(demoted: list[_BackendSlot],
                              preferred: list[_BackendSlot]) -> list[str]:
        """Log lines for slots entering/leaving the demoted set (edge-triggered via
        `slot.demote_logged` — one line per transition, not per pick). Runs under
        the pool lock, so it only formats; the caller emits outside the lock.
        Demotion was previously invisible in the daemon log (benchmark round 5:
        ~15 minutes lost to a degraded backend with zero trace of why)."""
        lines = []
        for s in demoted:
            if not s.demote_logged:
                s.demote_logged = True
                why = ("recent failure/stall" if s.recent_failure
                       else f"throughput {s.avg_bps / _MB:.1f} MB/s under "
                            f"{_DEMOTE_BPS_FRACTION:.0%} of the pool's best")
                lines.append(f"remote render: backend {s.url} demoted ({why}) — "
                             "used only when no better slot is available")
        for s in preferred:
            if s.demote_logged:
                s.demote_logged = False
                lines.append(f"remote render: backend {s.url} re-promoted — "
                             "back in normal selection")
        return lines

    @staticmethod
    def _throughput_rank(slot: _BackendSlot) -> float:
        """Tiebreak key among equally-busy slots: higher recent throughput first;
        slots with no samples rank as best so new/idle slots get measured."""
        return slot.avg_bps if slot.recent_bps else float("inf")

    def _acquire_slot(self, exclude: _BackendSlot | None = None) -> _BackendSlot | None:
        """Reserve a healthy verified slot: non-demoted slots first (see _demoted;
        demoted ones only when nothing else is available — never starve), then
        fewest in-flight requests, then highest recent upload throughput,
        round-robin among remaining ties. Canary rule: after `_CANARY_EVERY`
        consecutive picks that all went to non-demoted slots, one render is routed
        to the best demoted slot so it can earn the success that clears its
        demotion (without it, a demoted slot with healthy peers never recovered).
        Returns None when no eligible slot exists — callers decide whether that is
        an error (`_healthy_slots` raises)."""
        while True:
            slot = None
            canary = False
            transitions: list[str] = []
            with self._pool_lock:
                eligible = [s for s in self._pool
                            if s.healthy and s.verified and s is not exclude]
                if not eligible and not self._verify_version:
                    eligible = [s for s in self._pool if s.healthy and s is not exclude]
                if eligible:
                    state = self._pool_state
                    demoted = [s for s in eligible if self._demoted(s, eligible)]
                    preferred = [s for s in eligible if s not in demoted]
                    transitions = self._demotion_transitions(demoted, preferred)
                    if not preferred:
                        candidates = eligible  # everything degraded — never starve
                        state.non_demoted_streak = 0
                    elif demoted and state.non_demoted_streak >= _CANARY_EVERY:
                        candidates = demoted  # canary: a recovery shot
                        state.non_demoted_streak = 0
                        canary = True
                    else:
                        candidates = preferred
                        state.non_demoted_streak += 1
                    low = min(s.in_flight for s in candidates)
                    ties = [s for s in candidates if s.in_flight == low]
                    best = max(self._throughput_rank(s) for s in ties)
                    top = [s for s in ties if self._throughput_rank(s) == best]
                    slot = top[state.rr % len(top)]
                    state.rr += 1  # persists across commands (shared state)
                    slot.in_flight += 1
            if slot is not None:
                for line in transitions:
                    logger.warning("%s", line)
                if canary:
                    logger.info(
                        "remote render: canary — routing one render to demoted "
                        "backend %s to let it prove recovery", slot.url)
                return slot
            # nothing verified yet: probe unverified slots (outside the lock — it's a
            # network call), then retry the pick; unhealthy ones drop out of the pool
            unverified = [s for s in self._pool
                          if s.healthy and not s.verified and s is not exclude]
            if not unverified:
                return None
            for s in unverified:
                self._probe_slot(s)

    def _release_slot(self, slot: _BackendSlot) -> None:
        with self._pool_lock:
            slot.in_flight = max(0, slot.in_flight - 1)

    def _execute_scheduled(self, stage, params: dict) -> tuple[list[dict], float]:
        """One remote render call, placed on the least-busy healthy backend.

        A one-entry pool goes straight through `_execute` (byte-identical to the
        single-backend client, including its error behavior)."""
        if len(self._pool) == 1:
            return self._execute(stage, params)
        slot = self._acquire_slot()
        if slot is None:
            self._healthy_slots()  # raises with the per-backend reasons
            raise RuntimeError("no healthy remote render backend available")
        self._release_slot(slot)  # _dispatch_to re-acquires; keep counters balanced

        def _validated(items: list[dict]) -> list[dict]:
            cams = len(params.get("cameras") or [])
            expected = cams * (len(params.get("frames") or []) or 1)
            if expected and len(items) != expected:
                raise RuntimeError(
                    f"remote renderer returned an incomplete result set "
                    f"({len(items)} of {expected})")
            return items

        # Package ONCE, then dispatch (and retry) the same immutable bundle.
        # Retrying via a second _execute repackaged the LIVE stage after the
        # early root-lock release — an unlocked stage traversal (review v6 P1-3).
        # This also makes the retry cheaper: no second packaging pass.
        _reap_stale_staging()
        scene = _scene_stem(stage)
        t0 = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{os.getpid()}_",
                                         dir=str(_work_staging_root())))
        try:
            usdz_path, send_path, compression = self._package_payload(
                stage, work_dir, scene)
            _signal_stage_released()  # stage no longer needed: unblock readers
            try:
                items = _validated(self._dispatch_to(
                    slot, usdz_path, send_path, compression, params, scene))
            except Exception as exc:  # noqa: BLE001 — one retry on a sibling backend
                if not _retryable_render_error(exc):
                    raise
                alt = self._acquire_slot(exclude=slot)
                if alt is None:
                    raise
                self._release_slot(alt)  # _dispatch_to re-acquires
                logger.warning(
                    "remote render: failed on %s (%s) — retrying once on %s",
                    slot.url, exc, alt.url)
                try:
                    items = _validated(self._dispatch_to(
                        alt, usdz_path, send_path, compression, params, scene))
                except Exception as retry_exc:
                    raise RuntimeError(
                        f"remote render failed on {slot.url} ({exc}) and on retry "
                        f"via {alt.url} ({retry_exc})") from retry_exc
            return items, time.perf_counter() - t0
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    def _dispatch_to(self, slot: _BackendSlot, usdz_path: Path, send_path: Path,
                     compression: str, params: dict, scene: str) -> list[dict]:
        """Upload the shared bundle to one slot (its own httpx client — one per thread)
        and validate the chunk's result count, so a short result set fails — and can be
        retried — right here."""
        import httpx  # lazy

        if (compression == "zstd" and isinstance(slot.info, dict)
                and slot.info and "zstd" not in (slot.info.get("features") or [])):
            # mixed fleet: this slot was verified after packaging chose zstd —
            # transcode to gzip for it rather than earn a 400
            send_path, compression = self._gzip_from_zstd(send_path), "gzip"
        self._acquire_specific(slot)
        try:
            with httpx.Client() as client:
                items = self._child(slot)._dispatch(client, usdz_path, send_path,
                                                    compression, params, scene)
        finally:
            self._release_slot(slot)
        frames = params.get("frames")
        if frames is not None and len(items) != len(frames):
            self._pool_state.record_failure(slot)  # short result sets demote too
            raise RuntimeError(
                f"backend {slot.url} returned {len(items)} images for "
                f"{len(frames)} frames (does the service support batched `frames`?)")
        self._bind_backend_identity(items, endpoint=slot.url, info=slot.info)
        return items

    def _acquire_specific(self, slot: _BackendSlot) -> None:
        with self._pool_lock:
            slot.in_flight += 1

    def _render_chunk(self, slot: _BackendSlot, usdz_path: Path, send_path: Path,
                      compression: str, params: dict, chunk: list[float],
                      scene: str) -> list[dict]:
        """Render one contiguous frame chunk on `slot`; on failure retry ONCE on the
        least-busy other healthy backend before surfacing a per-chunk error."""
        chunk_params = {**params, "frames": list(chunk)}
        try:
            return self._dispatch_to(slot, usdz_path, send_path, compression,
                                     chunk_params, scene)
        except Exception as exc:  # noqa: BLE001 — one retry on a sibling backend
            if not _retryable_render_error(exc):
                raise  # deterministic (4xx/local) failure: identical everywhere
            alt = self._acquire_slot(exclude=slot)
            if alt is None:
                raise
            self._release_slot(alt)  # _dispatch_to re-acquires; keep counters balanced
            logger.warning(
                "remote render (%s): frames %g..%g failed on %s (%s) — retrying once "
                "on %s", scene, chunk[0], chunk[-1], slot.url, exc, alt.url)
            try:
                return self._dispatch_to(alt, usdz_path, send_path, compression,
                                         chunk_params, scene)
            except Exception as retry_exc:
                raise RuntimeError(
                    f"remote render ({scene}): frames {chunk[0]:g}..{chunk[-1]:g} "
                    f"failed on {slot.url} ({exc}) and on retry via {alt.url} "
                    f"({retry_exc})") from retry_exc

    def _execute_sharded(self, stage, params: dict,
                         frames: list[float]) -> tuple[list[dict], float]:
        """Shard `frames` across the healthy pool: package ONCE, upload the same bundle
        to each backend in parallel, render each contiguous chunk remotely, and return
        the items reassembled in frame order.

        Degraded slots (recent failure/stall, or throughput far below the pool's
        best — see _demoted) are left out of the fan-out while any non-demoted slot
        exists: a slow node in the shard set is the long pole of the whole batch
        (benchmark: 9m12s vs 4.5s for the same 18 MB upload). When every slot is
        degraded they are all used anyway — never starve."""
        from concurrent.futures import ThreadPoolExecutor

        healthy = self._healthy_slots()  # raises when the whole pool is out
        with self._pool_lock:
            preferred = [s for s in healthy if not self._demoted(s, healthy)]
        if preferred and len(preferred) < len(healthy):
            logger.info(
                "remote render: leaving %d degraded backend(s) out of the frame "
                "shard fan-out: %s", len(healthy) - len(preferred),
                ", ".join(s.url for s in healthy if s not in preferred))
            healthy = preferred
        n_chunks = min(len(healthy), len(frames))
        if n_chunks <= 1:
            # one usable backend: identical to the single-backend batched path —
            # but still reserve the slot (this branch used to run without touching
            # in_flight, so concurrent scheduling piled work onto an "idle" slot)
            slot = healthy[0]
            self._acquire_specific(slot)
            try:
                if slot is self._pool[0]:
                    return self._execute(stage, params)
                return self._child(slot)._execute(stage, params)
            finally:
                self._release_slot(slot)

        _reap_stale_staging()  # sweep leftovers from cancelled/killed renders
        scene = _scene_stem(stage)
        t0 = time.perf_counter()
        work_dir = Path(tempfile.mkdtemp(prefix=f"{_STAGING_PREFIX}{os.getpid()}_",
                                         dir=str(_work_staging_root())))
        try:
            usdz_path, send_path, compression = self._package_payload(
                stage, work_dir, scene)
            _signal_stage_released()  # stage no longer needed: unblock readers
            chunks = _shard_frames(frames, n_chunks)
            logger.info(
                "remote render (%s): sharding %d frames into %d chunk(s) across %d "
                "backend(s)", scene, len(frames), len(chunks), len(healthy))
            with ThreadPoolExecutor(max_workers=len(chunks),
                                    thread_name_prefix="ov_remote_shard") as pool:
                futures = [pool.submit(self._render_chunk, healthy[i], usdz_path,
                                       send_path, compression, params, chunks[i], scene)
                           for i in range(len(chunks))]
                # collect in submit order — chunks are contiguous, so concatenating
                # per-chunk results reproduces the requested frame order exactly
                errors = []
                per_chunk: list[list[dict]] = []
                for i, fut in enumerate(futures):
                    try:
                        per_chunk.append(fut.result())
                    except Exception as exc:  # noqa: BLE001 — gather every chunk error
                        errors.append(f"frames {chunks[i][0]:g}..{chunks[i][-1]:g}: {exc}")
                if errors:
                    raise RuntimeError(
                        f"remote render ({scene}): {len(errors)} of {len(chunks)} frame "
                        "chunk(s) failed — " + "; ".join(errors))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
        items = [item for chunk_items in per_chunk for item in chunk_items]
        return items, time.perf_counter() - t0

    def render(self, stage, cameras, width, height, out_dir, mode: str = "quality",
               names: list[str] | None = None, frame: float | None = None) -> list[RenderResult]:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        params = {"cameras": list(cameras), "image_width": width,
                  "image_height": height, "mode": mode}
        cam_defs = _managed_camera_defs(stage, params["cameras"])
        token = None
        if cam_defs:
            # protocol v2: tool-authored cameras travel as specs (the bundle is
            # packaged without them — see _strip_cameras_from_prepared). This
            # covers user `camera create` cameras too (they carry
            # ovAuthoredCamera), so precise-framing churn no longer invalidates
            # the viewpoint-independent bundle cache.
            params["camera_defs"] = cam_defs
            token = _CAMERA_STRIP_PATHS.set(frozenset(d["path"] for d in cam_defs))
        try:
            # One remote call even for several cameras (the service ingests the scene
            # once for all of them) — the pool only chooses WHICH backend takes it.
            items, dt = self._execute_scheduled(stage, params)
        finally:
            if token is not None:
                _CAMERA_STRIP_PATHS.reset(token)
        if len(items) != len(cameras):
            raise RuntimeError("remote renderer returned an incomplete result set")
        results: list[RenderResult] = []
        for i, item in enumerate(items):
            cam = item.get("camera", "camera")
            stem = names[i] if names and i < len(names) else (cam.strip("/").replace("/", "_") or "camera")
            path = out_dir / f"{stem}.png"
            self._write_image(item, path)
            results.append(RenderResult(path=str(path), camera=cam, width=width,
                                        height=height, backend=self.name, render_time=dt,
                                        renderer_identity=item.get(
                                            "_render_backend_identity"),
                                        ovrtx_render_mode=item.get(
                                            "ovrtx_render_mode"),
                                        ovrtx_num_sensor_updates=item.get(
                                            "ovrtx_num_sensor_updates"),
                                        active_aov=item.get("active_aov")))
        _flag_blank_suspects(results)
        return results

    def render_frames(self, stage, camera: str, width, height, out_dir, frames: list[float],
                      mode: str = "quality", names: list[str] | None = None) -> list[RenderResult]:
        """Batch animation: upload the scene ONCE and render every frame server-side.

        Replaces the per-frame loop's N uploads + N stage-opens with one of each, and — unlike
        the single-frame `render` path used in a client loop — actually renders each frame at
        its USD time code (the service ingests the animated scene once and samples per frame).
        Results come back in the requested `frames` order.

        With a multi-entry backend pool the frame list is sharded into contiguous
        chunks — one per healthy backend, packaged once, uploaded in parallel — and the
        results are reassembled locally in the same frame order (see _execute_sharded).
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        frame_list = [float(f) for f in frames]
        params = {"cameras": [camera], "image_width": width, "image_height": height,
                  "mode": mode, "frames": frame_list}
        cam_defs = _managed_camera_defs(stage, [camera])
        token = None
        if cam_defs:
            params["camera_defs"] = cam_defs
            token = _CAMERA_STRIP_PATHS.set(frozenset(d["path"] for d in cam_defs))
        try:
            if len(self._pool) > 1 and len(frame_list) > 1:
                items, dt = self._execute_sharded(stage, params, frame_list)
            else:
                items, dt = self._execute(stage, params)
        finally:
            if token is not None:
                _CAMERA_STRIP_PATHS.reset(token)
        if len(items) != len(frames):
            raise RuntimeError(
                f"remote renderer returned {len(items)} images for {len(frames)} frames "
                "(does the service support batched `frames`?)")
        # Trust the requested order; the service renders frames sequentially in the order sent.
        results: list[RenderResult] = []
        for i, item in enumerate(items):
            cam = item.get("camera", camera)
            stem = names[i] if names and i < len(names) else f"f{int(round(frames[i])):04d}"
            path = out_dir / f"{stem}.png"
            self._write_image(item, path)
            results.append(RenderResult(path=str(path), camera=cam, width=width,
                                        height=height, backend=self.name, render_time=dt,
                                        renderer_identity=item.get(
                                            "_render_backend_identity"),
                                        ovrtx_render_mode=item.get(
                                            "ovrtx_render_mode"),
                                        ovrtx_num_sensor_updates=item.get(
                                            "ovrtx_num_sensor_updates"),
                                        active_aov=item.get("active_aov")))
        _flag_blank_suspects(results)
        return results

    @staticmethod
    def _write_image(item: dict, path: Path) -> None:
        try:
            image = base64.b64decode(item["image_base64"], validate=True)
        except (KeyError, ValueError) as exc:
            raise RuntimeError("remote renderer returned invalid image data") from exc
        path.write_bytes(image)
