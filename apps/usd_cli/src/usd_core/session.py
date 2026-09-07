# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Session — the library-facing engine API (PR-9 holds the live stage in memory).

This is the surface you import and call directly when using usd-cli as a library. Every
public method maps 1:1 to a CLI command and returns a `Response`. The bodies do the real
work directly against USD (`pxr.Usd.Stage`) and the OVRTX render backends (local / remote);
the one exception is `eval`, the code-as-action escape hatch, which still raises
`NotImplementedError`.

Keeping these as plain methods on a plain object is what makes "run it as a library
later" free — the CLI and server are thin adapters over exactly these calls.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import logging
import math
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
import weakref
from contextlib import ExitStack, contextmanager, suppress
from pathlib import Path, PurePosixPath

from usd_core.config import Config, load_config
from usd_core.history import History, Op
from usd_core.models import Artifact, Issue, Response
from usd_core.refs import RefEntry, RefTable

logger = logging.getLogger(__name__)

# Captured ONCE at import (single-threaded): reading the umask later via the
# set-and-restore dance is process-global and races concurrent daemon workers,
# briefly widening every file they create.
_PROCESS_UMASK = os.umask(0)
os.umask(_PROCESS_UMASK)

_SESSION_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")

# Commands that preserve resolved mesh points/topology/transforms/visibility.
# Reusing their verified topology capture keeps progressive material workflows
# observable without rebuilding a large line sidecar for every assigned prim.
_VIEWER_GEOMETRY_PRESERVING_MUTATIONS = {
    "appearance_clear", "camera_create", "camera_fit", "camera_look_at",
    "camera_orbit", "camera_pan", "camera_zoom", "material", "material_apply",
    "physics_apply", "remove_api",
}

# ── cross-session root-layer ownership ────────────────────────────────────────────────
# USD caches SdfLayers process-wide by identifier, so two Sessions that open the same
# file share ONE mutable root layer: the second open's force-reload would discard the
# first session's unsaved edits, and later edits would bleed both ways. Full per-session
# layer cloning is out of scope; instead we track which live Session owns each root
# layer and refuse a concurrent second open, so nothing is silently corrupted.
_ROOT_LAYER_LOCK = threading.Lock()
_ROOT_LAYER_OWNERS: dict[str, tuple[str, "weakref.ref[Session]"]] = {}
# Read-only sessions (`open --read-only`) register here instead: they deliberately
# share the writer's live root layer, so the owners guard must not block them (nor
# they it). Bookkeeping only — any number of readers may coexist with one writer.
_ROOT_LAYER_READERS: dict[str, dict[str, "weakref.ref[Session]"]] = {}


def _prune_readers(key: str) -> None:
    """Drop reader entries whose Session died or moved to a different stage.
    Callers must hold `_ROOT_LAYER_LOCK`."""
    readers = _ROOT_LAYER_READERS.get(key)
    if not readers:
        _ROOT_LAYER_READERS.pop(key, None)
        return
    for name, ref in list(readers.items()):
        s = ref()
        if s is None or getattr(s, "_reader_key", None) != key:
            del readers[name]
    if not readers:
        del _ROOT_LAYER_READERS[key]


# ── per-destination publish serialization ─────────────────────────────────────────────
# Two sessions publishing (save / export / convert) to the SAME destination used to
# interleave their export→verify→replace windows: last-writer-wins with a torn middle
# where one session's verified bytes could be replaced by the other's mid-report. One
# module-level lock per canonical destination path serializes the whole pipeline.
_PUBLISH_LOCKS_LOCK = threading.Lock()
_PUBLISH_LOCKS: dict[str, threading.Lock] = {}


def _publish_dest_lock(key: str) -> threading.Lock:
    """The per-destination publish lock for canonical (realpath) destination `key`."""
    with _PUBLISH_LOCKS_LOCK:
        return _PUBLISH_LOCKS.setdefault(key, threading.Lock())


def _mutating(method):
    """Guard a stage-mutating Session command against read-only sessions.

    A session opened with `open --read-only` shares the writer session's LIVE root
    layer (USD caches SdfLayers process-wide) — that is the point, and the peril:
    any authoring here would bleed straight into the writer's unsaved stage. One
    decorator on every mutating command is the single choke point, instead of
    scattered per-method checks."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        if self.read_only:
            cmd = method.__name__.strip("_").replace("_", "-")
            return self._err(cmd, RuntimeError(
                f"session '{self.name}' is read-only (opened with --read-only) — "
                "mutating commands are blocked; re-open without --read-only to edit"))
        result = method(self, *args, **kwargs)
        # Advance the mutation epoch on every SUCCESSFUL mutating command: the
        # unsaved-edit reopen guard compares this too, so mutations that bypass
        # the undo history (physics.apply patches, camera authoring, set, …)
        # still count as unsaved work (review v6, P1-5). Commands that persist
        # state WITHOUT creating unsaved stage edits are exempt — save stamps
        # the epoch itself, and checkpoints copy layers without touching the
        # live stage.
        if (getattr(result, "ok", False)
                and method.__name__ not in ("save", "checkpoint_save",
                                            "checkpoint_delete")):
            self._mutation_epoch += 1
            self._viewer_revision += 1
            if method.__name__ not in _VIEWER_GEOMETRY_PRESERVING_MUTATIONS:
                self._invalidate_viewer_line_geometry()
        return result
    # The daemon fast-fails mutating commands on read-only sessions BEFORE they
    # queue on the session lock (a blocked `camera fit` used to wait out a long
    # render only to be rejected at execution — benchmark round 5).
    wrapper._is_mutating = True
    return wrapper


def _root_layer_owner(key: str, me: "Session | None") -> str | None:
    """The name of ANOTHER live Session that currently has `key` open, else None.

    Stale entries (owner garbage-collected, or owner has since opened a different
    stage) are pruned on the way. Callers must hold `_ROOT_LAYER_LOCK`.
    """
    entry = _ROOT_LAYER_OWNERS.get(key)
    if entry is None:
        return None
    name, ref = entry
    owner = ref()
    if owner is None or getattr(owner, "_layer_key", None) != key:
        del _ROOT_LAYER_OWNERS[key]
        return None
    return None if owner is me else name


# snapshot banks a full-stage snapdiff baseline for a later `-D`; past this many
# prims that capture is deferred until diffing is requested (see Session.snapshot)
_SNAPDIFF_CAPTURE_MAX_PRIMS = 25_000

class Session:
    """Holds a live USD stage, the ref table, the op-log, and the snapshot baseline."""

    def __init__(self, config: Config | None = None, name: str = "default") -> None:
        # `name` identifies this session (the daemon's `--session foo`); it namespaces
        # the on-disk defaults (renders / checkpoints / physics staging) so concurrent
        # sessions don't overwrite each other. The default name preserves the historical
        # un-namespaced layout exactly.
        if not isinstance(name, str) or not _SESSION_NAME_RE.fullmatch(name):
            raise ValueError("session name must be 1-64 letters, digits, '.', '_' or '-' "
                             "and must start with a letter or digit")
        self.name = name
        # Each Session owns a private DEEP COPY of its Config: all sessions sharing one
        # mutable Config let one session's (former) in-place renderer override corrupt
        # every other session's — and the daemon's — settings.
        self.config = copy.deepcopy(config) if config is not None else load_config()
        # Session-start snapshot of the project config file's [render] section:
        # render hot-reload (`_fresh_render_section`) uses it to detect keys the
        # agent has since REMOVED from the file, so they fall back to defaults
        # instead of persisting for the session's lifetime.
        self._render_project_baseline = self._read_project_render_section()
        self.refs = RefTable()
        self.history = History()
        self._stage = None  # pxr.Usd.Stage once wired
        self._stage_path: str | None = None
        self._layer_key: str | None = None  # root-layer registry key while a file is open
        self._reader_key: str | None = None  # readers-registry key (open --read-only)
        self.read_only = False  # `_mutating` blocks every stage edit while True
        self._prev_snapshot = None  # baseline for snapshot --diff (PR-2.5)
        self._active_cam: str | None = None  # active camera prim path (PR-6.2)
        self._selection: list[str] = []  # selected prim paths (PR-4.8)
        # visibility opinions isolate authored, path -> the edit layer's PRIOR
        # authored token (None = no spec existed). `isolate --restore` undoes
        # exactly these; save warns while any are live (round 8: leaked
        # isolation opinions gutted a saved deliverable — 19/8037 visible).
        self._isolation: dict[str, str | None] = {}
        # history.total_recorded at the last open/save: differing counts mean
        # unsaved edits (guards the same-file force-reload in _open).
        # _mutation_epoch counts every successful @_mutating command (plus the
        # mutating flags of read commands: validate --fix, sublayers --drop-dead)
        # — mutations that never touch the undo history still count.
        self._saved_op_serial = 0
        self._mutation_epoch = 0
        self._saved_mutation_epoch = 0
        # Monotonic observation state for external live viewers.
        self._viewer_revision = 0
        self._viewer_generation = 0
        self._viewer_line_geometry_capture = None
        self._viewer_line_geometry_publications: dict[str, dict] = {}
        # `space` query state: the ingested triangle soup, keyed by mutation epoch.
        # Derived from the stage, so it resets on open/new alongside the ref table.
        self._space_cache = None

    # -- lifecycle ---------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path, config: Config | None = None,
             name: str = "default") -> "Session":
        s = cls(config, name=name)
        s._open(str(path))
        return s

    def open_stage(self, file: str, read_only: bool = False,
                   force_reload: bool = False) -> Response:
        """Instance-level `open` command (PR-8.1) — the daemon routes here."""
        return self._open(file, read_only=read_only, force_reload=force_reload)

    def _invalidate_viewer_line_geometry(self) -> None:
        self._viewer_line_geometry_capture = None
        self._viewer_line_geometry_publications.clear()

    def _release_layer_key_locked(self) -> None:
        """Drop this session's root-layer registry entry (caller holds _ROOT_LAYER_LOCK)."""
        if self._layer_key:
            entry = _ROOT_LAYER_OWNERS.get(self._layer_key)
            if entry is not None and entry[1]() is self:
                del _ROOT_LAYER_OWNERS[self._layer_key]
        self._layer_key = None

    def _release_reader_key_locked(self) -> None:
        """Drop this session's readers-registry entry (caller holds _ROOT_LAYER_LOCK)."""
        if self._reader_key:
            readers = _ROOT_LAYER_READERS.get(self._reader_key)
            if readers is not None:
                ref = readers.get(self.name)
                if ref is not None and ref() is self:
                    del readers[self.name]
                if not readers:
                    del _ROOT_LAYER_READERS[self._reader_key]
        self._reader_key = None

    @staticmethod
    def _corrupt_open_error(path: str, exc: Exception) -> RuntimeError:
        """A clean, actionable error for a USD file whose bytes don't parse.

        Without this, a corrupt crate either fails deep inside crateFile.cpp on open
        or — worse — opens SILENTLY (prims list fine) and only explodes much later
        when something reads the damaged data ranges."""
        kind = ("USD crate error"
                if os.path.splitext(path)[1].lower() in (".usd", ".usdc", ".usdz")
                else "USD error")
        return RuntimeError(
            f"failed to open '{path}': file appears corrupt ({kind}: {exc}) — "
            "restore from a checkpoint or re-save it from a live session")

    def _open(self, path: str, read_only: bool = False,
              force_reload: bool = False) -> Response:  # PR-8.1
        from pxr import Sdf, Usd, UsdGeom

        key = os.path.realpath(path)
        with _ROOT_LAYER_LOCK:
            if not read_only:
                # Re-opening the file THIS session already holds, with edits not yet
                # saved, force-reloads from disk and silently discards them. That bit
                # hardest when a second agent shared the default session (its plain
                # `open` bypassed the cross-session guard below because the owner was
                # "itself"). Refuse with steering unless explicitly forced.
                if (self._layer_key == key and self._stage is not None
                        and (self.history.total_recorded != self._saved_op_serial
                             or self._mutation_epoch != self._saved_mutation_epoch)
                        and not force_reload):
                    raise RuntimeError(
                        f"this file is already open in session '{self.name}' with "
                        "unsaved edits — re-running `open` reloads from disk and "
                        "would DISCARD them. If you are another agent sharing this "
                        "session, use `open --read-only` or your own `--session "
                        "<name>` instead; to deliberately drop the unsaved edits, "
                        "re-run with `--force-reload`")
                owner = _root_layer_owner(key, self)
                if owner is not None:
                    # A discarded Session can linger in reference cycles (a caught
                    # exception's traceback holds frames that hold `self`) until the
                    # cyclic GC runs — collect once and re-check before declaring a
                    # conflict.
                    import gc
                    gc.collect()
                    owner = _root_layer_owner(key, self)
                if owner is not None:
                    # USD would hand us the SAME cached root SdfLayer that session
                    # already holds — reloading or editing it here would clobber that
                    # session's unsaved work. Refuse, loudly, instead of silently
                    # corrupting. (Parallel inspection is fine: `open --read-only`.)
                    raise RuntimeError(
                        f"file is open in session '{owner}' — USD sessions share the "
                        "layer cache; close it there first, open a copy, or use "
                        "`open --read-only` for parallel inspection")
                # USD caches SdfLayers by identifier, so a layer we edited in-memory
                # during a prior `open` of this same file would otherwise be reused with
                # those edits. `open` must reflect on-disk state, so revert any cached
                # layer for this path first. Safe: the guard above proves no OTHER live
                # session owns this layer, so the only unsaved edits a reload can
                # discard are this session's own — which is exactly what re-running
                # `open` means. (Read-only opens SKIP this: their whole purpose is to
                # observe the writer's live, unsaved layer.)
                cached = Sdf.Layer.Find(path)
                if cached is not None:
                    try:
                        if cached.Reload(force=True) is False:
                            raise RuntimeError("Reload(force=True) reported failure")
                    except Exception as exc:  # noqa: BLE001 — pxr raises Tf errors
                        raise self._corrupt_open_error(path, exc) from exc
            try:
                stage = Usd.Stage.Open(path)
            except Exception as exc:  # noqa: BLE001 — a truncated/corrupt crate raises
                # deep pxr errors; turn them into something an agent can act on
                if not os.path.exists(path):
                    raise FileNotFoundError(f"failed to open USD file: {path}") from exc
                raise self._corrupt_open_error(path, exc) from exc
            if not stage:
                raise FileNotFoundError(f"failed to open USD file: {path}")
            # A corrupt crate can open cleanly (its prim-index table intact) and
            # explode only when a damaged value/time-sample block is first read —
            # deep-probe the candidate FULLY before touching ANY session or
            # ownership state. The old order released the old stage's registry
            # entry and registered the candidate first: a failing probe then left
            # the old writable stage live but UNOWNED, so another session could
            # open the same file and clobber its unsaved edits. Probing under
            # `_ROOT_LAYER_LOCK` keeps validate + ownership swap atomic.
            try:
                self._deep_read_stage(stage)
            except Exception as exc:  # noqa: BLE001
                raise self._corrupt_open_error(path, exc) from exc
            self._release_layer_key_locked()  # opening a different stage frees the old one
            self._release_reader_key_locked()
            _prune_readers(key)
            if read_only:
                _ROOT_LAYER_READERS.setdefault(key, {})[self.name] = weakref.ref(self)
                self._reader_key = key
            else:
                _ROOT_LAYER_OWNERS[key] = (self.name, weakref.ref(self))
                self._layer_key = key
        # Reload resets refs (PR-1.4) and the undo history (state-model.md §1/§3).
        self.read_only = bool(read_only)
        self._stage = stage
        self._stage_path = path
        self.refs.reset()
        self._space_cache = None
        self.history.reset_baseline()
        self._saved_op_serial = self.history.total_recorded  # freshly opened = no unsaved edits
        self._saved_mutation_epoch = self._mutation_epoch
        self._prev_snapshot = None
        self._active_cam = next(iter(self._camera_paths()), None)  # adopt an existing camera
        self._selection = []
        self._isolation = {}
        self._viewer_revision += 1
        self._viewer_generation += 1
        self._invalidate_viewer_line_geometry()
        n = self._index_prims()
        up = UsdGeom.GetStageUpAxis(stage)
        summary = {"stage": path, "prims": n, "up_axis": str(up),
                   "active_camera": self._active_cam or "(none yet)"}
        issues = []
        if read_only:
            summary["read_only"] = True
            with _ROOT_LAYER_LOCK:
                owner = _root_layer_owner(key, self)
            if owner is not None:
                # a 4-minute false-alarm in round 8: a verifier read the writer's
                # live in-memory edits and concluded the on-disk file was
                # contaminated — say it up front
                issues.append(Issue(
                    "info",
                    f"sharing the LIVE layer of writer session '{owner}' — you "
                    "see its unsaved in-memory edits, NOT necessarily the "
                    "on-disk file"))
        return Response(command="open", summary=summary, issues=issues)

    # -- internals ---------------------------------------------------------------
    def _require_stage(self):
        if self._stage is None:
            raise RuntimeError("no stage open — run 'open <file.usd>' first")
        return self._stage

    def _index_prims(self) -> int:
        """Assign stable refs to meaningful prims (a minimal snapshot precursor, PR-1).

        Descends into native-instance proxies so geometry inside instanceable references is
        addressable (find/properties/material/bounds work on it). Proxies are read-only in
        USD; edit commands reject them with a pointer to the instance root (`_edit_path_of`).
        """
        from pxr import Usd, UsdGeom

        count = 0
        for prim in self._stage.Traverse(Usd.TraverseInstanceProxies()):
            if not prim.IsActive() or prim.GetPath().pathString == "/":
                continue
            type_name = prim.GetTypeName()
            if prim.IsA(UsdGeom.Camera):
                ns = "c"
            elif "Light" in str(type_name):
                ns = "l"
            elif str(type_name) == "Material":
                ns = "m"  # materials are addressable (`--bind @m1`) though not Imageable
            elif prim.IsA(UsdGeom.Imageable):
                ns = "n"
            else:
                continue
            self.refs.assign(ns, prim.GetPath().pathString, str(type_name), prim.GetName())
            count += 1
        return count

    def _camera_paths(self) -> list[str]:
        from usd_core.camera import list_cameras
        return list_cameras(self._stage) if self._stage else []

    def _default_cam_path(self) -> str:
        root = self._stage.GetDefaultPrim()
        base = root.GetPath().pathString if root and root.IsValid() else ""
        return f"{base}/usd_cam" if base and base != "/" else "/usd_cam"

    def _path_of(self, ref_or_path: str) -> str:
        """Resolve '@n1' to its SdfPath (checking liveness), or pass a literal path through."""
        if ref_or_path.startswith("@"):
            from usd_core.refs import StaleRefError
            entry = self.refs.resolve(ref_or_path)
            prim = self._stage.GetPrimAtPath(entry.path)
            # A deactivated prim is how `delete` removes things — treat it as stale (a
            # later undo reactivates it and the ref resolves again). Inactive prims report
            # IsValid()==True, so check IsActive() explicitly.
            if not prim.IsValid() or not prim.IsActive():
                raise StaleRefError(ref_or_path, entry.path)
            return entry.path
        from pxr import Sdf
        s = str(ref_or_path)
        if not Sdf.Path.IsValidPathString(s) or not s.startswith("/"):
            # Agents strip the '@' off refs and send bare numbers ("6346"): pxr used
            # to flood the daemon log with 'Ill-formed SdfPath' warnings while every
            # lookup silently resolved to a null prim. Catch it here, cleanly.
            hint = f" — did you mean '@n{s}'?" if s.isdigit() else ""
            raise ValueError(
                f"'{s}' is not a ref or an absolute prim path{hint} "
                "(refs look like '@n1'; prim paths start with '/')")
        # Absolute property / variant-selection paths ARE valid SdfPaths but resolve
        # to null PRIMS — they used to pass this gate and fail later with a bare
        # "no prim at …". Require a plain absolute prim path ('/' — the pseudo-root —
        # stays allowed for scoping commands like `snapshot /`).
        sdf_path = Sdf.Path(s)
        if sdf_path.ContainsPrimVariantSelection():
            raise ValueError(
                f"'{s}' contains a variant selection — commands take plain prim "
                "paths without {variant=…} selections (select the variant first, "
                "then address the prim path)")
        if sdf_path.IsPropertyPath():
            raise ValueError(
                f"'{s}' is a property path — commands take prim paths "
                f"(e.g. '{sdf_path.GetPrimPath()}', not '{s}'; use `set`/`properties` "
                "for attributes)")
        if not sdf_path.IsAbsoluteRootPath() and not sdf_path.IsPrimPath():
            raise ValueError(
                f"'{s}' is not a prim path (refs look like '@n1'; prim paths look "
                "like '/World/cube')")
        return s

    def _edit_path_of(self, ref_or_path: str) -> str:
        """`_path_of` for mutating commands: rejects instance proxies up front.

        USD forbids authoring inside a native instance, so without this guard an edit on a
        proxy dies with a cryptic Tf coding error. Reads (properties/material/bounds/find)
        keep using `_path_of` — proxies are fully inspectable."""
        path = self._path_of(ref_or_path)
        prim = self._stage.GetPrimAtPath(path)
        if prim.IsValid() and prim.IsInstanceProxy():
            raise ValueError(
                f"{ref_or_path} is inside a native instance (instance proxy) — USD cannot "
                "author edits there. Edit the instance root, or its prototype under the "
                "referenced source prim.")
        return path

    def _hidden_prims_under(self, paths: list[str]) -> int:
        """Count invisible subtree roots at/under `paths` that actually contain
        BOUNDABLE geometry (nested invisibles collapse into their topmost hidden
        ancestor). Bbox computation skips invisible geometry, so 'no renderable
        bounds' on a hidden prim reads as 'no geometry' and sends agents down the
        wrong diagnostic path. A hidden bare Xform/Scope with nothing renderable
        underneath must NOT count: `show`ing it can't produce bounds, so the hint
        would send the agent chasing a phantom."""
        from pxr import Usd, UsdGeom
        count = 0
        for p in paths:
            prim = self._stage.GetPrimAtPath(p)
            if not prim.IsValid():
                continue
            it = iter(Usd.PrimRange(prim, Usd.TraverseInstanceProxies()))
            for child in it:
                if child.IsA(UsdGeom.Imageable) and UsdGeom.Imageable(child) \
                        .ComputeVisibility() == UsdGeom.Tokens.invisible:
                    if any(d.IsA(UsdGeom.Boundable) for d in
                           Usd.PrimRange(child, Usd.TraverseInstanceProxies())):
                        count += 1
                    it.PruneChildren()
        return count

    def _target_range(self, refs: list[str] | None):
        """Union world bbox of the given refs/paths, or the whole scene if None."""
        from usd_core.spatial import combined_bbox, scene_bbox
        if refs:
            paths = [self._path_of(r) for r in refs]
            rng = combined_bbox(self._stage, paths)
            if rng is None:
                msg = f"no renderable bounds for {refs} (no geometry under those prims)"
                hidden = self._hidden_prims_under(paths)
                if hidden:
                    # the geometry EXISTS but is invisible — say so, or the agent
                    # burns renders chasing a phantom missing-geometry problem
                    msg = (f"no renderable bounds for {refs} — {hidden} prim(s) under "
                           "the focus target are hidden — `show` them first")
                raise RuntimeError(msg)
        else:
            rng = scene_bbox(self._stage)
            if rng is None:
                raise RuntimeError("scene has no renderable geometry to frame (empty stage)")
        return rng

    def ref(self, ref: str) -> RefEntry:
        """Resolve a `@n1`-style ref to its entry (raises StaleRefError if gone)."""
        return self.refs.resolve(ref)

    # -- perception (evidence, never decisions) ----------------------------------
    def resolve(self, refs: list[str] | None = None) -> Response:  # PR-1.3
        """Reveal full SdfPath(s) for ref(s); with no args, list the whole ref table.

        This translates a short ref back to the prim it names, and nothing else. A
        `@s` region does not name a prim — it is a measurement — so it resolves to
        the placeholder path it was minted under. Its geometry is in the answer that
        minted it, not here.
        """
        items = self.refs._by_ref.items() if not refs else \
            [(r, self.refs.resolve(r)) for r in refs]
        lines = [f"{r:6} {e.type:10} {e.path}" for r, e in items]
        return Response(command="resolve", summary={"refs": len(lines)},
                        data={"text": "\n".join(lines)})

    def snapshot(self, scope: str | None = None, type: list[str] | None = None,
                 depth: int | None = None, visible: bool = False, materials: bool = False,
                 bounds: bool = False, properties: bool = False, diff: bool = False,
                 structural: bool = False, since: str | None = None, **_ignored) -> Response:  # PR-2
        """Compact text tree of meaningful prims with their stable refs (PR-2).

        Implements scope / -t type / -d depth / -v visible / -m materials / -b bounds /
        -p properties, plus the structural diff (PR-2.5): `-D`/`--diff` compares against the
        previous snapshot in this session, `--since <checkpoint>` against a named checkpoint,
        and `--structural` renders attribute-level deltas.
        """
        from pxr import Usd, UsdGeom

        self._require_stage()
        n_indexed = self._index_prims()  # refresh stable refs (same path -> same @ref)

        if scope:
            root_path = self._path_of(scope)
        else:
            dp = self._stage.GetDefaultPrim()
            root_path = dp.GetPath().pathString if dp and dp.IsValid() else "/"
        root = self._stage.GetPrimAtPath(root_path)
        base = root_path.rstrip("/").count("/")
        type_filter = {t.lower() for t in type} if type else None

        # A whole-scene snapshot stays compact: native instances collapse to their root
        # (marked "instance"). Scoping *to* an instance opts into its proxy contents.
        into_instance = root.IsValid() and (root.IsInstance() or root.IsInstanceProxy())
        prim_iter = Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if into_instance \
            else Usd.PrimRange(root)

        lines, total, shown, instances, nodes = [], 0, 0, 0, []
        for prim in prim_iter:
            path = prim.GetPath().pathString
            if path == "/":
                continue
            total += 1
            ref = self.refs.ref_for_path(path)
            if ref is None:
                continue
            d = path.rstrip("/").count("/") - base
            if depth is not None and d > depth:
                continue
            tname = str(prim.GetTypeName())
            if type_filter and tname.lower() not in type_filter:
                continue
            if visible and prim.IsA(UsdGeom.Imageable) and \
                    UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
            annot, node = self._snapshot_annotations(
                path, materials=materials, bounds=bounds, properties=properties)
            marker = ""
            if prim.IsInstance():
                instances += 1
                marker = " (instance)"
                node["instance"] = True
            lines.append(f"{'  ' * max(0, d)}{ref} [{tname}] \"{prim.GetName()}\"{marker}{annot}")
            node.update({"ref": ref, "path": path, "type": tname, "name": prim.GetName()})
            nodes.append(node)
            shown += 1

        data = {"text": "\n".join(lines)}
        if materials or bounds or properties:
            data["nodes"] = nodes

        summary = {"stage": Path(self._stage_path).name if self._stage_path else "(mem)",
                   "prims": total, "refs": shown}
        if instances:
            summary["instances"] = instances
            summary["note"] = "instance contents are collapsed — 'snapshot <instance-ref>' expands one"

        # -- structural diff (PR-2.5) --------------------------------------------
        # Capture the current scene state so a later `-D` can diff against it (the
        # perceive → edit → perceive loop), then compare on request. The capture
        # is always the FULL stage (pseudo-root, unfiltered): baselines banked by scoped
        # or filtered snapshots must stay comparable, and edits outside the defaultPrim
        # (e.g. /Looks materials, prototype-source binds) must be diffable. The display
        # filters are applied at diff time, symmetrically to both sides.
        #
        # On LARGE stages the capture dominates snapshot cost (119k prims: 8.4s of
        # an 11.1s snapshot — and 30-212s in the round-8 sandbox, where it held the
        # root lock and starved sibling sessions into daemon-busy timeouts), taxing
        # every recon snapshot for a diff that may never be requested. Past
        # _SNAPDIFF_CAPTURE_MAX_PRIMS the baseline banks only when diffing is
        # actually in play (`-D`/`--since`); a banked baseline then persists, so
        # `-D` → edit → `-D` works exactly as before.
        from usd_core import snapdiff
        issues = []
        bank_baseline = bool(diff or since) or n_indexed <= _SNAPDIFF_CAPTURE_MAX_PRIMS
        if not bank_baseline:
            summary["diff_baseline"] = "not banked (large stage)"
            issues.append(Issue(
                "info", f"diff baseline not banked on this {n_indexed}-prim stage "
                        "(the capture is most of snapshot's cost at this scale) — "
                        "`snapshot -D` banks one and diffs against it from then on"))
            return Response(command="snapshot", summary=summary, data=data,
                            issues=issues)
        current_state = snapdiff.capture_state(self._stage, "/", refs=self.refs)
        if diff or since:
            try:
                if since:
                    baseline, warns = self._checkpoint_state(since)
                    issues += [Issue("warn", w) for w in warns]
                else:
                    baseline = self._prev_snapshot
            except Exception as exc:  # noqa: BLE001
                self._prev_snapshot = current_state
                return self._err("snapshot", exc)
            if baseline is None:  # `-D` with nothing recorded yet
                note = "no previous snapshot to diff against — baseline recorded"
                data["diff"] = {"note": note}
                summary["diff"] = "baseline"
                issues.append(Issue("info", note))
            else:
                result = snapdiff.diff_states(
                    baseline, current_state, scope=root_path if scope else None,
                    type_filter=type, depth=depth, visible_only=visible)
                data["diff"] = result
                data["tree"] = data["text"]  # keep the full tree available alongside the diff
                data["text"] = snapdiff.format_diff(result, structural=structural)
                summary.update({"against": since or "previous", **result["counts"]})
        self._prev_snapshot = current_state

        return Response(command="snapshot", summary=summary, data=data, issues=issues)

    def _snapshot_annotations(self, path: str, *, materials: bool, bounds: bool,
                              properties: bool) -> tuple[str, dict]:
        """Build the inline annotation suffix + a structured node for -m/-b/-p."""
        node: dict = {}
        bits = []
        if bounds:
            from usd_core.spatial import get_world_bbox
            b = get_world_bbox(self._stage, path)
            if b:
                node["bounds"] = {k: b[k] for k in ("min", "max", "center", "size")}
                bits.append("bbox=" + ",".join(f"{s:.2f}" for s in b["size"]))
        if materials:
            from usd_core.materials import bound_material, effective_materials_under
            m = bound_material(self._stage, path)
            node["material"] = m
            if m["bound_material_path"]:
                bits.append("mat=" + m["bound_material_path"].rsplit("/", 1)[-1])
            prim = self._stage.GetPrimAtPath(path)
            if prim.IsValid() and prim.IsInstance():
                # a collapsed instance still reveals the effective materials inside it
                agg = effective_materials_under(self._stage, path)
                node["materials_within"] = agg
                if agg["materials"]:
                    inner = ",".join(f"{p.rsplit('/', 1)[-1]}×{n}"
                                     for p, n in sorted(agg["materials"].items(),
                                                        key=lambda kv: -kv[1])[:4])
                    more = len(agg["materials"]) - 4
                    bits.append("mats={" + inner + (f",+{more}" if more > 0 else "") + "}")
        if properties:
            from usd_core.query import prim_properties
            p = prim_properties(self._stage, path, max_attrs=12)
            node["properties"] = p
            bits.append(f"attrs={p['attribute_count']}")
        return ((" {" + " ".join(bits) + "}") if bits else ""), node

    def _open_unshared(self, target: str | Path, *, tag: str,
                       isolate_clean: bool = False):
        """Open `target` without inheriting a layer another session already holds.

        USD caches SdfLayers by identifier, so a plain `Usd.Stage.Open(target)`
        hands back whatever layer is already open for that path — including a
        peer session's *unsaved* edits, and without any claim on that session's
        lock. Reading it reports in-memory state as though it were on disk;
        authoring into it mutates someone else's scene.

        Both are avoided by reading a side copy placed in the target's own
        directory, so relative sublayers, references and payloads keep resolving
        against the same anchor. Only the root layer is copied: a *sublayer* a
        peer holds dirty is still read through the cache.

        `isolate_clean=False` copies only when the cached layer has actually
        diverged from disk, which is all read-only inspection needs. Callers
        that author into the stage must pass True — a clean shared layer is
        still somebody else's layer.

        Returns `(stage, side_path_or_None)`; pass the second value to
        `_discard_side_copy` when the stage is no longer needed.
        """
        from pxr import Sdf, Usd

        path = Path(os.path.abspath(str(target)))
        if not path.exists():
            raise FileNotFoundError(str(path))
        cached = Sdf.Layer.Find(str(path))
        side = None
        if cached is not None and (isolate_clean or cached.dirty):
            side = path.with_name(
                f".{path.stem}.{tag}.{os.getpid()}.{uuid.uuid4().hex[:8]}{path.suffix}")
            shutil.copy2(path, side)
        try:
            stage = Usd.Stage.Open(str(side or path))
            if not stage:
                raise RuntimeError(f"could not open {path}")
            # The side copy isolates only the ROOT layer; composition still
            # resolves sublayers/references/payloads through USD's
            # process-global layer cache. A peer session's unsaved edits in any
            # layer of the composed closure would silently leak into this
            # stage (a verify could PASS against state that is not on disk),
            # so fail closed when the closure is not clean.
            root_identifiers = {str(side or path), str(path)}
            dirty_shared = sorted(
                layer.identifier
                for layer in stage.GetUsedLayers(includeClipLayers=True)
                if not layer.anonymous
                and layer.dirty
                and layer.identifier not in root_identifiers
            )
            if dirty_shared:
                shown = ", ".join(dirty_shared[:5])
                raise RuntimeError(
                    "cannot open an isolated stage: composed layers carry "
                    f"unsaved in-process edits from another session: {shown}"
                    + (" …" if len(dirty_shared) > 5 else "")
                    + " — save or discard those edits first")
        except Exception:
            self._discard_side_copy(side)
            raise
        return stage, side

    @staticmethod
    def _discard_side_copy(side: Path | None) -> None:
        """Remove an `_open_unshared` side copy. A temp file that cannot be
        unlinked must never fail an otherwise successful command."""
        if side is not None:
            with suppress(OSError):
                side.unlink(missing_ok=True)

    def _checkpoint_state(self, name: str) -> tuple[dict, list[str]]:
        """Capture a named checkpoint's scene state for `snapshot --since` (read-only).

        Opens the checkpoint stage on the side (never touching the live stage) and captures
        its full tree, so the diff aligns with the live snapshot's full capture (display
        filters are applied at diff time, to both sides).
        Returns (state, warnings): a non-`--full` checkpoint of a referenced asset can lose
        its external payloads/references when reopened from `.usd-cli/checkpoints/`, which would
        make everything look "added" — we detect that and warn rather than diff bad data.
        """
        import shutil

        from pxr import Sdf, Usd

        from usd_core import snapdiff

        target = self.checkpoint_path(name)
        if not target.exists():
            raise FileNotFoundError(f"no checkpoint '{name}'")
        # After `checkpoint load`, the live stage's root layer IS this file — USD caches
        # SdfLayers by identifier, so Usd.Stage.Open(target) would silently reuse the
        # live, edited in-memory layer and every `--since` diff would report "(no
        # changes)". Diff must see the *saved bytes*: read a side-copy in the same
        # directory (same dir keeps relative asset paths anchored) instead of reloading
        # the cached layer, which would destroy the user's unsaved edits.
        open_path, side = target, None
        cached = Sdf.Layer.Find(str(target))
        if cached is not None and cached.dirty:
            side = target.with_name(
                f".{target.stem}.since.{os.getpid()}.{uuid.uuid4().hex[:8]}{target.suffix}")
            shutil.copy2(target, side)
            open_path = side
        try:
            stage = Usd.Stage.Open(str(open_path))
            if not stage:
                raise RuntimeError(f"failed to open checkpoint '{name}'")
            warnings = []
            errors = stage.GetCompositionErrors() if hasattr(stage, "GetCompositionErrors") else []
            if errors:
                warnings.append(
                    f"checkpoint '{name}' has {len(errors)} unresolved reference(s); its stored "
                    "content is incomplete, so the diff may be unreliable — re-save with "
                    "'checkpoint save --full' for a self-contained, faithfully diffable checkpoint")
            state = snapdiff.capture_state(stage, "/", refs=self.refs)
        finally:
            if side is not None:
                stage = None  # release the side stage before removing its file
                side.unlink(missing_ok=True)
        return state, warnings

    def _read_project_render_section(self) -> dict:
        """The project config file's `[render]` table as currently on disk ({} when
        there is no project dir / file / section, or the TOML is half-written)."""
        project_dir = self.config.project_dir
        if project_dir is None:
            return {}
        from usd_core.config import CONFIG_NAME, _read_toml, state_dir_for
        try:
            section = _read_toml(
                state_dir_for(Path(project_dir)) / CONFIG_NAME).get("render")
        except Exception:  # noqa: BLE001 — a half-written config must not kill renders
            return {}
        return dict(section) if isinstance(section, dict) else {}

    def _fresh_render_section(self) -> dict:
        """This session's render settings re-resolved for ONE render call, mirroring
        `load_config`'s precedence for the hot-reloadable layers:

            session-baked base (defaults → global file → project file → OV_* env,
              resolved at session start; the global file itself is not re-read)
            → keys REMOVED from the project file since session start fall back to
              their built-in defaults
            → the project file's current [render] table
            → current USD_CLI_RENDER_* environment overrides (env stays ABOVE project)

        Agents edit `.usd-cli/config.toml` mid-session (e.g. bumping
        remote_max_upload_mb) and expect the NEXT render to honor it. The old code
        pasted the raw project TOML on top of the already env-resolved session
        settings — reversing the documented `environment > project` precedence (a
        project remote_url could override USD_CLI_RENDER_REMOTE_URL) — and keys deleted
        from the file persisted for the session's lifetime."""
        # A host may lock the daemon to its startup render configuration before
        # granting an untrusted child write access to the project directory. In
        # that mode, hot-reloading the child-writable config could replace an
        # attested endpoint/pool and turn the daemon into a network deputy.
        if os.environ.get("USD_CLI_LOCK_RENDER_CONFIG") == "1":
            return dict(self.config.render)

        from usd_core.config import DEFAULTS, _env_overrides
        render = dict(self.config.render)
        fresh = self._read_project_render_section()
        for key in self._render_project_baseline:
            if key in fresh or key not in render:
                continue  # still in the file (overlaid below), or never baked in
            if key in DEFAULTS["render"]:
                render[key] = DEFAULTS["render"][key]
            else:
                del render[key]
        render.update(fresh)
        env = _env_overrides().get("render")
        if isinstance(env, dict):
            render.update(env)
        return render

    def _render_config(self, renderer: str | None,
                       max_upload_mb: float | None = None) -> Config:
        """The Config handed to the render-backend factory for ONE render call.

        The render section is refreshed from the project config file on every call
        (see `_fresh_render_section`); a `--renderer` override rides the same
        ephemeral copy — the old scheme of mutating `self.config` and restoring it
        afterwards raced concurrent renders and could permanently corrupt the
        renderer setting."""
        render = self._fresh_render_section()
        if renderer:
            render["renderer"] = renderer
        if max_upload_mb is not None:
            # per-call override: the flagship large scenes exceed the safe default
            # cap by design, and hand-editing config.toml (then remembering to
            # revert) was the only escape hatch (round 7). 0 = unlimited.
            render["remote_max_upload_mb"] = float(max_upload_mb)
        if render == self.config.render:
            return self.config
        from dataclasses import replace
        return replace(self.config, render=render)

    def _session_dir(self, kind: str) -> Path:
        """Default output dir for `kind` ("renders" / "physics" / "checkpoints"),
        namespaced per session so concurrent sessions never overwrite each other's
        files. The "default" session keeps the historical un-namespaced layout exactly
        (back-compat for single-session users and existing tooling)."""
        base = self.config.state_dir / kind
        return base if self.name == "default" else base / self.name

    def _read_only_render_block(self, *, camera=None, focus=None, orbit=None,
                                exclude=None) -> Exception | None:
        """Why a render is NOT allowed in this read-only session, or None if it is.

        A read-only session shares the writer's LIVE root layer, so a render may
        not author ANYTHING on it — not the transient usd_cam a focus/auto-frame
        pass creates, not orbit cameras, not `--exclude` visibility opinions. The
        only permitted render is a no-authoring one: an EXISTING camera (explicit
        `--camera`, or an already-authored active camera) with none of those
        options."""
        from pxr import UsdGeom
        reason = None
        if focus:
            reason = "--focus reframes (authors) the render camera"
        elif exclude:
            reason = "--exclude temporarily authors visibility on the shared layer"
        elif orbit and int(orbit) >= 1:
            reason = "--orbit authors transient orbit cameras"
        else:
            try:
                cam_path = self._path_of(camera) if camera else self._active_cam
            except Exception:  # noqa: BLE001 — bad ref reads as "no camera"
                cam_path = None
            prim = self._stage.GetPrimAtPath(cam_path) if cam_path else None
            if not (prim and prim.IsValid() and prim.IsA(UsdGeom.Camera)):
                reason = "no existing camera to render from"
        if reason is None:
            return None
        return RuntimeError(
            f"read-only session cannot author render cameras ({reason}) — render "
            "from the writer session, or pass --camera <existing camera> without "
            "--focus/--orbit/--exclude")

    def render_probe(
        self,
        required_engine: str = "ovrtx",
        output_dir: str | None = None,
    ) -> Response:
        """Exercise the renderer in the daemon process, never in its client."""

        from usd_core.render.probe import probe_render_engine

        probe = probe_render_engine(
            self.config,
            required_engine=required_engine,
            output_dir=Path(output_dir).expanduser().resolve() if output_dir else None,
        )
        ready = probe.get("ready") is True
        return Response(
            command="render-probe",
            ok=ready,
            summary={
                "engine": probe.get("engine"),
                "transport": probe.get("transport"),
                "ready": ready,
            },
            data={"probe": probe},
            issues=(
                []
                if ready
                else [Issue("error", str(probe.get("error") or "renderer not ready"))]
            ),
        )

    def render(self, res=None, mode: str | None = None, focus: str | None = None,
               camera: str | None = None, output: str | None = None,
               orbit: int | None = None, elevation: float | None = None,
               exclude: list[str] | None = None, photoreal: bool = False, depth: bool = False,
               normals: bool = False, seg: bool = False, wireframe: bool = False,
               annotate: bool = False, inline: bool = False, renderer: str | None = None,
               against: str | None = None, diff: str | None = None,
               max_upload_mb: float | None = None, **_ignored) -> Response:  # PR-3
        """Multimodal render of the active camera (PR-3).

        Every invocation produces an OVRTX-backed RGB image through the configured
        backend. Depth / normals / segmentation / wireframe are CPU-computed auxiliary
        artifacts from geometry + camera intrinsics. `--annotate` overlays @ref labels on beauty;
        `--against` diffs beauty vs a prior image; `--inline` adds base64 to the payload.
        """
        from usd_core.render import make_backend

        self._require_stage()
        if self.read_only:
            err = self._read_only_render_block(camera=camera, focus=focus,
                                               orbit=orbit, exclude=exclude)
            if err:
                return self._err("render", err)
        target = [focus] if focus else None
        if focus:
            self.camera_fit([focus])
        elif camera:
            self.camera_use(camera)
        if not self._active_cam or not self._stage.GetPrimAtPath(self._active_cam).IsValid():
            self.camera_fit(None)  # create + frame the whole scene

        if isinstance(res, str):
            from usd_cli.parsing import resolution  # only if a raw "WxH" slipped through
            res = resolution(res)
        if not res:
            res = [int(x) for x in str(self.config.render.get("resolution", "1024x1024")).lower().split("x")]
        w, h = res
        render_config = self._render_config(renderer, max_upload_mb)
        mode = str(mode or render_config.render.get("mode") or "quality").strip().lower()
        if mode not in {"fast", "quality"}:
            raise ValueError(
                f"unknown render mode {mode!r} (expected 'fast' or 'quality')"
            )
        aov_mods = [m for m, on in (("depth", depth), ("normals", normals),
                                    ("segmentation", seg), ("wireframe", wireframe)) if on]
        # `-o foo.png` names a single output file when the render produces exactly one
        # image (single beauty view: no orbit, no extra AOVs, no annotate/against pass).
        # Otherwise it names a directory that holds the per-view/per-modality files.
        forced_stem = None
        single_file = bool(
            output and str(output).lower().endswith(".png")
            and not (orbit and orbit >= 1) and not aov_mods and not annotate and not against)
        if single_file:
            op = Path(output)
            out_dir, forced_stem = op.parent, op.stem
        else:
            out_dir = Path(output) if output else self._session_dir("renders")
        # `render` is always an OVRTX-backed operation. Analytic AOVs augment its
        # RGB output; they are not a CPU-only rendering backend.
        beauty_needed = True

        artifacts: list[Artifact] = []
        data: dict = {}
        issues: list[Issue] = []
        backend = make_backend(render_config)
        # The daemon lets a remote render release the per-root write lock once
        # packaging completes — valid ONLY when nothing touches the live stage
        # afterwards. Orbit renders author/remove cameras between packagings,
        # exclusion restore mutates the session layer on exit, and AOV/annotate/
        # against passes traverse the stage after the beauty upload — all of
        # those must keep the lock for the whole command.
        if orbit or exclude or aov_mods or annotate or against:
            from usd_core.render import remote as _remote_mod
            _remote_mod.STAGE_RELEASE_HOOK.set(None)
        # Capture the effective camera pose while the per-root write lock is
        # still guaranteed to be held: a plain beauty render may release the
        # lock right after packaging, and a peer session could then move the
        # camera before the summary is assembled. The recorded pose must be
        # the pose that was actually rendered.
        pose_at_dispatch = self._camera_pose() or {}
        with self._remove_output_dir_on_failure(out_dir), \
                self._excluded(exclude):  # hide excluded prims for all passes (PR-3)
            results = []
            if beauty_needed and orbit and orbit >= 1:
                results = self._render_orbit(backend, orbit, elevation, target,
                                             exclude, w, h, out_dir, mode)
            elif beauty_needed:
                stem = forced_stem or self._unique_name(out_dir, self._render_stem(
                    self._active_cam, w, h, mode, focus=focus, exclude=exclude))
                results = backend.render(self._stage, [self._active_cam], w, h, out_dir,
                                         mode=mode, names=[stem])
            if beauty_needed and not results:
                # A "successful" render with zero images is a lie the caller acts on
                # (benchmark agents shipped work believing a render had verified it).
                raise RuntimeError(
                    f"render produced no images (backend: {backend.name})")
            for result in results:
                if result.camera_world_transform is not None:
                    continue
                pose = (
                    self._camera_pose(result.camera) if orbit else pose_at_dispatch
                ) or {}
                result.camera_world_transform = pose.get("camera_world_transform")
                result.camera_pos = pose.get("camera_pos")
                result.camera_dir = pose.get("camera_dir")
            missing = [r.path for r in results if not Path(r.path).is_file()]
            if missing:
                raise RuntimeError(
                    f"render backend {backend.name} reported images that were never "
                    f"written: {missing[:3]}")
            artifacts += [Artifact(path=r.path, kind="render", label=f"rgb:{r.camera}")
                          for r in results]
            data["results"] = [
                {
                    "camera": r.camera,
                    "path": r.path,
                    "render_time_s": round(r.render_time, 3),
                    **(
                        {"ovrtx_render_mode": r.ovrtx_render_mode}
                        if r.ovrtx_render_mode is not None
                        else {}
                    ),
                    **(
                        {"ovrtx_num_sensor_updates": r.ovrtx_num_sensor_updates}
                        if r.ovrtx_num_sensor_updates is not None
                        else {}
                    ),
                    **(
                        {"active_aov": r.active_aov}
                        if r.active_aov is not None
                        else {}
                    ),
                    **(
                        {"renderer_identity": r.renderer_identity}
                        if r.renderer_identity is not None
                        else {}
                    ),
                    **{
                        key: getattr(r, key)
                        for key in (
                            "camera_world_transform",
                            "camera_pos",
                            "camera_dir",
                        )
                        if getattr(r, key) is not None
                    },
                }
                for r in results
            ]
            # advisory blank-frame heuristic (remote backend): surface the flagged
            # paths so the agent sees the warning in the response, not just the log
            suspects = [r.path for r in results if getattr(r, "blank_suspect", False)]
            if suspects:
                data["blank_suspect"] = suspects
            beauty_path = results[0].path if results else None

            # Analytic AOVs augment the OVRTX RGB render. Preserve the per-view
            # artifacts for an OVRTX orbit render as well.
            if aov_mods:
                if orbit and orbit >= 1:
                    artifacts += self._render_aov_orbit(
                        aov_mods, orbit, elevation, target, w, h, out_dir, data)
                else:
                    artifacts += self._render_aov_artifacts(aov_mods, w, h, out_dir, data)

            # annotated overlay on the beauty pass
            if annotate and beauty_path:
                from usd_core.imaging import annotate as _annotate
                ann = _annotate(beauty_path, self._annotate_labels(w, h))
                if ann:
                    artifacts.append(Artifact(path=ann, kind="render", label="annotated"))

            # render-diff vs a prior render: visual (beauty) and/or geometry (segmentation).
            # Visual diff misses transparent/glass moves; the geometry diff compares
            # material-independent segmentation footprints, so it catches them.
            if against:
                artifacts += self._render_diffs(
                    (diff or "visual").lower(), against, beauty_path,
                    {a.label: a.path for a in artifacts}, w, h, out_dir, data)

        if inline:
            from usd_core.imaging import to_base64
            data["inline"] = {a.label: to_base64(a.path) for a in artifacts}

        summary = {"camera": self._active_cam, "res": f"{w}x{h}",
                   "modalities": (["rgb"] if beauty_needed else []) + aov_mods
                   + (["annotated"] if annotate else []) + (["diff"] if against else [])}
        # effective camera pose (position + view direction) — after an orbit this
        # is the FIRST orbit view, which is also what AOV/annotate passes used.
        # Orbit renders keep the root lock for the whole command (the release
        # hook is disabled above), so reading the live pose here is race-free;
        # every other path reports the dispatch-time pose captured under the
        # lock so an early post-packaging release cannot let a peer session
        # change the recorded pose mid-flight.
        if orbit and orbit >= 1:
            summary.update(self._camera_pose() or {})
        else:
            summary.update(pose_at_dispatch)
        if against:
            # the quantitative verdict, in the one-line summary the agent actually
            # reads — the stats lived only in `data` and went unnoticed (round 5:
            # "--against emits no numeric score")
            for src_key, out_key in (("diff", "similarity"),
                                     ("diff_geometry", "similarity_geometry")):
                stats = data.get(src_key) or {}
                if "changed_fraction" in stats:
                    summary[out_key] = round(1.0 - stats["changed_fraction"], 5)
                    summary[f"{out_key}_mean_abs_diff"] = stats.get("mean_abs_diff")
        if beauty_needed:
            summary["backend"] = backend.name
            settings = {
                (
                    result.ovrtx_render_mode,
                    result.ovrtx_num_sensor_updates,
                    result.active_aov,
                )
                for result in results
            }
            if len(settings) == 1:
                render_mode, sensor_updates, active_aov = settings.pop()
                if (
                    render_mode is not None
                    and sensor_updates is not None
                    and active_aov is not None
                ):
                    summary.update(
                        {
                            "ovrtx_render_mode": render_mode,
                            "ovrtx_num_sensor_updates": sensor_updates,
                            "active_aov": active_aov,
                        }
                    )
        if exclude:
            summary["excluded"] = len(exclude)
        if orbit and orbit >= 1 and (beauty_needed or aov_mods):
            summary["orbit"] = orbit
        return Response(command="render", summary=summary, data=data,
                        artifacts=artifacts, issues=issues)

    @contextmanager
    def _remove_output_dir_on_failure(self, out_dir):
        """Remove an output directory this render CREATED when the render fails
        before writing anything into it — a failed `render --seg -o x.png` left
        an empty directory named `x.png` in the workspace (round 5). A dir that
        pre-existed, or that got any file written, is left alone (rmdir only
        removes empty dirs)."""
        pre_existing = Path(out_dir).exists()
        try:
            yield
        except Exception:
            if not pre_existing:
                try:
                    Path(out_dir).rmdir()
                except OSError:
                    pass
            raise

    @staticmethod
    def _sibling_modality(beauty_path: str, mod: str) -> str:
        """The path of a sibling AOV render: same dir/camera/res, modality token swapped.
        Beauty stems are `cam__WxH__<mode>`, AOV stems `cam__WxH__<modality>`."""
        p = Path(beauty_path)
        stem = p.stem.rsplit("__", 1)[0]  # drop the trailing mode/modality token
        return str(p.with_name(f"{stem}__{mod}.png"))

    def _render_diffs(self, mode_diff: str, against: str, beauty_path, by_label,
                      w, h, out_dir, data) -> list[Artifact]:
        """Compute the requested render-diff(s) vs a prior render. Returns diff Artifacts.

        visual   → abs-diff of the beauty pass (appearance changes).
        geometry → abs-diff of the segmentation pass (occupancy/footprint changes) — robust
                   to material, so it catches a moved glass/transparent object that the
                   visual diff would miss. The prior segmentation is the `--against` beauty's
                   sibling render (so render the baseline with --seg).
        """
        from usd_core.imaging import render_diff
        arts: list[Artifact] = []
        if mode_diff in ("visual", "both") and beauty_path:
            dp, stats = render_diff(against, beauty_path)
            data["diff"] = stats  # back-compat key (visual)
            arts.append(Artifact(path=dp, kind="render", label="diff"))
        if mode_diff in ("geometry", "both"):
            seg_now = by_label.get("segmentation")
            if not seg_now:  # render the current segmentation on the fly for the diff
                seg_arts = self._render_aov_artifacts(["segmentation"], w, h, out_dir, data)
                arts += seg_arts
                seg_now = next((a.path for a in seg_arts if a.label == "segmentation"), None)
            prior_seg = self._sibling_modality(against, "segmentation")
            if seg_now and Path(prior_seg).exists():
                dp, stats = render_diff(prior_seg, seg_now, threshold=8)
                data["diff_geometry"] = stats
                arts.append(Artifact(path=dp, kind="render", label="diff:geometry"))
            else:
                data["diff_geometry"] = {"error": "no prior segmentation found alongside "
                                         f"{against} — render the baseline with --seg"}
        return arts

    @staticmethod
    def _write_seg_legend(png_path: str, legend: dict) -> str:
        """Write the FULL ref -> color legend as a text file next to the
        segmentation image. The CLI shows 12 rows and elides the rest — a
        2,800-entry legend belongs on disk beside the pixels it decodes, not
        in the agent's context (and not behind a giant --json dump)."""
        txt = Path(png_path).with_suffix(".legend.txt")
        lines = [f"{ref}\trgb({c[0]},{c[1]},{c[2]})" for ref, c in legend.items()]
        txt.write_text("\n".join(lines) + "\n")
        return str(txt)

    def _render_aov_artifacts(self, mods, w, h, out_dir, data) -> list[Artifact]:
        """Rasterize analytic AOVs and write their auxiliary artifacts."""
        import numpy as np

        from usd_core import raster
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        imgs = raster.render_aovs(self._stage, self._active_cam, w, h, mods,
                                  self.refs.ref_for_path)
        arts = []
        cam = self._short(self._active_cam)
        for mod in mods:
            img = imgs.get(mod)
            if img is None:
                continue
            stem = self._unique_name(out_dir, f"{cam}__{w}x{h}__{mod}")
            path = str(out_dir / f"{stem}.png")
            img.save(path)
            arts.append(Artifact(path=path, kind="render", label=mod))
            if mod == "depth":
                linear_depth = imgs.get("linear_depth")
                if not isinstance(linear_depth, np.ndarray):
                    raise RuntimeError("depth AOV omitted metric linear-depth values")
                raw_path = str(out_dir / f"{stem}__linear_depth.npy")
                np.save(raw_path, linear_depth, allow_pickle=False)
                arts.append(
                    Artifact(
                        path=raw_path,
                        kind="auxiliary",
                        label="linear_depth",
                    )
                )
                data["linear_depth_unit"] = "meter"
            if mod == "segmentation" and imgs.get("legend"):
                lf = self._write_seg_legend(path, imgs["legend"])
                data["segmentation_legend_file"] = lf
                arts.append(Artifact(path=lf, kind="legend", label="segmentation legend"))
        if "legend" in imgs:
            data["segmentation_legend"] = imgs["legend"]
            data["segmentation_legend_paths"] = {
                label: (
                    self.refs.resolve(label).path if label.startswith("@") else label
                )
                for label in imgs["legend"]
            }
        return arts

    def _render_aov_orbit(self, mods, n, elevation, target, w, h, out_dir, data) -> list[Artifact]:
        """Rasterize AOV frames from N orbit viewpoints — the AOV counterpart of
        `_render_orbit`. Filenames encode azimuth/elevation so frames never overwrite."""
        import numpy as np

        from usd_core import raster
        from usd_core.camera import author_camera, clipping_range, fit_distance, orbit_position
        from pxr import UsdGeom
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rng = self._target_range(target)
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        size = list(rng.GetMax() - rng.GetMin())
        distance = fit_distance(size)
        up_y = UsdGeom.GetStageUpAxis(self._stage) == UsdGeom.Tokens.y
        el = 20.0 if elevation is None else float(elevation)
        cam = self._short(self._active_cam)
        arts: list[Artifact] = []
        first_pose = None
        for i in range(int(n)):
            az = 360.0 * i / n
            pos = orbit_position(center, distance, az, el, up_axis_y=up_y)
            near, far = clipping_range(list(rng.GetMin()), list(rng.GetMax()), pos, center)
            author_camera(self._stage, self._active_cam, pos, center, near=near, far=far)
            if first_pose is None:
                first_pose = (pos, center, near, far)
            imgs = raster.render_aovs(self._stage, self._active_cam, w, h, mods,
                                      self.refs.ref_for_path)
            azel = f"az{int(round(az)):03d}-el{int(round(el)):02d}"
            for mod in mods:
                img = imgs.get(mod)
                if img is None:
                    continue
                stem = self._unique_name(out_dir, f"{cam}__{w}x{h}__{azel}__{mod}")
                path = str(out_dir / f"{stem}.png")
                img.save(path)
                arts.append(Artifact(path=path, kind="render", label=f"{mod}:{azel}"))
                if mod == "depth":
                    linear_depth = imgs.get("linear_depth")
                    if not isinstance(linear_depth, np.ndarray):
                        raise RuntimeError(
                            "depth AOV omitted metric linear-depth values"
                        )
                    raw_path = str(out_dir / f"{stem}__linear_depth.npy")
                    np.save(raw_path, linear_depth, allow_pickle=False)
                    arts.append(
                        Artifact(
                            path=raw_path,
                            kind="auxiliary",
                            label=f"linear_depth:{azel}",
                        )
                    )
                    data["linear_depth_unit"] = "meter"
                if mod == "segmentation" and imgs.get("legend"):
                    lf = self._write_seg_legend(path, imgs["legend"])
                    data["segmentation_legend_file"] = lf
                    arts.append(Artifact(path=lf, kind="legend",
                                         label=f"segmentation legend:{azel}"))
            if "legend" in imgs:
                data["segmentation_legend"] = imgs["legend"]
                data["segmentation_legend_paths"] = {
                    label: (
                        self.refs.resolve(label).path
                        if label.startswith("@")
                        else label
                    )
                    for label in imgs["legend"]
                }
        if first_pose is not None:
            # Match _render_orbit: leave the camera at the first orbit pose so any
            # follow-up passes pair with the first frame, not an arbitrary last one.
            pos, center, near, far = first_pose
            author_camera(self._stage, self._active_cam, pos, center, near=near, far=far)
        return arts

    def _annotate_labels(self, w: int, h: int):
        """(@ref, (px_u, px_v)) screen anchors for prims actually VISIBLE from the active
        camera — computed from a segmentation pass so occluded prims get no label and each
        anchor sits on that prim's visible-pixel centroid (not a bbox center that may fall
        behind other geometry). Returns [] if the segmentation pass yields nothing.
        """
        import numpy as np
        from usd_core.raster import render_aovs
        try:
            aov = render_aovs(self._stage, self._active_cam, w, h,
                              ["segmentation"], self.refs.ref_for_path)
        except Exception:  # noqa: BLE001 — annotation is best-effort
            return []
        legend = aov.get("legend", {})          # visible-only: {ref: [r,g,b]}
        seg = np.asarray(aov["segmentation"])   # H×W×3
        labels = []
        for ref, color in legend.items():
            if not ref:
                continue
            mask = np.all(seg == np.array(color, dtype=seg.dtype), axis=2)
            ys, xs = np.nonzero(mask)
            if len(xs) < 4:  # too few visible pixels — effectively occluded
                continue
            labels.append((ref, (int(xs.mean()), int(ys.mean()))))
        return labels

    # -- time-sampled animation render (PR-3.6 frames) ---------------------------
    @staticmethod
    def _parse_frames(spec) -> list[int | float]:
        """'0:30' -> [0..30]; '0,5,10' -> [0,5,10]; '42' -> [42]."""

        def time_code(value: str) -> int | float:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"invalid frame time code: {value!r}")
            return int(parsed) if parsed.is_integer() else parsed

        s = str(spec).strip()
        if ":" in s:
            a, b = s.split(":")[:2]
            start = time_code(a)
            end = time_code(b)
            if end < start:
                return []
            if isinstance(start, int) and isinstance(end, int):
                return list(range(start, end + 1))
            frames: list[int | float] = [start]
            next_frame = float(start) + 1.0
            while next_frame < float(end):
                frames.append(time_code(str(next_frame)))
                next_frame += 1.0
            if frames[-1] != end:
                frames.append(end)
            return frames
        if "," in s:
            return sorted({time_code(x) for x in s.split(",") if x.strip()})
        return [time_code(s)]

    @staticmethod
    def _stage_time_range(stage):
        """Return the exact authored time-code range, or None.

        Prefers the authored startTimeCode/endTimeCode; else the min/max over every
        attribute's authored time samples. Used so render-frames plays a recording's real
        frames instead of frame numbers the caller guessed (which render a static tail)."""
        if stage.HasAuthoredTimeCodeRange():
            start = float(stage.GetStartTimeCode())
            end = float(stage.GetEndTimeCode())
            return (
                int(start) if start.is_integer() else start,
                int(end) if end.is_integer() else end,
            )
        lo = hi = None
        from pxr import Usd
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            for attr in prim.GetAttributes():
                ts = attr.GetTimeSamples()
                if ts:
                    lo = ts[0] if lo is None else min(lo, ts[0])
                    hi = ts[-1] if hi is None else max(hi, ts[-1])
        if lo is None or hi == lo:
            return None
        start = float(lo)
        end = float(hi)
        return (
            int(start) if start.is_integer() else start,
            int(end) if end.is_integer() else end,
        )

    def _playback_focus_range(self, stage, focus_path):
        """World bound of a playback subject across its authored animation."""

        from pxr import Gf, Usd, UsdGeom
        from usd_core.spatial import get_bbox

        authored_range = self._stage_time_range(stage)
        if authored_range is None:
            return get_bbox(stage, focus_path)
        start, end = authored_range
        if end - start > 255:
            # Keep long recordings bounded while covering both endpoints and
            # uniformly sampling the motion between them. Check the numeric
            # span before materializing integer frames so extreme authored
            # ranges cannot allocate an unbounded list.
            frames = sorted(
                {
                    start + index * (end - start) / 255
                    for index in range(256)
                }
            )
        else:
            frames = self._parse_frames(f"{start}:{end}")
        prim = stage.GetPrimAtPath(focus_path)
        if not prim or not prim.IsValid():
            return None
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
        )
        union = None
        for frame in frames:
            bbox_cache.SetTime(Usd.TimeCode(float(frame)))
            bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bound.IsEmpty():
                continue
            union = (
                Gf.Range3d(bound)
                if union is None
                else Gf.Range3d.GetUnion(union, bound)
            )
        return union

    def _ensure_camera_on(self, stage, camera=None, focus=None) -> str:
        """Pick/author a camera prim path on an arbitrary stage (for --scene playback)."""
        from usd_core.camera import (author_camera, clipping_range, fit_distance,
                                      list_cameras, orbit_position)
        from usd_core.spatial import scene_bbox
        from pxr import UsdGeom
        if camera and stage.GetPrimAtPath(camera).IsValid():
            return camera
        cams = list_cameras(stage)
        if cams and not focus:
            return cams[0]
        rng = (
            self._playback_focus_range(stage, self._path_of(focus))
            if focus
            else scene_bbox(stage)
        )
        if rng is None:
            raise RuntimeError("scene has no geometry to frame")
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        size = list(rng.GetMax() - rng.GetMin())
        up_y = UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y
        pos = orbit_position(center, fit_distance(size), 35.0, 20.0, up_axis_y=up_y)
        near, far = clipping_range(list(rng.GetMin()), list(rng.GetMax()), pos, center)
        # Playback cameras must stay in world space. Authoring below the default
        # prim makes the camera inherit the simulated body's time-sampled pose,
        # so it follows/rotates with the subject and double-applies its transform.
        # The playback stage is an isolated private copy and is discarded after
        # rendering, so a top-level managed camera cannot leak into the asset.
        cam = author_camera(stage, "/usd_cam", pos, center, near=near, far=far)
        return cam.GetPrim().GetPath().pathString

    def render_frames(self, frames=None, res=None, camera=None, focus=None, fps: float | None = None,
                      scene: str | None = None, output=None, mode: str | None = None,
                      animate: bool = True, renderer: str | None = None, **_ignored) -> Response:
        """Render a USD over a frame range at its time samples (PR-3.6 animation).

        Renders the live stage by default, or a `--scene <path>` (e.g. a physics
        `recording.usda`) without disturbing the session. Each frame is rendered at its USD
        time code via the backend; optionally assembles an animated GIF for inline playback.

        When `--frames` is omitted, the stage's authored animation range is played in full;
        an explicit range is clamped to it. Requesting frames past the authored range used
        to render the last static pose repeatedly (a "static" GIF).
        """
        from usd_core.render import make_backend
        if scene and self._stage is None:
            # `--scene FILE` names its own stage — demanding a prior `open` was a
            # pointless extra turn (round 8: fresh render sessions errored once)
            if not os.path.exists(os.path.abspath(scene)):
                return self._err("render-frames", FileNotFoundError(scene))
        else:
            self._require_stage()
        if self.read_only and (
                not scene or (self._stage_path and
                              os.path.realpath(scene) == os.path.realpath(self._stage_path))):
            # session-stage playback (or --scene aliasing the shared open file):
            # same no-authoring rule as `render` (a DIFFERENT --scene file plays on
            # a private side stage, so its transient camera never touches the
            # writer's layer)
            err = self._read_only_render_block(camera=camera, focus=focus)
            if err:
                return self._err("render-frames", err)
        from pxr import Usd
        notes = []
        stage = side_scene = None
        try:
            if scene:
                # `--scene` may name a file another session already has open.
                # `_ensure_camera_on` authors a transient camera, and a plain
                # `Usd.Stage.Open` would put it in that peer's cached layer —
                # editing its scene without its lock or history, and leaving the
                # camera to be persisted by its next save. Play a private copy.
                stage, side_scene = self._open_unshared(
                    scene, tag="playback", isolate_clean=True)
                cam_path = self._ensure_camera_on(stage, camera, focus)
            else:
                stage = self._stage
                if focus:
                    self.camera_fit([focus])
                elif camera:
                    self.camera_use(camera)
                if not self._active_cam or not self._stage.GetPrimAtPath(self._active_cam).IsValid():
                    self.camera_fit(None)
                cam_path = self._active_cam
            # Resolve the frame set against the stage's authored animation range so a
            # recording plays its real frames, not caller-guessed numbers.
            arange = self._stage_time_range(stage)
            if frames is None:
                frame_nums = (
                    self._parse_frames(f"{arange[0]}:{arange[1]}")
                    if arange
                    else self._parse_frames("0:24")
                )
            else:
                frame_nums = self._parse_frames(frames)
                if arange:
                    lo, hi = arange
                    kept = [f for f in frame_nums if lo <= f <= hi]
                    if len(kept) != len(frame_nums):
                        notes.append(f"clamped {len(frame_nums)}→{len(kept)} frames to the "
                                     f"stage's authored range [{lo}, {hi}] (frames outside it "
                                     "would render a static pose)")
                        frame_nums = kept or self._parse_frames(f"{lo}:{hi}")
            if not frame_nums:
                raise ValueError("no frames to render")
            # Default the GIF fps to the stage's own playback rate so timing matches reality.
            if fps is None:
                tcps = stage.GetTimeCodesPerSecond() if stage.HasAuthoredTimeCodeRange() else 0
                fps = float(tcps) if tcps else 24.0
            w, h = res or [int(x) for x in str(self.config.render.get("resolution", "1024x1024")).lower().split("x")]
            # --output foo.gif names the assembled GIF exactly; frames land in a sibling
            # dir. --output <dir> (or none) keeps frames there and auto-names the GIF.
            gif_out = None
            if output and str(output).lower().endswith(".gif"):
                gif_out = Path(output)
                out_dir = gif_out.parent / (gif_out.stem + "_frames")
            else:
                out_dir = Path(output) if output else self._session_dir("renders")
            render_config = self._render_config(renderer)
            mode = str(mode or render_config.render.get("mode") or "quality").strip().lower()
            if mode not in {"fast", "quality"}:
                raise ValueError(
                    f"unknown render mode {mode!r} (expected 'fast' or 'quality')"
                )
            backend = make_backend(render_config)
            cam_short = self._short(cam_path)
            paths = []
            renderer_identities = []
            suspects = []  # blank-frame heuristic flags (advisory, remote backend)
            def frame_token(value: int | float) -> str:
                numeric = float(value)
                if numeric.is_integer():
                    return f"{int(numeric):04d}"
                return f"{numeric:.15g}".replace("-", "m").replace(".", "p")

            stems = [
                self._unique_name(
                    out_dir,
                    f"{cam_short}__{w}x{h}__f{frame_token(f)}",
                )
                for f in frame_nums
            ]
            with self._remove_output_dir_on_failure(out_dir):  # no empty-dir litter
                if len(frame_nums) > 1 and hasattr(backend, "render_frames"):
                    # Batch: one scene upload + one ingest, all frames rendered server-side at
                    # their real time codes (a client loop over the single-frame remote path
                    # re-uploads per frame AND renders them all at the default time).
                    rrs = backend.render_frames(stage, cam_path, w, h, out_dir,
                                                [float(f) for f in frame_nums],
                                                mode=mode, names=stems)
                    paths = [rr.path for rr in rrs]
                    renderer_identities = [
                        getattr(rr, "renderer_identity", None) for rr in rrs
                    ]
                    suspects = [rr.path for rr in rrs
                                if getattr(rr, "blank_suspect", False)]
                else:
                    for f, stem in zip(frame_nums, stems):
                        rr = backend.render(stage, [cam_path], w, h, out_dir, mode=mode,
                                            names=[stem], frame=float(f))
                        paths.append(rr[0].path)
                        renderer_identities.append(
                            getattr(rr[0], "renderer_identity", None)
                        )
                        if getattr(rr[0], "blank_suspect", False):
                            suspects.append(rr[0].path)
                # Never report success with no images: a frame set that renders nothing (e.g.
                # all frames fell outside what the backend could produce) used to exit 0 with
                # an empty output dir. Fail loudly instead.
                if not [p for p in paths if p and Path(p).exists()]:
                    raise ValueError(
                        f"render-frames produced no images for frames {frame_nums} "
                        f"(camera {self._short(cam_path)}); nothing was written")
        except Exception as exc:  # noqa: BLE001
            return self._err("render-frames", exc)
        finally:
            # The render above is complete, so the private playback copy (when
            # one was needed) has served its purpose.
            stage = None
            self._discard_side_copy(side_scene)
        artifacts = [Artifact(path=p, kind="render", label=f"frame:{n}")
                     for n, p in zip(frame_nums, paths)]
        gif = None
        if animate and len(paths) > 1:
            from usd_core.imaging import assemble_gif
            gif_path = str(gif_out) if gif_out else str(
                Path(paths[0]).with_name(f"{cam_short}__anim.gif"))
            gif = assemble_gif(paths, gif_path, fps)
            artifacts.append(Artifact(path=gif, kind="render", label="animation"))
        return Response(command="render-frames",
                        summary={"camera": cam_path, "frames": len(frame_nums),
                                 "res": f"{w}x{h}", "fps": fps, "backend": backend.name,
                                 "scene": scene or "(session)"},
                        data={"frame_paths": paths, "animation": gif, "notes": notes,
                              "renderer_identities": renderer_identities,
                              **({"blank_suspect": suspects} if suspects else {}),
                              "text": f"rendered {len(paths)} frames"
                                      + (f" → {gif}" if gif else "")
                                      + ("".join(f"\nnote: {n}" for n in notes))},
                        artifacts=artifacts)

    @contextmanager
    def _excluded(self, refs: list[str] | None):
        """Temporarily hide the given prims (visibility=invisible), restoring after.

        Unresolvable refs FAIL the render: silently skipping them meant
        `render --exclude <typo>` produced an image with the prim still in it, and
        the caller shipped it believing the exclusion had happened.

        The WHOLE list is resolved, validated, and deduplicated BEFORE the first
        prim is hidden — validating while already hiding meant
        `--exclude valid --exclude bogus` raised after the valid prim was hidden,
        outside the try/finally, leaving it invisible forever; and a duplicated ref
        saved 'invisible' as its second prev-value and restored the prim hidden."""
        from pxr import UsdGeom
        targets, seen = [], set()
        for r in refs or []:
            path = self._path_of(r)
            prim = self._stage.GetPrimAtPath(path)
            if not prim.IsValid() or not prim.IsActive():
                raise ValueError(f"--exclude {r}: no prim at that path — refusing to "
                                 "render with a silently ignored exclusion")
            if not prim.IsA(UsdGeom.Imageable):
                raise ValueError(f"--exclude {r}: that prim is not Imageable, so it "
                                 "cannot be hidden — exclude a geometry/xform prim")
            if path in seen:
                continue
            seen.add(path)
            targets.append(prim)
        saved = []  # (visibility_attr, had_authored, prev_value)
        try:
            for prim in targets:
                img = UsdGeom.Imageable(prim)
                attr = img.GetVisibilityAttr()
                saved.append((attr, attr.HasAuthoredValue(), attr.Get()))
                img.MakeInvisible()
            yield
        finally:
            for attr, had, prev in reversed(saved):
                if had:
                    attr.Set(prev)
                else:
                    attr.Clear()  # back to inherited (visible)

    def _render_orbit(self, backend, n: int, elevation, target, exclude, w, h, out_dir, mode):
        """Render N viewpoints around the target (whole scene unless `target` given).

        Each frame's filename encodes its azimuth/elevation, so frames never overwrite.
        """
        from usd_core.camera import author_camera, clipping_range, fit_distance, orbit_position
        from pxr import UsdGeom
        rng = self._target_range(target)
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        size = list(rng.GetMax() - rng.GetMin())
        distance = fit_distance(size)
        up_y = UsdGeom.GetStageUpAxis(self._stage) == UsdGeom.Tokens.y
        el = 20.0 if elevation is None else float(elevation)
        focus = target[0] if target else None
        # Author one distinct camera prim per orbit angle and render them ALL in a single
        # backend call. The remote service ingests the scene (open + GPU acceleration-
        # structure build) once per call — re-uploading and re-ingesting a large scene for
        # each angle made an N-view orbit N× slower than it needs to be. up to 8 cameras/
        # request (the service cap); larger orbits fall back to chunks of 8.
        base = self._active_cam.rsplit("/", 1)[0] or ""
        cam_paths, stems, first_pose = [], [], None
        for i in range(int(n)):
            az = 360.0 * i / n
            pos = orbit_position(center, distance, az, el, up_axis_y=up_y)
            near, far = clipping_range(list(rng.GetMin()), list(rng.GetMax()), pos, center)
            cam_path = f"{base}/usd_cam_orbit_{i:02d}"
            author_camera(self._stage, cam_path, pos, center, near=near, far=far)
            cam_paths.append(cam_path)
            stems.append(self._unique_name(out_dir, self._render_stem(
                self._active_cam, w, h, mode, focus=focus, exclude=exclude, az=az, el=el)))
            if first_pose is None:
                first_pose = (pos, center, near, far)
        results = []
        for k in range(0, len(cam_paths), 8):  # service accepts <=8 cameras/request
            chunk_cams, chunk_stems = cam_paths[k:k + 8], stems[k:k + 8]
            results.extend(backend.render(self._stage, chunk_cams, w, h, out_dir,
                                          mode=mode, names=chunk_stems))
        for result in results:
            pose = self._camera_pose(result.camera) or {}
            result.camera_world_transform = pose.get("camera_world_transform")
            result.camera_pos = pose.get("camera_pos")
            result.camera_dir = pose.get("camera_dir")
        for cp in cam_paths:  # transient orbit cameras — don't leave them on the stage
            if cp != self._active_cam:
                self._stage.RemovePrim(cp)
        if first_pose is not None:
            # Leave the camera at the FIRST orbit pose: the AOV / --annotate / --against
            # passes that follow pair with results[0], and projecting labels or AOVs from
            # the last orbit pose silently produced empty overlays and mismatched AOVs.
            pos, center, near, far = first_pose
            author_camera(self._stage, self._active_cam, pos, center, near=near, far=far)
        return results

    @staticmethod
    def _short(x: str) -> str:
        return x.lstrip("@").strip("/").replace("/", "_") or "camera"

    def _render_stem(self, cam, w, h, mode, focus=None, exclude=None, az=None, el=None) -> str:
        """A filename stem encoding the render params, so distinct renders don't collide."""
        parts = [self._short(cam), f"{w}x{h}", mode]
        if focus:
            parts.append("focus-" + self._short(focus))
        if exclude:
            joined = "-".join(self._short(r) for r in exclude)
            if len(joined) > 40:
                # Hundreds of --exclude values inlined verbatim overflowed the OS
                # filename limit ([Errno 74] Bad message) — encode long lists as
                # excl-<count>x-<8char-sha1> instead (still deterministic per set).
                import hashlib
                joined = f"{len(exclude)}x-{hashlib.sha1(joined.encode()).hexdigest()[:8]}"
            parts.append("excl-" + joined)
        if az is not None:
            parts.append(f"az{int(round(az)):03d}-el{int(round(el)):02d}")
        return "__".join(parts)

    def _unique_name(self, out_dir, stem: str) -> str:
        """Return a stem whose .png doesn't exist yet in out_dir (append _2, _3, … if needed)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if not (out_dir / f"{stem}.png").exists():
            return stem
        i = 2
        while (out_dir / f"{stem}_{i}.png").exists():
            i += 1
        return f"{stem}_{i}"

    # -- camera (PR-6) -----------------------------------------------------------
    def _set_active(self, cam) -> None:
        self._active_cam = cam.GetPrim().GetPath().pathString
        # a freshly authored managed camera should be addressable by ref too
        if self.refs.ref_for_path(self._active_cam) is None:
            self.refs.assign("c", self._active_cam, "Camera", cam.GetPrim().GetName())

    def _cam_position(self) -> tuple | None:
        from pxr import UsdGeom
        if not self._active_cam:
            return None
        prim = self._stage.GetPrimAtPath(self._active_cam)
        if not prim.IsValid():
            return None
        m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        t = m.ExtractTranslation()
        return (t[0], t[1], t[2])

    def _camera_pose(self, camera_path: str | None = None) -> dict | None:
        """World pose of the active camera — position + unit view direction (-Z).

        Echoed in every render summary: a field run (2026-07-20) rendered what
        the agent believed was a fresh bottom view of /World_usd_cam and got the
        previous top-oblique framing back — with the effective pose in the
        response, a stale/unchanged camera is visible in text, not just in the
        pixels. Best-effort: a pose failure must never sink a finished render."""
        from pxr import Gf, UsdGeom
        try:
            resolved_camera_path = camera_path or self._active_cam
            if not resolved_camera_path:
                return None
            prim = self._stage.GetPrimAtPath(resolved_camera_path)
            if not prim.IsValid():
                return None
            m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
            t = m.ExtractTranslation()
            d = m.TransformDir(Gf.Vec3d(0, 0, -1)).GetNormalized()
            world_transform = [
                [float(m[row][column]) for column in range(4)]
                for row in range(4)
            ]
            return {
                "camera": resolved_camera_path,
                "camera_world_transform": world_transform,
                "camera_pos": [round(float(t[i]), 3) for i in range(3)],
                "camera_dir": [round(float(d[i]), 3) for i in range(3)],
            }
        except Exception:  # noqa: BLE001
            return None

    def camera_list(self) -> Response:  # PR-6.1
        cams = self._camera_paths()
        return Response(command="camera.list",
                        summary={"count": len(cams), "active": self._active_cam},
                        data={"cameras": cams})

    def camera_use(self, ref: str) -> Response:  # PR-6.2
        from pxr import UsdGeom
        path = self._path_of(ref)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
            raise ValueError(f"{ref} is not a camera")
        self._active_cam = path
        return Response(command="camera.use", summary={"active": path})

    @_mutating
    def camera_look_at(self, ref: str) -> Response:  # PR-6.3
        from usd_core.camera import author_camera, fit_distance
        rng = self._target_range([ref])
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        pos = self._cam_position()
        if pos is None:  # no camera yet — place one a sensible distance back
            from usd_core.camera import orbit_position
            size = list(rng.GetMax() - rng.GetMin())
            pos = orbit_position(center, fit_distance(size), 35, 20)
        cam = author_camera(self._stage, self._active_cam or self._default_cam_path(), pos, center)
        self._set_active(cam)
        return Response(command="camera.look-at", summary={"camera": self._active_cam, "target": ref})

    @_mutating
    def camera_orbit(self, ref: str | None = None, az: float | None = None,
                     el: float | None = None, dist: float | None = None) -> Response:  # PR-6.4
        from usd_core.camera import author_camera, clipping_range, fit_distance, orbit_position
        from pxr import UsdGeom
        rng = self._target_range([ref] if ref else None)
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        size = list(rng.GetMax() - rng.GetMin())
        distance = dist if dist is not None else fit_distance(size)
        up_y = UsdGeom.GetStageUpAxis(self._stage) == UsdGeom.Tokens.y
        pos = orbit_position(center, distance, az or 0.0, el or 0.0, up_axis_y=up_y)
        near, far = clipping_range(list(rng.GetMin()), list(rng.GetMax()), pos, center)
        cam = author_camera(self._stage, self._active_cam or self._default_cam_path(),
                            pos, center, near=near, far=far)
        self._set_active(cam)
        return Response(
            command="camera.orbit",
            summary={
                "az": az or 0.0,
                "el": el or 0.0,
                "distance": round(distance, 3),
                **(self._camera_pose() or {"camera": self._active_cam}),
            },
        )

    @_mutating
    def camera_fit(self, refs: list[str] | None = None) -> Response:  # PR-6.5
        from usd_core.camera import author_camera, clipping_range, fit_distance, orbit_position
        from pxr import UsdGeom
        rng = self._target_range(refs)
        center = list((rng.GetMin() + rng.GetMax()) / 2.0)
        size = list(rng.GetMax() - rng.GetMin())
        distance = fit_distance(size)
        up_y = UsdGeom.GetStageUpAxis(self._stage) == UsdGeom.Tokens.y
        pos = orbit_position(center, distance, 35.0, 20.0, up_axis_y=up_y)  # pleasant 3/4 view
        near, far = clipping_range(list(rng.GetMin()), list(rng.GetMax()), pos, center)
        cam = author_camera(self._stage, self._active_cam or self._default_cam_path(),
                            pos, center, near=near, far=far)
        self._set_active(cam)
        return Response(command="camera.fit",
                        summary={"camera": self._active_cam,
                                 "framed": refs or "scene", "distance": round(distance, 3)})

    def _as_point(self, value, default):
        """Interpret a camera arg as a world point: 'x,y,z' coords, a [x,y,z] list, or a
        prim ref/path (→ its world-space bbox center)."""
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return tuple(float(v) for v in value)
        s = str(value).strip()
        if "," in s:
            try:
                parts = [float(x) for x in s.split(",")]
            except ValueError:
                parts = None
            if parts and len(parts) == 3:
                return tuple(parts)
        # treat as a prim ref/path and frame its center
        rng = self._target_range([s])
        return tuple((rng.GetMin() + rng.GetMax()) / 2.0)

    @_mutating
    def camera_create(
        self,
        name: str | None = None,
        at=None,
        look_at=None,
        focal: float | None = None,
        aperture: float | None = None,
    ) -> Response:  # PR-6.8
        from usd_core.camera import author_camera, camera_lens_mm_to_raw

        if name and name.startswith(("usd_cam", "ov_cam", "dsc_cam")):
            # the usd_cam*/ov_cam*/dsc_cam* namespace is reserved for the tool's transient
            # cameras: author_camera tags any such name as managed, so a USER
            # camera named this way would be silently stripped from every save
            # and render bundle
            return self._err(
                "camera.create",
                ValueError(
                    f"camera name '{name}' uses the reserved usd_cam*/ov_cam*/dsc_cam* "
                    "namespace (tool-managed transient cameras) — pick another name"
                ),
            )
        path = (
            self._default_cam_path()
            if not name
            else f"{self._default_cam_path().rsplit('/', 1)[0]}/{name}"
        )
        try:
            pos = self._as_point(at, (0.0, 0.0, 5.0))
            target = self._as_point(look_at, (0.0, 0.0, 0.0))
            focal_length = 60.0 if focal is None else float(focal)
            camera_aperture = 36.0 if aperture is None else float(aperture)
            if focal_length <= 0 or camera_aperture <= 0:
                raise ValueError("camera focal length and aperture must be positive")
            focal_raw = camera_lens_mm_to_raw(self._stage, focal_length)
            aperture_raw = camera_lens_mm_to_raw(self._stage, camera_aperture)
        except Exception as exc:  # noqa: BLE001
            return self._err("camera.create", exc)
        cam = author_camera(
            self._stage,
            path,
            pos,
            target,
            focal=focal_raw,
            h_ap=aperture_raw,
            v_ap=aperture_raw,
        )
        self._set_active(cam)
        return Response(command="camera.create", summary={"camera": self._active_cam})

    def camera_coverage(
        self,
        scope: str | None = None,
        cameras: list[str] | None = None,
        target: float = 0.95,
        per_cell: int = 1,
        grid: int = 32,
        cell_size: float | None = None,
        device: str = "cpu",
        output: str | None = None,
    ) -> Response:
        """Quantitatively audit an existing camera rig without changing the stage."""

        from usd_core.camera_analysis.cancellation import (
            CameraAnalysisCancelled,
            check_cancelled,
        )
        from usd_core.camera_analysis.coverage import (
            MAX_CAMERAS,
            evaluate_coverage,
            validate_grid_workload,
        )
        from usd_core.camera_analysis.contracts import SceneAnalysisPolicy
        from usd_core.camera_analysis.newton_backend import (
            NewtonVisibilityBackend,
            require_qualified_backend_versions,
        )
        from usd_core.camera_analysis.rig_export import publish_json_document
        from usd_core.camera_analysis.scene import (
            analysis_bounds,
            build_scene_analysis_ir,
        )
        from pxr import UsdGeom

        self._require_stage()
        publication_target = None
        try:
            check_cancelled()
            require_qualified_backend_versions()
            check_cancelled()
            scope_path = (
                self._path_of(scope)
                if scope
                else (
                    self._stage.GetDefaultPrim().GetPath().pathString
                    if self._stage.GetDefaultPrim().IsValid()
                    else "/"
                )
            )
            scope_prim = self._stage.GetPrimAtPath(scope_path)
            scene = build_scene_analysis_ir(
                self._stage,
                policy=SceneAnalysisPolicy(
                    scope_paths=(scope_path,),
                    floor_paths=(scope_path,) if scope_prim.IsA(UsdGeom.Gprim) else (),
                ),
            )
            bounds = analysis_bounds(scene, roots=(scope_path,))
            camera_paths = None
            if cameras:
                from pxr import Sdf, Usd

                known_cameras = {camera.prim_path for camera in scene.cameras}
                camera_paths = []
                for item in cameras:
                    selected_path = self._path_of(item)
                    if selected_path in known_cameras:
                        camera_paths.append(selected_path)
                        continue
                    selected_prim = self._stage.GetPrimAtPath(Sdf.Path(selected_path))
                    descendants = (
                        sorted(
                            prim.GetPath().pathString
                            for prim in Usd.PrimRange(
                                selected_prim, Usd.TraverseInstanceProxies()
                            )
                            if prim.IsA(UsdGeom.Camera)
                            and prim.GetPath().pathString in known_cameras
                        )
                        if selected_prim.IsValid()
                        else []
                    )
                    if not descendants:
                        raise ValueError(
                            "camera selection is neither an eligible camera nor a "
                            f"subtree containing one: {selected_path}"
                        )
                    camera_paths.extend(descendants)
            camera_count = (
                len(camera_paths)
                if camera_paths is not None
                else sum(camera.projection == "perspective" for camera in scene.cameras)
            )
            if camera_count > MAX_CAMERAS:
                raise ValueError(
                    f"camera coverage accepts at most {MAX_CAMERAS} cameras"
                )
            validate_grid_workload(
                bounds,
                grid=int(grid),
                cell_size_m=cell_size,
                view_count=camera_count,
                operation="coverage evaluation",
            )
            backend = NewtonVisibilityBackend(scene, device=device)
            evaluation = evaluate_coverage(
                scene,
                backend,
                bounds,
                scope_path=scope_path,
                camera_paths=camera_paths,
                target_coverage=float(target),
                per_cell=int(per_cell),
                grid=int(grid),
                cell_size_m=cell_size,
            )
            report = evaluation.report
            artifacts: list[Artifact] = []
            if output:
                check_cancelled()
                publication_target = self._prepare_camera_publication(output)
                with self._camera_publish_lock(publication_target):
                    check_cancelled()
                    path, digest = publish_json_document(report, publication_target)
                report["report_artifact"] = {"path": path, "digest": digest}
                artifacts.append(
                    Artifact(path=path, kind="summary", label="camera coverage")
                )
        except CameraAnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._err("camera.coverage", exc)
        finally:
            if publication_target is not None:
                publication_target.close()
        return Response(
            command="camera.coverage",
            summary={
                "coverage": round(report["coverage_fraction"], 6),
                "target": report["target_coverage"],
                "passed": report["passed"],
                "cameras": len(report["cameras"]),
                "uncovered_cells": report["uncovered_cells"],
            },
            data=report,
            artifacts=artifacts,
        )

    def camera_place(
        self,
        method: str,
        scope: str | None = None,
        target: str | None = None,
        target_coverage: float | None = None,
        per_cell: int | None = None,
        max_cameras: int | None = None,
        cameras: int | None = None,
        grid: int | None = None,
        cell_size: float | None = None,
        patch_size: float | None = None,
        candidates: int = 64,
        height: float | None = None,
        min_height: float | None = None,
        max_height: float | None = None,
        standoff: float | None = None,
        minimum_gain: float | None = None,
        min_look_down: float | None = None,
        max_look_down: float | None = None,
        yaw_ranges: str | None = None,
        occlusion_threshold: float | None = None,
        min_distance: float | None = None,
        max_distance: float | None = None,
        min_x: float | None = None,
        max_x: float | None = None,
        min_y: float | None = None,
        max_y: float | None = None,
        allow_fewer: bool = False,
        seed: int = 0,
        focal: float | None = None,
        aperture: float = 36.0,
        device: str = "cpu",
        preview: bool = False,
        author_under: str | None = None,
        on_existing: str = "error",
        output: str | None = None,
    ) -> Response:
        """Preview or atomically author a deterministic multi-camera rig."""

        from usd_core.camera_analysis.authoring import (
            author_camera_rig,
            camera_rig_destination,
        )
        from usd_core.camera_analysis.cancellation import (
            CameraAnalysisCancelled,
            check_cancelled,
        )
        from usd_core.camera_analysis.contracts import SceneAnalysisPolicy
        from usd_core.camera_analysis.look_at import (
            LookAtConfig,
            place_cameras_look_at,
        )
        from usd_core.camera_analysis.newton_backend import (
            NewtonVisibilityBackend,
            require_qualified_backend_versions,
        )
        from usd_core.camera_analysis.placement import (
            place_max_coverage,
            validate_height_envelope,
            validate_look_at_workload,
            validate_look_down_envelope,
            validate_placement_workload,
            validate_xy_bounds,
        )
        from usd_core.camera_analysis.rig_export import publish_json_document
        from usd_core.camera_analysis.scene import (
            analysis_bounds,
            build_scene_analysis_ir,
        )

        self._require_stage()
        publication_target = None
        try:
            check_cancelled()
            method = str(method).strip().lower()
            if method not in {"max_coverage", "look_at"}:
                raise ValueError("method must be one of: max_coverage, look_at")
            on_existing = str(on_existing).strip().lower()
            if on_existing not in {"error", "replace", "append"}:
                raise ValueError("--on-existing must be one of: error, replace, append")
            if not author_under and on_existing != "error":
                raise ValueError("--on-existing is valid only with --author-under")
            if preview and author_under:
                raise ValueError("--preview and --author-under are mutually exclusive")
            if author_under and self.read_only:
                raise RuntimeError(
                    f"session '{self.name}' is read-only — camera rig authoring is blocked"
                )
            if author_under:
                author_under = self._edit_path_of(author_under)
            if method == "max_coverage" and target is not None:
                raise ValueError(
                    "--target is valid only with --method look_at; use --scope"
                )
            if method == "max_coverage" and (yaw_ranges is not None or allow_fewer):
                raise ValueError(
                    "yaw ranges and --allow-fewer are valid only with --method look_at"
                )
            if method == "max_coverage" and any(
                value is not None
                for value in (cameras, occlusion_threshold, min_distance, max_distance)
            ):
                raise ValueError(
                    "--cameras, --occlusion-threshold, and distance limits are valid "
                    "only with --method look_at"
                )
            if method == "max_coverage" and any(
                value is not None
                for value in (min_height, max_height, min_x, max_x, min_y, max_y)
            ):
                raise ValueError(
                    "height envelopes and XY bounds are valid only with "
                    "--method look_at; max_coverage uses --height"
                )
            if method == "look_at" and scope is not None:
                raise ValueError(
                    "--scope is valid only with --method max_coverage; use --target"
                )
            if method == "look_at" and any(
                value is not None
                for value in (
                    target_coverage,
                    per_cell,
                    max_cameras,
                    grid,
                    cell_size,
                    patch_size,
                    standoff,
                    minimum_gain,
                )
            ):
                raise ValueError(
                    "coverage target, grid, patch size, standoff, minimum gain, and "
                    "--max-cameras are valid only with --method max_coverage"
                )

            if cell_size is not None and patch_size is not None:
                raise ValueError(
                    "--cell-size and --patch-size are aliases; provide only one"
                )
            effective_patch_size = patch_size if patch_size is not None else cell_size
            if (
                method == "max_coverage"
                and grid is not None
                and effective_patch_size is not None
            ):
                raise ValueError("--grid and --patch-size are mutually exclusive")
            if effective_patch_size is not None and (
                not math.isfinite(float(effective_patch_size))
                or float(effective_patch_size) <= 0.0
            ):
                raise ValueError("patch size must be finite and positive")
            if not 4 <= int(candidates) <= 512:
                raise ValueError("candidate count must be between 4 and 512")
            look_down_envelope = validate_look_down_envelope(
                min_look_down, max_look_down
            )
            xy_values = (min_x, max_x, min_y, max_y)
            if any(value is not None for value in xy_values) and not all(
                value is not None for value in xy_values
            ):
                raise ValueError(
                    "XY bounds require all of --min-x, --max-x, --min-y, and --max-y"
                )
            xy_bounds = (
                validate_xy_bounds(
                    ((float(min_x), float(max_x)), (float(min_y), float(max_y)))
                )
                if all(value is not None for value in xy_values)
                else None
            )
            height_envelope = validate_height_envelope(min_height, max_height)
            require_qualified_backend_versions()
            check_cancelled()

            if method == "max_coverage":
                from pxr import UsdGeom

                scope_path = (
                    self._path_of(scope)
                    if scope
                    else (
                        self._stage.GetDefaultPrim().GetPath().pathString
                        if self._stage.GetDefaultPrim().IsValid()
                        else "/"
                    )
                )
                scope_prim = self._stage.GetPrimAtPath(scope_path)
                policy = SceneAnalysisPolicy(
                    scope_paths=(scope_path,),
                    floor_paths=(scope_path,)
                    if scope_prim.IsValid() and scope_prim.IsA(UsdGeom.Gprim)
                    else (),
                )
            else:
                if target is None:
                    raise ValueError("--target is required with --method look_at")
                target_path = self._path_of(target)
                policy = SceneAnalysisPolicy(
                    scope_paths=(target_path,), target_paths=(target_path,)
                )
            scene = build_scene_analysis_ir(self._stage, policy=policy)
            if method == "max_coverage":
                bounds = analysis_bounds(scene, roots=(scope_path,))
                validate_placement_workload(
                    bounds,
                    grid=int(32 if grid is None else grid),
                    patch_size_m=effective_patch_size,
                    candidate_count=int(candidates),
                )
                backend = NewtonVisibilityBackend(scene, device=device)
                evaluation = place_max_coverage(
                    scene,
                    backend,
                    bounds,
                    scope_path=scope_path,
                    target_coverage=float(
                        0.95 if target_coverage is None else target_coverage
                    ),
                    per_cell=int(1 if per_cell is None else per_cell),
                    max_cameras=int(8 if max_cameras is None else max_cameras),
                    grid=int(32 if grid is None else grid),
                    patch_size_m=effective_patch_size,
                    candidate_count=int(candidates),
                    height_m=float(3.0 if height is None else height),
                    standoff_m=float(0.5 if standoff is None else standoff),
                    minimum_gain=float(0.0 if minimum_gain is None else minimum_gain),
                    min_look_down_deg=look_down_envelope[0],
                    max_look_down_deg=look_down_envelope[1],
                    seed=int(seed),
                    focal_length_mm=float(focal if focal is not None else 18.0),
                    aperture_mm=float(aperture),
                )
            else:
                validate_look_at_workload(
                    scene,
                    target_path,
                    candidate_count=int(candidates),
                )
                bounds = analysis_bounds(scene, roots=(target_path,))
                backend = NewtonVisibilityBackend(scene, device=device)
                evaluation = place_cameras_look_at(
                    scene,
                    backend,
                    bounds,
                    config=LookAtConfig(
                        target_path=target_path,
                        camera_count=int(4 if cameras is None else cameras),
                        yaw_ranges=yaw_ranges,
                        occlusion_threshold=float(
                            0.4 if occlusion_threshold is None else occlusion_threshold
                        ),
                        min_distance_m=min_distance,
                        max_distance_m=max_distance,
                        height_offset_m=float(0.0 if height is None else height),
                        min_height_m=height_envelope[0],
                        max_height_m=height_envelope[1],
                        min_look_down_deg=look_down_envelope[0],
                        max_look_down_deg=look_down_envelope[1],
                        xy_bounds_m=xy_bounds,
                        candidate_count=int(candidates),
                        allow_fewer=bool(allow_fewer),
                        seed=int(seed),
                        focal_length_mm=float(focal if focal is not None else 35.0),
                        aperture_mm=float(aperture),
                    ),
                )
            report = evaluation.report
            authored = None
            artifacts: list[Artifact] = []
            if author_under:
                if not evaluation.poses:
                    raise RuntimeError(
                        f"placement produced no accepted cameras ({report['stop_reason']}); "
                        "nothing was authored"
                    )
                planned_rig_path = camera_rig_destination(
                    self._stage, author_under, on_existing
                )
                if output:
                    check_cancelled()
                    publication_target = self._prepare_camera_publication(output)
                output_lock = None
                if output:
                    output_lock = self._camera_publish_lock(publication_target)
                    output_lock.acquire()
                try:
                    # This is the final cancellation boundary for authoring.  It is
                    # deliberately after the potentially blocking publication lock.
                    # Authoring/history and optional report publication then finish
                    # atomically with no cancellation checkpoint.
                    check_cancelled()
                    before = self._stage_layer_snapshot()
                    rig_history = self._camera_rig_history_seed(planned_rig_path)
                    history_before = self.history.snapshot_state()
                    try:
                        authored = author_camera_rig(
                            self._stage,
                            scene,
                            evaluation.poses,
                            report,
                            author_under=author_under,
                            on_existing=on_existing,
                        )
                        if authored["rig_path"] != planned_rig_path:
                            raise RuntimeError(
                                "camera rig destination changed during atomic authoring"
                            )
                        report["authored"] = authored
                        for record, path in zip(
                            report["cameras"], authored["camera_paths"], strict=True
                        ):
                            record["path"] = path
                        self._index_prims()
                        undo_change, redo_change = (
                            self._camera_rig_history_changes(rig_history)
                        )
                        self._record(
                            "camera.place",
                            f"author {method} camera rig {authored['rig_path']}",
                            undo_change,
                            redo_change,
                        )
                        if output:
                            # JSON replace is the final fallible commit operation. If
                            # it fails before replace, stage and history roll back; once
                            # replace succeeds the publisher cannot report failure.
                            path, digest = publish_json_document(
                                report,
                                publication_target,
                            )
                            report["report_artifact"] = {"path": path, "digest": digest}
                            artifacts.append(
                                Artifact(
                                    path=path, kind="summary", label="camera placement"
                                )
                            )
                    except Exception:
                        try:
                            self._restore_stage_layer_snapshot(before)
                        finally:
                            self.history.restore_state(history_before)
                        raise
                    self._mutation_epoch += 1
                    self._viewer_revision += 1
                    self._invalidate_viewer_line_geometry()
                finally:
                    if output_lock is not None:
                        output_lock.release()
            elif output:
                check_cancelled()
                publication_target = self._prepare_camera_publication(output)
                with self._camera_publish_lock(publication_target):
                    check_cancelled()
                    path, digest = publish_json_document(report, publication_target)
                report["report_artifact"] = {"path": path, "digest": digest}
                artifacts.append(
                    Artifact(path=path, kind="summary", label="camera placement")
                )
        except CameraAnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._err("camera.place", exc)
        finally:
            if publication_target is not None:
                publication_target.close()

        issues = []
        if not report["passed"]:
            issues.append(
                Issue(
                    "warn",
                    f"placement did not fully satisfy the request ({report['stop_reason']})",
                )
            )
        ok = not (method == "look_at" and not report["passed"] and not allow_fewer)
        return Response(
            command="camera.place",
            ok=ok,
            summary={
                "method": method,
                "preview": authored is None,
                "cameras": report["selected_count"],
                "passed": report["passed"],
                "stop_reason": report["stop_reason"],
                **(
                    {"coverage": round(report["achieved_coverage"], 6)}
                    if "achieved_coverage" in report
                    else {}
                ),
                **({"rig": authored["rig_path"]} if authored else {}),
            },
            data=report,
            artifacts=artifacts,
            issues=issues,
        )

    def camera_rig_export(
        self,
        rig: str,
        res=None,
        include_visibility: bool = False,
        verify: bool = False,
        scope: str | None = None,
        grid: int = 32,
        device: str = "cpu",
        output: str = "rig.json",
    ) -> Response:
        """Export canonical calibration, visibility, and optional OVRTX evidence."""

        from usd_core.camera_analysis.authoring import (
            read_rig_analysis_policy,
            read_rig_metadata,
        )
        from usd_core.camera_analysis.cancellation import (
            CameraAnalysisCancelled,
            check_cancelled,
            defer_cancellation,
        )
        from usd_core.camera_analysis.coverage import (
            MAX_CAMERAS,
            evaluate_coverage,
            validate_grid_workload,
        )
        from usd_core.camera_analysis.contracts import SceneAnalysisPolicy
        from usd_core.camera_analysis.evidence import (
            verification_semantic_roles,
            verify_rig_with_ovrtx,
        )
        from usd_core.camera_analysis.newton_backend import (
            NewtonVisibilityBackend,
            require_qualified_backend_versions,
        )
        from usd_core.camera_analysis.rig_export import (
            BoundVerificationGeneration,
            PreparedVerificationStaging,
            build_rig_document,
            copy_verification_staging,
            create_owned_private_directory,
            discard_unpublished_verification_generation,
            prepare_verification_generation,
            publish_rig_document,
            rename_noreplace,
            rig_camera_paths,
            validate_verification_generation,
        )
        from usd_core.camera_analysis.scene import (
            analysis_bounds,
            build_scene_analysis_ir,
        )
        from usd_core.render import make_backend

        self._require_stage()
        publication_target = None
        try:
            check_cancelled()
            output = output or "rig.json"
            rig_path = self._path_of(rig)
            if isinstance(res, str):
                from usd_cli.parsing import resolution

                res = resolution(res)
            width, height = tuple(res or (1920, 1080))
            if width < 1 or height < 1 or width * height > 67_108_864:
                raise ValueError(
                    "rig export resolution is invalid or exceeds 64 megapixels"
                )
            if include_visibility or verify:
                require_qualified_backend_versions()
                check_cancelled()
            paths = rig_camera_paths(self._stage, rig_path)
            if len(paths) > MAX_CAMERAS:
                raise ValueError(
                    f"camera rig export accepts at most {MAX_CAMERAS} cameras"
                )
            if verify:
                pixels_per_camera = int(width) * int(height)
                if pixels_per_camera > NewtonVisibilityBackend.max_camera_rays:
                    raise ValueError(
                        "rig verification resolution would trace "
                        f"{pixels_per_camera:,} rays per camera; limit is "
                        f"{NewtonVisibilityBackend.max_camera_rays:,}"
                    )
                total_verification_rays = len(paths) * pixels_per_camera
                if total_verification_rays > 67_108_864:
                    raise ValueError(
                        "rig verification would trace "
                        f"{total_verification_rays:,} Newton camera rays; limit is "
                        "67,108,864 (reduce cameras or --res)"
                    )
            rig_prim = self._stage.GetPrimAtPath(rig_path)
            stored_scope = str(
                rig_prim.GetCustomDataByKey("usdCameraRigScopePath") or ""
            )
            stored_target = str(
                rig_prim.GetCustomDataByKey("usdCameraRigTargetPath") or ""
            )
            requested_scope = self._path_of(scope) if scope else ""
            analysis_policy = read_rig_analysis_policy(rig_prim)
            tool_authored = (
                rig_prim.GetCustomDataByKey("usdCameraRigSchema")
                == "usd-cli.camera-rig.v1"
            )
            rig_method = str(rig_prim.GetCustomDataByKey("usdCameraRigMethod") or "")
            if requested_scope and tool_authored:
                expected_scope = stored_scope or (
                    analysis_policy.scope_paths[0]
                    if len(analysis_policy.scope_paths) == 1
                    else ""
                )
                if rig_method == "look_at":
                    from pxr import UsdGeom

                    scope_prim = self._stage.GetPrimAtPath(requested_scope)
                    analysis_policy = SceneAnalysisPolicy(
                        scope_paths=(str(requested_scope),),
                        include_paths=analysis_policy.include_paths,
                        exclude_paths=analysis_policy.exclude_paths,
                        helper_paths=analysis_policy.helper_paths,
                        target_paths=analysis_policy.target_paths,
                        floor_paths=(str(requested_scope),)
                        if scope_prim.IsA(UsdGeom.Gprim)
                        else (),
                    )
                elif requested_scope != expected_scope:
                    raise ValueError(
                        "--scope cannot override the analysis policy stored on an "
                        "authored camera rig"
                    )
            if not tool_authored and (requested_scope or stored_scope):
                ordinary_scope = str(requested_scope or stored_scope)
                from pxr import UsdGeom

                scope_prim = self._stage.GetPrimAtPath(ordinary_scope)
                analysis_policy = SceneAnalysisPolicy(
                    scope_paths=(ordinary_scope,),
                    floor_paths=(ordinary_scope,)
                    if scope_prim.IsA(UsdGeom.Gprim)
                    else (),
                )
            scene = build_scene_analysis_ir(
                self._stage,
                policy=analysis_policy,
                include_geometry=bool(include_visibility or verify),
            )
            coverage_report = None
            floor_bounds = None
            analytic_backend = None
            if include_visibility:
                scope_path = str(
                    requested_scope or (stored_scope if rig_method != "look_at" else "")
                )
                if not scope_path:
                    raise ValueError(
                        "--include-visibility requires --scope when the rig has no "
                        "stored coverage scope"
                    )
                floor_bounds = analysis_bounds(scene, roots=(scope_path,))
                validate_grid_workload(
                    floor_bounds,
                    grid=int(grid),
                    cell_size_m=None,
                    view_count=len(paths),
                    operation="rig visibility evaluation",
                )
                analytic_backend = NewtonVisibilityBackend(scene, device=device)
                coverage_report = evaluate_coverage(
                    scene,
                    analytic_backend,
                    floor_bounds,
                    scope_path=scope_path,
                    camera_paths=paths,
                    grid=int(grid),
                ).report
            verification = None
            evidence_paths: list[str] = []
            if verify:
                check_cancelled()
                publication_target = self._prepare_camera_publication(output)
            else:
                check_cancelled()
                document = build_rig_document(
                    self._stage,
                    scene,
                    rig_path,
                    resolution=(int(width), int(height)),
                    floor_bounds_m=floor_bounds,
                    coverage_report=coverage_report,
                    verification=None,
                )
                check_cancelled()
                publication_target = self._prepare_camera_publication(output)
            destination = publication_target.destination
            with ExitStack() as publication_locks:
                requested_locks = [self._camera_publish_lock(publication_target)]
                if verify:
                    requested_locks.append(
                        self._camera_publish_lock(
                            str(destination.parent / f"{destination.stem}_evidence")
                        )
                    )
                # A symlink can make the JSON and evidence names resolve to the same
                # process lock. Deduplicate by identity, then use one stable process-wide
                # ordering so neither self-deadlock nor cross-session ABBA is possible.
                unique_locks = {id(lock): lock for lock in requested_locks}
                for lock_identity in sorted(unique_locks):
                    publication_locks.enter_context(unique_locks[lock_identity])
                if verify:
                    check_cancelled()
                    staging: PreparedVerificationStaging | None = None
                    generation_dir: Path | None = None
                    unpublished_generation_name: str | None = None
                    unpublished_generation_identity: tuple[int, int] | None = None
                    evidence_parent_fd: int | None = None
                    evidence_root_fd: int | None = None
                    evidence_root_identity: tuple[int, int] | None = None
                    generation_fd: int | None = None
                    try:
                        staging = PreparedVerificationStaging(
                            prefix=f"usd-cli-{destination.stem}-evidence-"
                        )
                        render_backend = make_backend(self._render_config(None))
                        if analytic_backend is None:
                            analytic_backend = NewtonVisibilityBackend(
                                scene, device=device
                            )
                        # Only claim floor identity after the visibility probe has
                        # measured the exact accepted surface shapes. A broad scope
                        # ancestor may also contain shelves and machines, so assigning
                        # one inherited semantic label to it would not prove floor
                        # agreement. `--verify` without `--include-visibility` therefore
                        # retains exact target semantics but makes no floor assertion.
                        measured_floor_paths = [
                            str(floor_path)
                            for floor_path in (coverage_report or {})
                            .get("grid", {})
                            .get("accessible_surface_paths", [])
                        ]
                        semantic_roles = verification_semantic_roles(
                            scene,
                            target_path=stored_target,
                            floor_paths=measured_floor_paths,
                        )
                        verification_configuration = {
                            "rig_path": rig_path,
                            "rig_metadata": read_rig_metadata(rig_prim),
                            "analytic_backend": {
                                "name": "newton_warp",
                                **analytic_backend.versions,
                            },
                        }
                        verification, _ = verify_rig_with_ovrtx(
                            self._stage,
                            paths,
                            backend=render_backend,
                            resolution=(int(width), int(height)),
                            output_dir=staging.path,
                            source_digest=scene.source_digest,
                            artifact_base_dir=staging.path,
                            semantic_roles=semantic_roles,
                            configuration=verification_configuration,
                            analytic_backend=analytic_backend,
                            analysis_scene=scene,
                            analysis_policy=analysis_policy,
                        )
                        staging.validate_named_identity()
                        with defer_cancellation():
                            verification, generation_dir, evidence_paths = (
                                prepare_verification_generation(
                                    verification,
                                    staging_dir=staging.path,
                                    artifact_base_dir=destination.parent,
                                    evidence_root_name=(f"{destination.stem}_evidence"),
                                    staged_artifact_base_dir=staging.path,
                                )
                            )
                            generation_relative_path = (
                                f"{generation_dir.parent.name}/{generation_dir.name}"
                            )
                            generation_prefix = PurePosixPath(
                                generation_relative_path
                            ).parts
                            staged_artifact_paths = []
                            for artifact in verification["artifacts"]:
                                relative = PurePosixPath(artifact["relative_path"])
                                if (
                                    relative.parts[: len(generation_prefix)]
                                    != generation_prefix
                                    or len(relative.parts) <= len(generation_prefix)
                                ):
                                    raise ValueError(
                                        "verification artifact is outside its final "
                                        "generation"
                                    )
                                staged_artifact_paths.append(
                                    PurePosixPath(
                                        *relative.parts[len(generation_prefix) :]
                                    ).as_posix()
                                )
                            document = build_rig_document(
                                self._stage,
                                scene,
                                rig_path,
                                resolution=(int(width), int(height)),
                                floor_bounds_m=floor_bounds,
                                coverage_report=coverage_report,
                                verification=verification,
                            )
                            evidence_root = generation_dir.parent
                            directory_open_flags = (
                                os.O_RDONLY
                                | os.O_DIRECTORY
                                | os.O_NOFOLLOW
                            )
                            evidence_parent_fd = os.dup(
                                publication_target.directory_descriptor
                            )
                            publication_target.validate_parent_identity()
                            try:
                                (
                                    evidence_root_fd,
                                    evidence_root_identity,
                                ) = create_owned_private_directory(
                                    evidence_parent_fd, evidence_root.name
                                )
                            except FileExistsError:
                                try:
                                    evidence_root_fd = os.open(
                                        evidence_root.name,
                                        directory_open_flags,
                                        dir_fd=evidence_parent_fd,
                                    )
                                except OSError as exc:
                                    raise ValueError(
                                        "camera verification evidence root must be a "
                                        "real directory, not a symlink or file"
                                    ) from exc
                            except OSError as exc:
                                raise ValueError(
                                    "camera verification evidence root must be a "
                                    "real directory, not a symlink or file"
                                ) from exc
                            opened_evidence_root = os.fstat(evidence_root_fd)
                            named_evidence_root = os.stat(
                                evidence_root.name,
                                dir_fd=evidence_parent_fd,
                                follow_symlinks=False,
                            )
                            if (
                                not stat.S_ISDIR(opened_evidence_root.st_mode)
                                or not stat.S_ISDIR(named_evidence_root.st_mode)
                                or opened_evidence_root.st_uid != os.geteuid()
                                or named_evidence_root.st_uid != os.geteuid()
                                or stat.S_IMODE(opened_evidence_root.st_mode) & 0o022
                                or stat.S_IMODE(named_evidence_root.st_mode) & 0o022
                                or (
                                    evidence_root_identity is not None
                                    and (
                                        opened_evidence_root.st_dev,
                                        opened_evidence_root.st_ino,
                                    )
                                    != evidence_root_identity
                                )
                                or (
                                    opened_evidence_root.st_dev,
                                    opened_evidence_root.st_ino,
                                )
                                != (
                                    named_evidence_root.st_dev,
                                    named_evidence_root.st_ino,
                                )
                            ):
                                raise ValueError(
                                    "camera verification evidence root changed, is not "
                                    "owned by the current user, or is writable by group/other"
                                )
                            evidence_root_identity = (
                                opened_evidence_root.st_dev,
                                opened_evidence_root.st_ino,
                            )
                            try:
                                generation_fd = os.open(
                                    generation_dir.name,
                                    os.O_RDONLY
                                    | getattr(os, "O_DIRECTORY", 0)
                                    | getattr(os, "O_NOFOLLOW", 0),
                                    dir_fd=evidence_root_fd,
                                )
                            except FileNotFoundError:
                                generation_exists = False
                            except OSError as exc:
                                raise ValueError(
                                    "camera verification generation must be a "
                                    "real directory, not a symlink or file"
                                ) from exc
                            else:
                                generation_exists = True
                            if generation_exists:
                                validate_verification_generation(
                                    verification,
                                    artifact_base_dir=destination.parent,
                                    generation_dir_fd=generation_fd,
                                    generation_relative_path=generation_relative_path,
                                )
                                staging.cleanup()
                                staging = None
                            else:
                                unpublished_generation_name = (
                                    f".{generation_dir.name}.staging."
                                    f"{os.urandom(8).hex()}"
                                )
                                private_evidence_root = os.fstat(evidence_root_fd)
                                if (
                                    not stat.S_ISDIR(private_evidence_root.st_mode)
                                    or private_evidence_root.st_uid != os.geteuid()
                                    or stat.S_IMODE(private_evidence_root.st_mode) & 0o022
                                ):
                                    raise ValueError(
                                        "camera verification unpublished generations "
                                        "require a current-user private evidence root"
                                    )
                                (
                                    generation_fd,
                                    unpublished_generation_identity,
                                ) = create_owned_private_directory(
                                    evidence_root_fd, unpublished_generation_name
                                )
                                opened_unpublished = os.fstat(generation_fd)
                                named_unpublished = os.stat(
                                    unpublished_generation_name,
                                    dir_fd=evidence_root_fd,
                                    follow_symlinks=False,
                                )
                                opened_unpublished_identity = (
                                    opened_unpublished.st_dev,
                                    opened_unpublished.st_ino,
                                )
                                if (
                                    not stat.S_ISDIR(opened_unpublished.st_mode)
                                    or not stat.S_ISDIR(named_unpublished.st_mode)
                                    or opened_unpublished.st_uid != os.geteuid()
                                    or named_unpublished.st_uid != os.geteuid()
                                    or stat.S_IMODE(opened_unpublished.st_mode) & 0o777
                                    != 0o700
                                    or stat.S_IMODE(named_unpublished.st_mode) & 0o777
                                    != 0o700
                                    or opened_unpublished_identity
                                    != unpublished_generation_identity
                                    or opened_unpublished_identity
                                    != (
                                        named_unpublished.st_dev,
                                        named_unpublished.st_ino,
                                    )
                                ):
                                    raise ValueError(
                                        "camera verification unpublished generation "
                                        "changed during creation"
                                    )
                                copy_verification_staging(
                                    source_dir_fd=staging.directory_descriptor,
                                    destination_dir_fd=generation_fd,
                                    artifact_relative_paths=staged_artifact_paths,
                                )
                                validate_verification_generation(
                                    verification,
                                    artifact_base_dir=destination.parent,
                                    generation_dir_fd=generation_fd,
                                    generation_relative_path=generation_relative_path,
                                )
                                unpublished_metadata = os.fstat(generation_fd)
                                named_unpublished = os.stat(
                                    unpublished_generation_name,
                                    dir_fd=evidence_root_fd,
                                    follow_symlinks=False,
                                )
                                if (
                                    not stat.S_ISDIR(named_unpublished.st_mode)
                                    or unpublished_metadata.st_dev
                                    != named_unpublished.st_dev
                                    or unpublished_metadata.st_ino
                                    != named_unpublished.st_ino
                                ):
                                    raise ValueError(
                                        "camera verification generation changed during "
                                        "cross-filesystem copy"
                                    )
                                try:
                                    rename_noreplace(
                                        unpublished_generation_name,
                                        generation_dir.name,
                                        source_dir_fd=evidence_root_fd,
                                        destination_dir_fd=evidence_root_fd,
                                    )
                                except Exception:
                                    try:
                                        final_after_error = os.stat(
                                            generation_dir.name,
                                            dir_fd=evidence_root_fd,
                                            follow_symlinks=False,
                                        )
                                    except OSError:
                                        final_after_error = None
                                    final_matches_held = (
                                        final_after_error is not None
                                        and (
                                            final_after_error.st_dev,
                                            final_after_error.st_ino,
                                        )
                                        == unpublished_generation_identity
                                    )
                                    try:
                                        source_after_error = os.stat(
                                            unpublished_generation_name,
                                            dir_fd=evidence_root_fd,
                                            follow_symlinks=False,
                                        )
                                    except OSError:
                                        source_after_error = None
                                    source_still_owned = (
                                        source_after_error is not None
                                        and (
                                            source_after_error.st_dev,
                                            source_after_error.st_ino,
                                        )
                                        == unpublished_generation_identity
                                    )
                                    promotion_consumed = (
                                        final_matches_held or not source_still_owned
                                    )
                                    if promotion_consumed:
                                        # Once the protected source name is consumed,
                                        # the held fd may already be the immutable final
                                        # generation. Never recursively discard it.
                                        unpublished_generation_name = None
                                        unpublished_generation_identity = None
                                    if not promotion_consumed:
                                        raise
                                else:
                                    unpublished_generation_name = None
                                    unpublished_generation_identity = None
                                os.fsync(evidence_root_fd)
                                published_generation = os.stat(
                                    generation_dir.name,
                                    dir_fd=evidence_root_fd,
                                    follow_symlinks=False,
                                )
                                if (
                                    not stat.S_ISDIR(published_generation.st_mode)
                                    or unpublished_metadata.st_dev
                                    != published_generation.st_dev
                                    or unpublished_metadata.st_ino
                                    != published_generation.st_ino
                                ):
                                    raise ValueError(
                                        "camera verification generation changed during "
                                        "atomic publication"
                                    )
                                staging.cleanup()
                                staging = None
                            if generation_fd is None:
                                raise RuntimeError(
                                    "camera verification generation was not opened"
                                )
                            opened_evidence_root = os.fstat(evidence_root_fd)
                            named_evidence_root = os.stat(
                                evidence_root.name,
                                dir_fd=evidence_parent_fd,
                                follow_symlinks=False,
                            )
                            if (
                                not stat.S_ISDIR(named_evidence_root.st_mode)
                                or opened_evidence_root.st_dev
                                != named_evidence_root.st_dev
                                or opened_evidence_root.st_ino
                                != named_evidence_root.st_ino
                            ):
                                raise ValueError(
                                    "camera verification evidence root changed "
                                    "during validation"
                                )
                            opened_generation = os.fstat(generation_fd)
                            named_generation = os.stat(
                                generation_dir.name,
                                dir_fd=evidence_root_fd,
                                follow_symlinks=False,
                            )
                            if (
                                not stat.S_ISDIR(named_generation.st_mode)
                                or opened_generation.st_dev != named_generation.st_dev
                                or opened_generation.st_ino != named_generation.st_ino
                            ):
                                raise ValueError(
                                    "camera verification generation changed during "
                                    "validation"
                                )
                            validate_verification_generation(
                                verification,
                                artifact_base_dir=destination.parent,
                                generation_dir_fd=generation_fd,
                                generation_relative_path=generation_relative_path,
                            )
                            publication_target.validate_parent_identity()
                            bound_generation = BoundVerificationGeneration(
                                artifact_base_dir=destination.parent,
                                generation_relative_path=generation_relative_path,
                                publication_parent_dir_fd=(
                                    publication_target.directory_descriptor
                                ),
                                evidence_parent_dir_fd=evidence_parent_fd,
                                evidence_root_name=evidence_root.name,
                                evidence_root_dir_fd=evidence_root_fd,
                                generation_name=generation_dir.name,
                                generation_dir_fd=generation_fd,
                            )
                            path, digest = publish_rig_document(
                                document,
                                publication_target,
                                verification_generation=bound_generation,
                            )
                            opened_fd = generation_fd
                            generation_fd = None
                            with suppress(OSError):
                                os.close(opened_fd)
                    except Exception:
                        # Leave a promoted content-addressed generation in place for
                        # safe reuse. An unpublished generation is cleared only through
                        # its held descriptor; a substituted name remains untouched.
                        if (
                            generation_fd is not None
                            and evidence_root_fd is not None
                            and unpublished_generation_name is not None
                            and unpublished_generation_identity is not None
                        ):
                            with suppress(OSError, ValueError):
                                discard_unpublished_verification_generation(
                                    evidence_root_dir_fd=evidence_root_fd,
                                    generation_dir_fd=generation_fd,
                                    generation_name=unpublished_generation_name,
                                    generation_identity=(
                                        unpublished_generation_identity
                                    ),
                                )
                        if generation_fd is not None:
                            with suppress(OSError):
                                os.close(generation_fd)
                            generation_fd = None
                        if staging is not None:
                            with suppress(OSError, ValueError):
                                staging.cleanup()
                            staging = None
                        unpublished_generation_name = None
                        unpublished_generation_identity = None
                        if evidence_root_fd is not None:
                            with suppress(OSError):
                                os.close(evidence_root_fd)
                            evidence_root_fd = None
                        if evidence_parent_fd is not None:
                            with suppress(OSError):
                                os.close(evidence_parent_fd)
                            evidence_parent_fd = None
                        # Do not remove evidence pathnames here. Another process can
                        # replace those names, while the descriptor-bound complete
                        # generation is a safe unreferenced orphan.
                        raise
                    else:
                        if evidence_root_fd is not None:
                            with suppress(OSError):
                                os.close(evidence_root_fd)
                        if evidence_parent_fd is not None:
                            with suppress(OSError):
                                os.close(evidence_parent_fd)
                else:
                    check_cancelled()
                    path, digest = publish_rig_document(document, publication_target)
        except CameraAnalysisCancelled:
            raise
        except Exception as exc:  # noqa: BLE001
            return self._err("camera.rig-export", exc)
        finally:
            if publication_target is not None:
                publication_target.close()
        artifacts = [Artifact(path=path, kind="export", label="camera rig")]
        artifacts.extend(
            Artifact(path=item, kind="evidence", label="OVRTX camera verification")
            for item in evidence_paths
        )
        return Response(
            command="camera.rig-export",
            summary={
                "rig": rig_path,
                "cameras": len(document["cameras"]),
                "output": path,
                "verified": bool(verify),
            },
            data={"schema": document["schema"], "digest": digest, "document": document},
            artifacts=artifacts,
        )

    def describe(self, ref: str | None = None) -> Response:  # PR-11.5
        """Rule-based natural-language description of the scene or one object."""
        from usd_core import query
        self._require_stage()
        try:
            if ref:
                text = query.describe_prim(self._stage, self._path_of(ref), ref=ref)
            else:
                text = query.describe_scene(self._stage, self._stage_path)
        except Exception as exc:  # noqa: BLE001
            return self._err("describe", exc)
        return Response(command="describe", data={"text": text})

    def find(self, name=None, type=None, material=None, has_attr=None, in_box=None,
             in_sphere=None, visible=False, hidden=False, selected=False,
             under=None, count_only=False, where=None, **_ignored) -> Response:  # PR-11
        """Search the scene; returns matching @refs + paths (with -q, bare refs for piping)."""
        from usd_core import query
        self._require_stage()
        self._index_prims()
        try:
            box = self._parse_box(in_box)
            sphere = self._parse_sphere(in_sphere)
            under_path = str(self._path_of(under)) if under else None
            hits = query.find_prims(self._stage, name=name, type_=type, material=material,
                                    has_attr=has_attr, in_box=box, in_sphere=sphere,
                                    visible=visible, hidden=hidden, under=under_path,
                                    where=list(where) if where else None)
            if selected:
                sel = set(self._selection)
                hits = [h for h in hits if h["path"] in sel]
        except Exception as exc:  # noqa: BLE001
            return self._err("find", exc)
        if count_only:
            by_type: dict[str, int] = {}
            for h in hits:
                by_type[h["type"] or "?"] = by_type.get(h["type"] or "?", 0) + 1
            text = "\n".join(f"{n:6}  {tp}" for tp, n in
                              sorted(by_type.items(), key=lambda kv: -kv[1]))
            return Response(command="find", summary={"matches": len(hits)},
                            data={"text": text, "counts_by_type": by_type})
        rows = [{"ref": self.refs.ref_for_path(h["path"]), **h} for h in hits]
        refs = [r["ref"] for r in rows if r["ref"]]
        # ref + type + name ALWAYS (bare refs forced 99 identity roundtrips in the
        # round-7 traces); -q still emits bare refs for pipes (see output.emit)
        table = "\n".join(f'{(r["ref"] or "-"):>5} [{r["type"]}] "{r["name"]}"' for r in rows)
        return Response(command="find", summary={"matches": len(rows)},
                        data={"text": table, "results": rows, "refs": refs})

    def bounds(self, ref: str | None = None, **_ignored) -> Response:
        """World-space bbox one-liner for a prim (or the whole stage) — agents
        guessed this verb in round 8 and read installed sources when it 404'd."""
        from usd_core.spatial import get_world_bbox
        self._require_stage()
        try:
            path = self._path_of(ref) if ref else "/"
            b = get_world_bbox(self._stage, path)
        except Exception as exc:  # noqa: BLE001
            return self._err("bounds", exc)
        if not b:
            return self._err("bounds", ValueError(
                f"{ref or 'stage'} has no computable bounds (non-boundable, or empty)"))
        summary = {"target": ref or "/",
                   "size": [round(v, 6) for v in b["size"]],
                   "center": [round(v, 6) for v in b["center"]]}
        data = {k: b[k] for k in ("min", "max", "center", "size")}
        data["text"] = (f"min {[round(v, 4) for v in b['min']]}  "
                        f"max {[round(v, 4) for v in b['max']]}")
        return Response(command="bounds", summary=summary, data=data)

    def visibility(self, ref: str, **_ignored) -> Response:
        """Report a prim's computed + authored visibility and, when it is hidden,
        WHICH ancestor's opinion hides it — the post-isolate question round-8
        agents answered by trawling properties output. Read-only: set with
        show/hide (or isolate --restore for isolate's own opinions)."""
        from pxr import UsdGeom
        self._require_stage()
        try:
            path = self._path_of(ref)
        except Exception as exc:  # noqa: BLE001
            return self._err("visibility", exc)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return self._err("visibility", ValueError(f"no prim at {path}"))
        img = UsdGeom.Imageable(prim)
        if not prim.IsA(UsdGeom.Imageable):
            return self._err("visibility", ValueError(f"{ref} is not imageable"))
        computed = str(img.ComputeVisibility())
        attr = img.GetVisibilityAttr()
        authored = str(attr.Get()) if attr and attr.HasAuthoredValue() else None
        summary = {"ref": ref, "computed": computed}
        if authored:
            summary["authored"] = authored
        lines = [f"{ref}: {computed}"
                 + (f" (authored: {authored})" if authored else " (no authored opinion)")]
        if computed == "invisible" and authored != "invisible":
            cause = prim.GetParent()
            while cause and cause.IsValid() and cause.GetPath().pathString != "/":
                ca = UsdGeom.Imageable(cause).GetVisibilityAttr()
                if ca and ca.HasAuthoredValue() and str(ca.Get()) == "invisible":
                    cref = self.refs.ref_for_path(cause.GetPath().pathString)
                    summary["hidden_by"] = cref or cause.GetPath().pathString
                    lines.append(f"  hidden by ancestor "
                                 f"{cref or cause.GetPath().pathString} — "
                                 "`show` it (or `isolate --restore`)")
                    break
                cause = cause.GetParent()
        return Response(command="visibility", summary=summary,
                        data={"text": "\n".join(lines), "computed": computed,
                              "authored": authored})

    def properties(self, ref: str,
                   attr: str | list[str] | None = None) -> Response:  # PR-11
        """Full prim introspection: attributes, relationships, bounds, material binding.

        With attr name(s), returns just those values (full arrays up to 256
        elements instead of the aggressive whole-prim elision). Multiple names
        in one call — agents batch shader-param reads (round 8: ~50 usage
        errors from `properties @m1 inputs:a inputs:b` before this existed)."""
        import json as _json

        from usd_core import query
        self._require_stage()
        try:
            names = ([attr] if isinstance(attr, str) else list(attr or []))
            if names:
                path = self._path_of(ref)
                lines, attrs_out = [], {}
                for name in names:
                    try:
                        info = query.prim_attribute(self._stage, path, name)
                    except ValueError as exc:
                        lines.append(f"{ref}.{name}: {exc}")
                        attrs_out[name] = {"error": str(exc)}
                        continue
                    lines.append(f"{ref}.{name} ({info['type']}) = "
                                 f"{_json.dumps(info['value'])}"
                                 + ("" if info["authored"]
                                    else "  [schema default, not authored]"))
                    attrs_out[name] = {k: info[k]
                                       for k in ("type", "authored", "value")}
                return Response(command="properties",
                                summary={"ref": ref, "attrs": len(names)},
                                data={"path": path, "attrs": attrs_out,
                                      "text": "\n".join(lines)})
            props = query.prim_properties(self._stage, self._path_of(ref))
        except Exception as exc:  # noqa: BLE001
            return self._err("properties", exc)
        # ref-keyed summary + concise text; the full SdfPath lives in the JSON data only.
        lines = [f'{ref} [{props["type"]}] "{props["name"]}"  '
                 f'{"visible" if props["visible"] else "hidden"}, attrs={props["attribute_count"]}']
        if props.get("bounds"):
            lines.append(f"  size {[round(s, 3) for s in props['bounds']['size']]}")
        if props.get("material"):
            mref = self.refs.ref_for_path(props["material"]["bound_material_path"])
            lines.append(f"  material {mref or props['material']['bound_material_path'].rsplit('/', 1)[-1]}")
        if props.get("attributes"):
            lines.append("  attributes: " + ", ".join(sorted(props["attributes"])[:12]))
        props["text"] = "\n".join(lines)
        return Response(command="properties",
                        summary={"ref": ref, "type": props["type"], "name": props["name"]},
                        data=props)

    def material_binding(self, ref: str) -> Response:  # material query
        from pxr import UsdShade
        from usd_core.materials import (bound_material, describe_material,
                                        effective_materials_under)
        self._require_stage()
        try:
            path = self._path_of(ref)
            prim = self._stage.GetPrimAtPath(path)
            if UsdShade.Material(prim):  # a material ref: describe the material itself
                shader = describe_material(self._stage, path)
                text = [f"{ref}: material ({shader['kind'] or 'no shader'})"]
                if shader["kind"] == "mdl":
                    text.append(f"  MDL {shader['mdl_module']} :: {shader['mdl_subidentifier']}")
                text += [f"  {k} = {v}" for k, v in sorted(shader["inputs"].items())]
                return Response(command="material-binding",
                                summary={"ref": ref, "shader": shader["kind"]},
                                data={**shader, "text": "\n".join(text)})
            data = bound_material(self._stage, path)
            if data["bound_material_path"]:  # enrich with the surface shader (preview vs MDL)
                data["shader"] = describe_material(self._stage, data["bound_material_path"])
            if prim.IsValid() and prim.IsInstance():
                # effective materials inside the collapsed instance, without needing
                # prototype paths
                data["materials_within"] = effective_materials_under(self._stage, path)
        except Exception as exc:  # noqa: BLE001
            return self._err("material-binding", exc)
        summary = {"ref": ref, "binding": data["binding_type"]}
        sh = data.get("shader") or {}
        if sh.get("kind"):
            summary["shader"] = sh["kind"]
        # readable, ref/name-keyed description (no full SdfPaths) — verifies preview vs MDL
        mat = data.get("bound_material_path")
        if mat:
            mat_label = self.refs.ref_for_path(mat) or mat.rsplit("/", 1)[-1]
            lines = [f"{ref}: {data['binding_type']} → material {mat_label}"]
            if sh.get("kind") == "mdl":
                lines.append(f"  shader: MDL {sh.get('mdl_module')} :: {sh.get('mdl_subidentifier')}")
            elif sh.get("kind"):
                lines.append(f"  shader: {sh['kind']}")
            inputs = sh.get("inputs") or {}
            if inputs:
                lines.append("  inputs: " + ", ".join(
                    f"{k}={v}" for k, v in list(inputs.items())[:8]))
            data["text"] = "\n".join(lines)
        else:
            data["text"] = f"{ref}: no material bound"
        within = data.get("materials_within")
        if within and within["materials"]:
            data["text"] += "\n  within instance: " + ", ".join(
                f"{p.rsplit('/', 1)[-1]}×{n}" for p, n in sorted(
                    within["materials"].items(), key=lambda kv: -kv[1]))
            if within["unbound"]:
                data["text"] += f" (+{within['unbound']} unbound gprims)"
        return Response(command="material-binding", summary=summary, data=data)

    # -- manipulation (PR-4) -----------------------------------------------------
    def _err(self, command: str, exc: Exception) -> Response:
        from usd_core.models import Issue
        return Response(command=command, ok=False, issues=[Issue("error", str(exc) or type(exc).__name__)])

    def _record(self, command: str, label: str, undo_change: dict, redo_change: dict) -> None:
        self.history.record(Op(command=command, label=label,
                               inverse={"undo": undo_change, "redo": redo_change}))

    def _stage_layer_by_name(self, name: str):
        if name == "root":
            return self._stage.GetRootLayer()
        if name == "session":
            return self._stage.GetSessionLayer()
        raise ValueError(f"unsupported editable layer: {name}")

    def _stage_edit_layer_name(self) -> str:
        target = self._stage.GetEditTarget().GetLayer()
        root = self._stage.GetRootLayer()
        session = self._stage.GetSessionLayer()
        if target.identifier == root.identifier:
            return "root"
        if target.identifier == session.identifier:
            return "session"
        raise RuntimeError(
            "camera rig history supports only the root or session edit target"
        )

    @staticmethod
    def _layer_prim_siblings(layer, path):
        """Return the ordered child-spec view containing ``path``."""

        from pxr import Sdf

        parent_path = path.GetParentPath()
        if parent_path == Sdf.Path.absoluteRootPath:
            return layer.rootPrims
        parent = layer.GetPrimAtPath(parent_path)
        return None if parent is None else parent.nameChildren

    @staticmethod
    def _layer_subtree_fragment(layer, path) -> str | None:
        """Serialize one authored prim subtree, excluding unrelated layer specs."""

        from pxr import Sdf

        if layer.GetPrimAtPath(path) is None:
            return None
        fragment = Sdf.Layer.CreateAnonymous("usd-cli-history.usda")
        # CopySpec requires destination ancestors to exist.  These inert scaffold
        # overs are not copied back; restoration copies only ``path`` itself.
        Sdf.CreatePrimInLayer(fragment, path)
        if not Sdf.CopySpec(layer, path, fragment, path):
            raise RuntimeError(f"failed to capture USD subtree at {path}")
        return fragment.ExportToString()

    @staticmethod
    def _layer_instanceable_state(layer, path) -> dict:
        spec = layer.GetPrimAtPath(path)
        if spec is None:
            raise RuntimeError(f"missing editable prim spec at {path}")
        authored = spec.HasInfo("instanceable")
        return {
            "path": path.pathString,
            "authored": authored,
            **({"value": bool(spec.GetInfo("instanceable"))} if authored else {}),
        }

    def _layer_subtree_snapshot(
        self,
        layer_name: str,
        path,
        *,
        instanceable_paths: tuple = (),
    ) -> dict:
        """Small exact history payload for a rig subtree and touched ancestors."""

        layer = self._stage_layer_by_name(layer_name)
        spec = layer.GetPrimAtPath(path)
        siblings = self._layer_prim_siblings(layer, path)
        if spec is not None and siblings is None:
            raise RuntimeError(f"missing parent spec for camera rig history at {path}")
        return {
            "kind": "layer_subtree_snapshot",
            "layer": layer_name,
            "path": path.pathString,
            "fragment": self._layer_subtree_fragment(layer, path),
            "sibling_index": None if spec is None else siblings.index(path.name),
            "instanceable": [
                self._layer_instanceable_state(layer, ancestor)
                for ancestor in instanceable_paths
            ],
            "edit_target": self._stage_edit_layer_name(),
            "active_camera": self._active_cam,
        }

    def _camera_rig_history_seed(self, rig_path: str) -> dict:
        """Capture the bounded pre-state touched by one planned rig authoring."""

        from pxr import Sdf

        path = Sdf.Path(rig_path)
        layer_name = self._stage_edit_layer_name()
        layer = self._stage_layer_by_name(layer_name)
        anchor = path
        for prefix in path.GetPrefixes():
            if prefix == Sdf.Path.absoluteRootPath:
                continue
            if layer.GetPrimAtPath(prefix) is None:
                # Defining the rig will create this missing ancestor and every
                # descendant below it.  Capturing at the first missing spec lets
                # undo remove that otherwise-left-behind scaffold exactly.
                anchor = prefix
                break

        # Rig authoring may disable a regular instance ancestor before defining
        # the child.  Existing edit-layer ancestors lie outside ``anchor`` and
        # therefore need their tiny authored instanceable opinion captured too.
        instanceable_paths = tuple(
            prefix
            for prefix in path.GetPrefixes()
            if prefix != Sdf.Path.absoluteRootPath
            and prefix != path
            and not prefix.HasPrefix(anchor)
            and self._stage.GetPrimAtPath(prefix).IsValid()
            and self._stage.GetPrimAtPath(prefix).IsInstance()
        )
        return {
            "layer": layer_name,
            "path": anchor,
            "instanceable_paths": instanceable_paths,
            "before": self._layer_subtree_snapshot(
                layer_name,
                anchor,
                instanceable_paths=instanceable_paths,
            ),
        }

    def _camera_rig_history_changes(self, seed: dict) -> tuple[dict, dict]:
        """Finish a rig history entry without retaining whole-stage snapshots."""

        if self._stage_edit_layer_name() != seed["layer"]:
            raise RuntimeError("camera rig authoring changed the USD edit target")
        after = self._layer_subtree_snapshot(
            seed["layer"],
            seed["path"],
            instanceable_paths=seed["instanceable_paths"],
        )
        if after["fragment"] is None:
            raise RuntimeError("camera rig authoring produced no editable subtree")
        return seed["before"], after

    def _restore_layer_subtree_snapshot(self, snapshot: dict) -> None:
        """Restore one targeted rig-history payload exactly in its edit layer."""

        from pxr import Sdf, Usd

        layer = self._stage_layer_by_name(snapshot["layer"])
        path = Sdf.Path(snapshot["path"])
        if not path.IsAbsolutePath() or not path.IsPrimPath():
            raise ValueError(f"invalid camera rig history path: {path}")
        fragment_text = snapshot.get("fragment")
        fragment = None
        if fragment_text is not None:
            fragment = Sdf.Layer.CreateAnonymous("usd-cli-history-restore.usda")
            if not fragment.ImportFromString(fragment_text):
                raise RuntimeError(f"failed to parse camera rig history at {path}")
            if fragment.GetPrimAtPath(path) is None:
                raise RuntimeError(f"camera rig history is missing its subtree at {path}")

        with Sdf.ChangeBlock():
            current = layer.GetPrimAtPath(path)
            if current is not None:
                siblings = self._layer_prim_siblings(layer, path)
                if siblings is None:
                    raise RuntimeError(
                        f"missing parent while replacing camera rig history at {path}"
                    )
                del siblings[path.name]

            sibling_index = snapshot.get("sibling_index")
            if fragment is not None:
                if not Sdf.CopySpec(fragment, path, layer, path):
                    raise RuntimeError(f"failed to restore camera rig subtree at {path}")
                siblings = self._layer_prim_siblings(layer, path)
                restored = layer.GetPrimAtPath(path)
                if siblings is None or restored is None:
                    raise RuntimeError(f"restored camera rig subtree is missing at {path}")
                ordered = list(siblings)
                ordered.remove(restored)
                if (
                    not isinstance(sibling_index, int)
                    or sibling_index < 0
                    or sibling_index > len(ordered)
                ):
                    raise RuntimeError(
                        f"invalid camera rig sibling order at {path}: {sibling_index}"
                    )
                ordered.insert(sibling_index, restored)
                siblings[:] = ordered
            elif sibling_index is not None:
                raise RuntimeError(
                    f"camera rig history has an index without a subtree at {path}"
                )

            for state in snapshot.get("instanceable", ()):
                ancestor = Sdf.Path(state["path"])
                spec = layer.GetPrimAtPath(ancestor)
                if spec is None:
                    raise RuntimeError(
                        f"missing instance ancestor while restoring {ancestor}"
                    )
                if state["authored"]:
                    spec.SetInfo("instanceable", bool(state["value"]))
                else:
                    spec.ClearInfo("instanceable")

        edit_layer = self._stage_layer_by_name(snapshot["edit_target"])
        self._stage.SetEditTarget(Usd.EditTarget(edit_layer))
        self._active_cam = snapshot.get("active_camera")
        self._index_prims()

    def _stage_layer_snapshot(self) -> dict:
        """Exact editable-layer state for one atomic multi-prim operation."""

        root = self._stage.GetRootLayer()
        session = self._stage.GetSessionLayer()
        target = self._stage.GetEditTarget().GetLayer()
        return {
            "kind": "stage_layer_snapshot",
            "root": root.ExportToString(),
            "session": session.ExportToString(),
            "edit_target": (
                "session" if target.identifier == session.identifier else "root"
            ),
            "active_camera": self._active_cam,
        }

    def _restore_stage_layer_snapshot(self, snapshot: dict) -> None:
        from pxr import Usd

        root = self._stage.GetRootLayer()
        session = self._stage.GetSessionLayer()
        if not root.ImportFromString(snapshot["root"]):
            raise RuntimeError("failed to restore the USD root layer snapshot")
        if not session.ImportFromString(snapshot["session"]):
            raise RuntimeError("failed to restore the USD session layer snapshot")
        layer = session if snapshot.get("edit_target") == "session" else root
        self._stage.SetEditTarget(Usd.EditTarget(layer))
        self._active_cam = snapshot.get("active_camera")
        self._index_prims()

    def _prepare_camera_publication(self, output: str):
        """Retain the real output parent for descriptor-bound publication.

        The daemon supplies project-relative outputs as normalized absolute paths and
        gives its Sessions an explicit write-root capability. Direct library/CLI calls
        retain their historical cwd-relative and unrestricted-absolute semantics.
        """

        from usd_core.camera_analysis.rig_export import prepare_json_publication

        raw_roots = self.config.server.get("allowed_write_roots")
        if raw_roots in (None, "", [], ()):
            return prepare_json_publication(output)
        if isinstance(raw_roots, str):
            raw_roots = [part for part in raw_roots.split(os.pathsep) if part]
        if not isinstance(raw_roots, (list, tuple)) or not raw_roots:
            raise ValueError(
                "server.allowed_write_roots must contain at least one path"
            )
        roots: list[Path] = []
        for raw_root in raw_roots:
            if not isinstance(raw_root, str) or not raw_root:
                raise ValueError(
                    "server.allowed_write_roots entries must be non-empty paths"
                )
            roots.append(Path(os.path.abspath(Path(raw_root).expanduser())))
        return prepare_json_publication(output, allowed_roots=roots)

    @staticmethod
    def _camera_publish_lock(output: str | os.PathLike[str]) -> threading.Lock:
        """Return the process-wide lock for one canonical camera output path."""

        destination = Path(output).expanduser().resolve()
        return _publish_dest_lock(str(destination))

    def _abort_txn(self) -> None:
        """Abort an open transaction AND revert the stage mutations already applied.

        history.rollback() only drops the buffered op entries; without re-applying their
        inverses the partial edits would persist with no way to undo them. We apply each
        rolled-back op's undo change (rollback returns them newest-first)."""
        for op in self.history.rollback():
            try:
                self._apply_change(op.inverse["undo"])
            except Exception:  # noqa: BLE001 — best-effort revert; keep aborting the rest
                pass

    def _apply_change(self, change: dict) -> None:
        """Apply a single reversible change dict (used by undo/redo). In-process only."""
        from usd_core import edit, materials
        kind = change["kind"]
        if kind == "set_trs":
            trs = change["trs"]
            edit.set_trs(self._stage, change["path"], trs["translate"], trs["rotate"], trs["scale"])
        elif kind == "set_active":
            edit.set_active(self._stage, change["path"], change["active"])
        elif kind == "set_vis":
            edit.set_visibility(self._stage, change["path"], change["visible"])
        elif kind == "raw_attr":
            prim = self._stage.GetPrimAtPath(change["path"])
            if change.get("remove"):
                prim.RemoveProperty(change["attr"])  # attr was created by this op
            else:
                a = prim.GetAttribute(change["attr"])
                if change.get("unset"):
                    if a:
                        a.Clear()
                else:
                    edit.set_raw_attr(self._stage, change["path"], change["attr"], change["value"])
        elif kind == "rel_targets":
            prim = self._stage.GetPrimAtPath(change["path"])
            rel = prim.GetRelationship(change["attr"]) or prim.CreateRelationship(
                change["attr"], custom=True)
            if change["targets"]:
                rel.SetTargets(change["targets"])
            else:
                rel.ClearTargets(True)
        elif kind == "set_instanceable":
            prim = self._stage.GetPrimAtPath(change["path"])
            if change.get("clear"):
                prim.ClearInstanceable()
            else:
                prim.SetInstanceable(bool(change["value"]))
        elif kind == "bind":
            materials.bind_material(self._stage, change["material"], change["path"])
        elif kind == "unbind":
            materials.unbind_material(self._stage, change["path"])
        elif kind == "block_material_binding":
            # ``appearance clear`` masks weaker bindings with an explicit-empty
            # target list in the strongest session layer.  A speculative bind
            # followed by the ordinary ``unbind`` inverse clears that opinion
            # and silently exposes the source material again.  Restore the mask
            # itself so undo returns to the exact clean-slate semantics.
            prim = self._stage.OverridePrim(change["path"])
            rel = prim.GetRelationship("material:binding")
            if not rel:
                rel = prim.CreateRelationship("material:binding", custom=False)
            if rel.SetTargets([]) is False or rel.GetTargets():
                raise RuntimeError(
                    f"failed to restore appearance mask at {change['path']}"
                )
        elif kind == "rename":
            edit.rename_prim(self._stage, change["path"], change["new_name"])
        elif kind == "reparent":
            edit.reparent_prim(self._stage, change["path"], change["new_parent"])
        elif kind == "apply_api":
            # real remove-api undo: re-apply the schema, then re-author the property
            # opinions captured before removal
            from pxr import Sdf, Usd
            prim = self._stage.GetPrimAtPath(change["path"])
            tf_type = Usd.SchemaRegistry.GetTypeFromSchemaTypeName(change["schema"])
            if change.get("instance"):
                prim.ApplyAPI(tf_type, change["instance"])
            else:
                prim.ApplyAPI(tf_type)
            for spec in change.get("properties", []):
                if spec["kind"] == "relationship":
                    rel = prim.GetRelationship(spec["name"]) or prim.CreateRelationship(
                        spec["name"], custom=False)
                    if spec["targets"]:
                        rel.SetTargets(spec["targets"])
                else:
                    attr = prim.GetAttribute(spec["name"])
                    if not attr or not attr.GetTypeName():
                        attr = prim.CreateAttribute(
                            spec["name"], Sdf.ValueTypeNames.Find(spec["type"]),
                            spec.get("custom", False))
                    if spec.get("had_value") and spec.get("value") is not None:
                        attr.Set(spec["value"])
        elif kind == "remove_api":
            edit.remove_api(self._stage, change["path"], change["api"])
        elif kind == "shader_inputs":
            # material update-in-place undo/redo: restore a snapshot of the surface
            # shader's inputs (values, authored-ness, and connections)
            from pxr import Sdf, UsdShade
            prim = self._stage.GetPrimAtPath(change["path"])
            shader = UsdShade.Shader(prim)
            for spec in change["inputs"]:
                if not spec.get("exists"):
                    prim.RemoveProperty("inputs:" + spec["name"])
                    continue
                inp = shader.CreateInput(spec["name"],
                                         Sdf.ValueTypeNames.Find(spec["type"]))
                conn = spec.get("connection")
                if conn:
                    src = UsdShade.ConnectableAPI(self._stage.GetPrimAtPath(conn["prim"]))
                    inp.ConnectToSource(src, conn["name"])
                elif inp.HasConnectedSource():
                    inp.DisconnectSource()
                if spec.get("had_value") and spec.get("value") is not None:
                    inp.Set(spec["value"])
                elif not spec.get("had_value"):
                    inp.GetAttr().Clear()  # back to the unauthored (fallback) state
        elif kind == "session_layer_snapshot":
            # `appearance clear` can touch thousands of binding/display properties and
            # de-instance nested references. One exact session-layer snapshot makes the
            # whole command one atomic undo unit without mutating any source layer.
            from usd_core.appearance import restore_session_layer

            restore_session_layer(self._stage, change["layer"], change["edit_target"])
            self._index_prims()
        elif kind == "stage_layer_snapshot":
            # Camera-rig authoring can create a parent plus many camera prims and
            # metadata in either the root or session edit target.  Full snapshots
            # remain the transient rollback guard used by undo/redo themselves.
            self._restore_stage_layer_snapshot(change)
        elif kind == "layer_subtree_snapshot":
            # Retained camera-place history is bounded to the authored rig branch
            # plus any instanceable opinions changed on existing ancestors.  A
            # whole root layer can be hundreds of MB and must not be pinned twice
            # per history entry merely to restore one rig.
            self._restore_layer_subtree_snapshot(change)

    @_mutating
    def transform(self, ref: str, tx=None, ty=None, tz=None, rx=None, ry=None, rz=None,
                  sx=None, sy=None, sz=None, translate=None, rotate=None, scale=None,
                  **_ignored) -> Response:  # PR-4.1
        from usd_core import edit
        self._require_stage()
        try:
            path = self._edit_path_of(ref)
            before, after = edit.apply_transform(
                self._stage, path, tx=tx, ty=ty, tz=tz, rx=rx, ry=ry, rz=rz,
                sx=sx, sy=sy, sz=sz, translate=translate, rotate=rotate, scale=scale)
        except Exception as exc:  # noqa: BLE001
            return self._err("transform", exc)
        self._record("transform", f"transform {ref}",
                     {"kind": "set_trs", "path": path, "trs": before},
                     {"kind": "set_trs", "path": path, "trs": after})
        return Response(command="transform", summary={"ref": ref},
                        data={"before": before, "after": after})

    @_mutating
    def create(self, type=None, name=None, parent=None, at=None, shape=None, **_ignored) -> Response:  # PR-4.2
        from usd_core import edit
        self._require_stage()
        try:
            parent_path = self._path_of(parent) if parent else self._default_parent()
            path = edit.create_prim(self._stage, type_=type or "xform", name=name,
                                    parent=parent_path, shape=shape, at=at)
        except Exception as exc:  # noqa: BLE001
            return self._err("create", exc)
        self._index_prims()
        self._record("create", f"create {type} {path}",
                     {"kind": "set_active", "path": path, "active": False},
                     {"kind": "set_active", "path": path, "active": True})
        return Response(command="create",
                        summary={"ref": self.refs.ref_for_path(path), "name": path.rsplit("/", 1)[-1]},
                        data={"path": path})

    @_mutating
    def delete(self, refs: list[str] | None = None, **_ignored) -> Response:  # PR-4.3
        from usd_core import edit
        self._require_stage()
        done = []
        try:
            self.history.begin()
            for r in refs or []:
                path = self._edit_path_of(r)
                edit.set_active(self._stage, path, False)
                self._record("delete", f"delete {r}",
                             {"kind": "set_active", "path": path, "active": True},
                             {"kind": "set_active", "path": path, "active": False})
                done.append(path)
            self.history.commit(f"delete {len(done)} prim(s)")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("delete", exc)
        return Response(command="delete", summary={"deleted": len(done)}, data={"paths": done})

    @_mutating
    def remove_api(self, ref: str, api: str, **_ignored) -> Response:
        """Remove an applied API schema (and the properties it owns) from a prim."""
        from usd_core import edit
        self._require_stage()
        try:
            path = self._edit_path_of(ref)
            info = edit.remove_api(self._stage, path, api)
        except Exception as exc:  # noqa: BLE001
            return self._err("remove-api", exc)
        captured = info.pop("captured", None)
        label = f"remove-api {info['api']} {path}"
        if captured is not None:
            # REAL undo: re-apply the schema and re-author the captured property
            # opinions (the old fake set_active(True)→set_active(True) pair made undo
            # report success while restoring nothing)
            self._record("remove-api", label,
                         {"kind": "apply_api", "path": path, "schema": captured["schema"],
                          "instance": captured["instance"],
                          "properties": captured["properties"]},
                         {"kind": "remove_api", "path": path, "api": info["api"]})
        else:
            # pre-removal state could not be captured — record an honest non-undoable
            # marker so `undo` warns instead of claiming a restore that never happened
            self.history.record(Op(command="remove-api", label=label,
                                   non_undoable="remove-api"))
        summary = {"ref": self.refs.ref_for_path(path), "api": info["api"],
                   "properties_removed": len(info["properties_removed"])}
        if info.get("removed") and not info["properties_removed"] \
                and not info.get("properties_masked"):
            # the schema itself WAS removed — a bare 'properties_removed: 0' made a
            # removal that mattered read like a no-op
            del summary["properties_removed"]
            summary["result"] = "api removed (no authored properties)"
        issues = []
        if info.get("properties_masked"):
            # RemoveProperty only strips the edit-target opinion — these still resolve
            # from a weaker (referenced) layer, so saying "removed" would be a lie
            summary["properties_masked"] = len(info["properties_masked"])
            issues = [Issue("warn", f"{m['name']}: masked, opinion remains in "
                            f"{m['layer']}")
                      for m in info["properties_masked"]]
        return Response(command="remove-api", summary=summary, data=info, issues=issues)

    @_mutating
    def duplicate(self, ref: str, at=None, name=None, **_ignored) -> Response:  # PR-4.4
        from usd_core import edit
        self._require_stage()
        try:
            dst = edit.duplicate_prim(self._stage, self._edit_path_of(ref), name=name,
                                      at=at)
        except Exception as exc:  # noqa: BLE001
            return self._err("duplicate", exc)
        self._index_prims()
        self._record("duplicate", f"duplicate {ref} -> {dst}",
                     {"kind": "set_active", "path": dst, "active": False},
                     {"kind": "set_active", "path": dst, "active": True})
        return Response(command="duplicate",
                        summary={"ref": self.refs.ref_for_path(dst), "name": dst.rsplit("/", 1)[-1]},
                        data={"path": dst})

    @_mutating
    def reparent(self, ref: str, under: str, **_ignored) -> Response:  # PR-4.5
        from usd_core import edit
        self._require_stage()
        try:
            src = self._edit_path_of(ref)
            parent = self._edit_path_of(under)
            new_path = edit.reparent_prim(self._stage, src, parent)
        except Exception as exc:  # noqa: BLE001
            return self._err("reparent", exc)
        self._record("reparent", f"reparent {ref} under {under}",
                     {"kind": "reparent", "path": new_path, "new_parent": src.rsplit("/", 1)[0]},
                     {"kind": "reparent", "path": src, "new_parent": parent})
        self._index_prims()
        return Response(command="reparent",
                        summary={"ref": self.refs.ref_for_path(new_path),
                                 "name": new_path.rsplit("/", 1)[-1],
                                 "note": "ref changed (path moved) — re-snapshot"},
                        data={"path": new_path})

    @_mutating
    def rename(self, ref: str, new_name: str, **_ignored) -> Response:  # PR-4.10
        from usd_core import edit
        self._require_stage()
        try:
            src = self._edit_path_of(ref)
            new_path = edit.rename_prim(self._stage, src, new_name)
        except Exception as exc:  # noqa: BLE001
            return self._err("rename", exc)
        self._record("rename", f"rename {ref} -> {new_name}",
                     {"kind": "rename", "path": new_path, "new_name": src.rsplit("/", 1)[-1]},
                     {"kind": "rename", "path": src, "new_name": new_name})
        self._index_prims()
        return Response(command="rename",
                        summary={"ref": self.refs.ref_for_path(new_path), "name": new_name,
                                 "note": "ref changed (path moved) — re-snapshot"},
                        data={"path": new_path})

    @_mutating
    def group(self, refs: list[str] | None = None, name: str | None = None, **_ignored) -> Response:  # PR-4.9
        from usd_core import edit
        self._require_stage()
        try:
            paths = [self._edit_path_of(r) for r in (refs or [])]
            if not paths:
                raise ValueError("group needs at least one ref")
            parent = paths[0].rsplit("/", 1)[0] or ""
            group_path = edit.create_prim(self._stage, type_="xform", name=name or "Group",
                                          parent=parent, shape=None)
            self.history.begin()
            self._record("group", f"create group {group_path}",
                         {"kind": "set_active", "path": group_path, "active": False},
                         {"kind": "set_active", "path": group_path, "active": True})
            moved = []
            for p in paths:
                np = edit.reparent_prim(self._stage, p, group_path)
                self._record("group", f"reparent {p}",
                             {"kind": "reparent", "path": np, "new_parent": p.rsplit("/", 1)[0]},
                             {"kind": "reparent", "path": p, "new_parent": group_path})
                moved.append(np)
            self.history.commit(f"group {len(moved)} prim(s)")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("group", exc)
        self._index_prims()
        return Response(command="group",
                        summary={"ref": self.refs.ref_for_path(group_path),
                                 "name": group_path.rsplit("/", 1)[-1], "grouped": len(moved)},
                        data={"group": group_path, "children": moved})

    def _capture_shader_inputs(self, material_path: str) -> tuple[str | None, dict]:
        """(shader_prim_path, {input_name: state}) for a material's surface shader.

        The before/after snapshots that make `material <mat> --color …` undo REAL:
        each input's typed value, authored-ness, and connection are recorded so a
        `shader_inputs` history op can restore them exactly."""
        from pxr import UsdShade
        material = UsdShade.Material(self._stage.GetPrimAtPath(material_path))
        source = material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
        shader = source[0] if source else None
        if not shader:
            return None, {}
        state: dict = {}
        for inp in shader.GetInputs():
            attr = inp.GetAttr()
            conn = None
            if inp.HasConnectedSource():
                src, src_name, _src_type = inp.GetConnectedSource()
                conn = {"prim": src.GetPrim().GetPath().pathString, "name": src_name}
            had = attr.HasAuthoredValue()
            state[inp.GetBaseName()] = {
                "type": str(attr.GetTypeName()), "had_value": had,
                "value": attr.Get() if had else None, "connection": conn}
        return shader.GetPrim().GetPath().pathString, state

    @staticmethod
    def _shader_inputs_changes(shader_path: str | None, before: dict,
                               after: dict) -> tuple[dict | None, dict | None]:
        """Diff two `_capture_shader_inputs` snapshots into (undo, redo) change dicts
        for the `shader_inputs` op kind — (None, None) when nothing changed."""
        if shader_path is None:
            return None, None
        undo_entries, redo_entries = [], []
        for name in sorted(set(before) | set(after)):
            b, a = before.get(name), after.get(name)
            if b == a:
                continue
            undo_entries.append({"name": name, "exists": b is not None, **(b or {})})
            redo_entries.append({"name": name, "exists": a is not None, **(a or {})})
        if not undo_entries:
            return None, None
        return ({"kind": "shader_inputs", "path": shader_path, "inputs": undo_entries},
                {"kind": "shader_inputs", "path": shader_path, "inputs": redo_entries})

    @staticmethod
    def _parse_shader_inputs(inputs) -> dict | None:
        """Normalize --input k=v pairs (or a ready dict) into a raw shader-inputs dict."""
        if not inputs:
            return None
        if isinstance(inputs, dict):
            return inputs
        out = {}
        for item in inputs:
            key, sep, raw = str(item).partition("=")
            if not sep or not key:
                raise ValueError(f"--input expects name=value, got '{item}'")
            raw = raw.strip()
            if raw.lower() in ("true", "false"):
                out[key] = raw.lower() == "true"
            elif "," in raw:
                out[key] = [float(x) for x in raw.split(",")]
            else:
                try:
                    out[key] = float(raw)
                except ValueError:
                    out[key] = raw  # string → asset path
        return out

    def _subset_path(self, mesh_path: str, subset_name: str) -> str:
        """The GeomSubset named `subset_name` under a mesh (descends into proxies)."""
        from pxr import Usd
        root = self._stage.GetPrimAtPath(mesh_path)
        for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
            if str(prim.GetTypeName()) == "GeomSubset" and prim.GetName() == subset_name:
                return prim.GetPath().pathString
        raise ValueError(f"no GeomSubset '{subset_name}' under {mesh_path} — "
                         "run 'subsets' on the mesh to list them")

    def _material_targets(self, ref, *, all=False, type=None, where=None, under=None,
                          subset=None, prototype=False) -> tuple[list[str], bool]:
        """Resolve the target prim path(s) for `material`: one ref (optionally narrowed
        to a named GeomSubset) or a bulk rule-based selection. Returns (paths, bulk)."""
        from usd_core import selector
        bulk = bool(all or type or where or under)
        if bulk and ref:
            raise ValueError("give either a positional ref or bulk selectors "
                             "(--all/--type/--where/--under), not both")
        if not bulk:
            if not ref:
                raise ValueError("material needs a ref, or a bulk selection like "
                                 "--all --type Mesh --where 'name~=Conductor*'")
            path = self._path_of(ref)
            if subset:
                path = self._subset_path(path, subset)
            return [path], False
        if subset:
            raise ValueError("--subset applies to a single mesh ref, not a bulk selection")
        prims = selector.select_prims(
            self._stage, types=type, where=where,
            under=self._path_of(under) if under else None)
        if where and not prototype:
            # A --where match on a prim INSIDE an instance would redirect the edit to the
            # whole instance root — so a glob like name~=R* silently over-binds every
            # instance sharing a prototype whose internals happen to match. Keep proxy
            # matches only when the enclosing instance root satisfies the rules itself.
            # (--prototype mode is exempt: there, matching prototype internals through
            # proxies is exactly the intent — the bind collapses to the source.)
            kept, dropped = [], 0
            for p in prims:
                if p.IsInstanceProxy() and not selector.matches_rules(
                        self._stage, selector.instance_root_of(p), where):
                    dropped += 1
                    continue
                kept.append(p)
            if dropped:
                logger.warning("bulk selection: skipped %d instance-internal match(es) "
                               "whose instance root does not match --where", dropped)
            prims = kept
        paths = [p.GetPath().pathString for p in prims]
        if not paths:
            raise ValueError("no prims match the selection — check --type/--where/--under")
        return paths, True

    def _authored_bind_path(self, target: str, prototype: bool) -> str:
        """Where a bind on `target` will actually be authored (prototype source, the
        editable instance root for a proxy, or the prim itself). With `prototype`, a
        non-instanced prim is its own source — a bulk selection legitimately matches
        both the source subtree and its instances' proxies."""
        from usd_core import materials
        prim = self._stage.GetPrimAtPath(target)
        if prototype and prim.IsValid() and (prim.IsInstanceProxy() or prim.IsInstance()):
            return materials.prototype_source_path(self._stage, target)[0]
        prim, redirected = materials._editable_binding_prim(self._stage, target)
        if not prim or not prim.IsValid():
            raise ValueError(f"{target} is not a valid prim")
        return prim.GetPath().pathString

    def _bind_many(self, targets: list[str], mat_path: str, prototype: bool) -> list[str]:
        """Bind one material to many targets in a single undo transaction. Targets that
        collapse onto the same authored prim (instances of one prototype, proxies under
        one instance root) are bound once. Returns the authored binding paths."""
        from usd_core import materials
        authored, seen = [], set()
        self.history.begin()
        try:
            for target in targets:
                done_path = self._authored_bind_path(target, prototype)
                if done_path in seen:
                    continue
                seen.add(done_path)
                prev = materials.bound_material(self._stage, done_path)
                if prototype:
                    # done_path is already the editable prototype source — bind there
                    materials.bind_material(self._stage, mat_path, done_path)
                else:
                    materials.bind_material(self._stage, mat_path, target)
                prev_direct = (prev["bound_material_path"]
                               if prev["binding_type"] == "direct" else None)
                undo = self._material_bind_undo(done_path, prev_direct)
                self._record("material", f"bind {done_path}", undo,
                             {"kind": "bind", "path": done_path, "material": mat_path})
                authored.append(done_path)
            self.history.commit(f"material bind {len(authored)} prim(s)")
        except Exception:
            self._abort_txn()
            raise
        return authored

    def _material_bind_undo(self, path: str, previous_direct: str | None) -> dict:
        """Return an inverse that preserves a clean-slate appearance overlay."""
        if previous_direct:
            return {"kind": "bind", "path": path, "material": previous_direct}
        from usd_core.appearance import appearance_clear_active

        if appearance_clear_active(self._stage):
            return {"kind": "block_material_binding", "path": path}
        return {"kind": "unbind", "path": path}

    @_mutating
    def material(self, ref: str | None = None, color=None, metallic=None, roughness=None,
                 opacity=None, clearcoat=None, clearcoat_roughness=None, ior=None,
                 emissive=None, bind=None, unbind=False, library=None, name=None,
                 library_prim=None,
                 mdl=None, subidentifier=None, diffuse_texture=None, normal_texture=None,
                 orm_texture=None, roughness_texture=None, metallic_texture=None,
                 uv_set=None, tex_scale=None, tex_rotate=None, tex_translate=None,
                 inputs=None, all=False, type=None, where=None, under=None,
                 subset=None, prototype=False, **_ignored) -> Response:  # PR-4.6
        from usd_core import materials
        self._require_stage()
        if library_prim and not library:
            return self._err(
                "material",
                ValueError("--library-prim requires --library"),
            )
        # Standalone library import: `material --library <lib> --name <M>` with no ref and
        # no bulk selectors pulls the material into /Looks without binding (the scene-repair
        # path — import the palette, then re-bind existing prims by name).
        if library and not ref and not (all or type or where or under) and not unbind:
            try:
                mat_path = materials.import_library_material(
                    self._stage,
                    library_path=library,
                    material_name=name or "Material",
                    source_prim_path=library_prim,
                )
            except Exception as exc:  # noqa: BLE001
                return self._err("material", exc)
            self._index_prims()
            self._record("material", f"import {mat_path}",
                         {"kind": "set_active", "path": mat_path, "active": False},
                         {"kind": "set_active", "path": mat_path, "active": True})
            return Response(command="material",
                            summary={"imported": self.refs.ref_for_path(mat_path),
                                     "material": mat_path},
                            data={"material_path": mat_path})
        try:
            # binding is the one edit that works on instance proxies: usd_core.materials
            # redirects the authoring to the editable instanceable root — so _path_of, not
            # _edit_path_of.
            targets, bulk = self._material_targets(ref, all=all, type=type, where=where,
                                                   under=under, subset=subset,
                                                   prototype=prototype)
            raw_inputs = self._parse_shader_inputs(inputs)
            if unbind:
                if bind or library or mdl or raw_inputs or any(
                        v is not None for v in (color, metallic, roughness, opacity,
                                                clearcoat, clearcoat_roughness, ior,
                                                emissive)):
                    raise ValueError("--unbind cannot be combined with other material options")
                if bulk:
                    return self._material_unbind_many(targets)
                return self._material_unbind(
                    ref, targets[0], materials.bound_material(self._stage, targets[0]))
            # `material @m1 --color …` on a Material prim edits it in place —
            # creating and binding a fresh material onto a Material corrupted
            # stages in the wu round (task-12).
            if not bulk and not bind and not library and not unbind:
                from pxr import UsdShade
                tgt_prim = self._stage.GetPrimAtPath(targets[0])
                if tgt_prim and UsdShade.Material(tgt_prim):
                    # snapshot the shader inputs around the update so undo can restore
                    # the previous values/connections for real (the old fake
                    # set_active(True)→set_active(True) entry restored nothing)
                    shader_path, before_inputs = self._capture_shader_inputs(targets[0])
                    info = materials.update_material(
                        self._stage, targets[0], color=color, metallic=metallic,
                        roughness=roughness, opacity=opacity, clearcoat=clearcoat,
                        clearcoat_roughness=clearcoat_roughness, ior=ior,
                        emissive=emissive, inputs=raw_inputs)
                    _, after_inputs = self._capture_shader_inputs(targets[0])
                    undo, redo = self._shader_inputs_changes(
                        shader_path, before_inputs, after_inputs)
                    if undo is not None:
                        self._record("material", f"update {targets[0]}", undo, redo)
                    return Response(command="material",
                                    summary={"updated": self.refs.ref_for_path(targets[0]),
                                             "params": info["updated"]},
                                    data=info)
            base_name = name or (f"{self._short(ref)}_mat" if ref else "bulk_mat")
            if not name and not ref:
                # Repeated unnamed bulk runs must not silently recolor prims the
                # previous run bound: uniquify the default bulk material name.
                from usd_core.edit import unique_child_path
                base_name = unique_child_path(
                    self._stage, materials._looks_scope(self._stage),
                    base_name).rsplit("/", 1)[-1]
            if bind:
                mat_path = self._path_of(bind)
            elif library:
                if prototype:
                    raise ValueError("--library cannot be combined with --prototype — "
                                     "reference the material first, then bind it")
                # Anchor a relative library path to the project dir (the CLI's CWD),
                # not the daemon's — they usually coincide, but not always.
                from pathlib import Path as _Path
                lib_p = _Path(library).expanduser()
                if not lib_p.is_absolute():
                    library = str(((self.config.project_dir or _Path.cwd()) / lib_p).resolve())
                default_prim = self._stage.GetDefaultPrim()
                default_path = default_prim.GetPath().pathString if default_prim else ""
                looks = self._stage.GetPrimAtPath(f"{default_path}/Looks")
                local_material_active = (
                    {
                        child.GetPath().pathString: child.IsActive()
                        for child in looks.GetAllChildren()
                    }
                    if looks.IsValid()
                    else {}
                )
                # reference_library_material binds targets[0] itself; record that undo here
                done0 = self._authored_bind_path(targets[0], False)
                prev0 = materials.bound_material(self._stage, done0)
                mat_path = materials.reference_library_material(
                    self._stage, library_path=library, material_name=name or "Material",
                    source_prim_path=library_prim,
                    prim_path=targets[0])
                prev0_direct = (prev0["bound_material_path"]
                                if prev0["binding_type"] == "direct" else None)
                reactivated = local_material_active.get(mat_path) is False
                if reactivated:
                    self.history.begin()
                    self._record(
                        "material",
                        f"reactivate {mat_path}",
                        {"kind": "set_active", "path": mat_path, "active": False},
                        {"kind": "set_active", "path": mat_path, "active": True},
                    )
                self._record("material", f"bind {done0}",
                             self._material_bind_undo(done0, prev0_direct),
                             {"kind": "bind", "path": done0, "material": mat_path})
                if reactivated:
                    self.history.commit(f"material reactivate and bind {done0}")
            elif mdl:  # mdl carries the module asset (e.g. "OmniPBR.mdl")
                mat_path = materials.create_mdl_material(
                    self._stage, name=base_name, module=mdl,
                    subidentifier=subidentifier, color=color, metallic=metallic,
                    roughness=roughness, opacity=opacity, emissive=emissive,
                    diffuse_texture=diffuse_texture, normal_texture=normal_texture,
                    orm_texture=orm_texture, inputs=raw_inputs)
            else:
                mat_path = materials.create_preview_material(
                    self._stage, name=base_name, color=color, metallic=metallic,
                    roughness=roughness, opacity=opacity, clearcoat=clearcoat,
                    clearcoat_roughness=clearcoat_roughness, ior=ior, emissive=emissive,
                    diffuse_texture=diffuse_texture, normal_texture=normal_texture,
                    roughness_texture=roughness_texture, metallic_texture=metallic_texture,
                    uv_set=uv_set, tex_scale=tex_scale, tex_rotate=tex_rotate,
                    tex_translate=tex_translate, inputs=raw_inputs)
            # `library` already bound targets[0]; bind the rest (and everything otherwise)
            bind_list = targets[1:] if library else targets
            authored = self._bind_many(bind_list, mat_path, prototype) if bind_list else []
            if library:
                authored = [done0, *authored]
        except Exception as exc:  # noqa: BLE001
            return self._err("material", exc)
        self._index_prims()
        # summary names the material by ref (if it has one) or leaf name — never the full path
        mat_label = self.refs.ref_for_path(mat_path) or mat_path.rsplit("/", 1)[-1]
        if bulk:
            refs = [self.refs.ref_for_path(p) for p in authored]
            labels = [r or p.rsplit("/", 1)[-1] for r, p in zip(refs, authored)]
            return Response(
                command="material",
                summary={"targets": len(authored), "matched": len(targets),
                         "material": mat_label},
                data={"material_path": mat_path, "bound_paths": authored,
                      "refs": [r for r in refs if r],
                      "text": f"{len(authored)} prim(s) ← material {mat_label}: "
                              + ", ".join(labels[:12])
                              + (" …" if len(labels) > 12 else "")})
        target_label = ref + (f"/{subset}" if subset else "")
        note = " (authored on prototype source)" if prototype else ""
        return Response(command="material",
                        summary={"ref": target_label, "material": mat_label},
                        data={"material_path": mat_path, "bound_paths": authored,
                              "text": f"{target_label} ← material {mat_label}{note}"})

    @_mutating
    def material_apply(self, library: str, bindings: list[dict]) -> Response:
        """Apply an exact heterogeneous material plan in one daemon transaction.

        Unlike ``material`` bulk selectors, this command performs no target inference:
        every target and exact source material prim is supplied by the caller. Distinct
        source materials are imported once, and all bindings form one undo entry.
        """
        from pxr import Sdf, UsdShade

        from usd_core import materials

        self._require_stage()
        if not isinstance(bindings, list) or not bindings:
            return self._err(
                "material.apply", ValueError("bindings must be a non-empty list")
            )
        if len(bindings) > 256:
            return self._err(
                "material.apply", ValueError("material plan exceeds 256 assignment groups")
            )
        lib_p = Path(library).expanduser()
        if not lib_p.is_absolute():
            lib_p = ((self.config.project_dir or Path.cwd()) / lib_p).resolve()
        library = str(lib_p)

        normalized: list[dict] = []
        target_owner: dict[str, str] = {}
        target_count = 0
        try:
            for index, raw_group in enumerate(bindings):
                if not isinstance(raw_group, dict):
                    raise ValueError(f"bindings[{index}] must be an object")
                material_name = str(raw_group.get("material_name") or "").strip()
                material_path = str(raw_group.get("material_path") or "").strip()
                if not material_name or not materials._is_valid_sdf_path_string(material_path):
                    raise ValueError(
                        f"bindings[{index}] requires a material name and exact absolute "
                        "material_path"
                    )
                source_path = Sdf.Path(material_path)
                if (
                    not source_path.IsAbsolutePath()
                    or not source_path.IsPrimPath()
                    or source_path == Sdf.Path.absoluteRootPath
                ):
                    raise ValueError(
                        f"bindings[{index}] requires a material name and exact absolute "
                        "material_path"
                    )
                prim_paths = raw_group.get("prim_paths")
                if not isinstance(prim_paths, list) or not prim_paths:
                    raise ValueError(f"bindings[{index}].prim_paths must be non-empty")
                normalized_paths: list[tuple[str, str]] = []
                for raw_path in prim_paths:
                    prim_path = str(raw_path or "").strip()
                    if not materials._is_valid_sdf_path_string(prim_path):
                        raise ValueError(
                            f"bindings[{index}] has invalid target path {prim_path!r}"
                        )
                    path = Sdf.Path(prim_path)
                    if (
                        not path.IsAbsolutePath()
                        or not path.IsPrimPath()
                        or path == Sdf.Path.absoluteRootPath
                    ):
                        raise ValueError(
                            f"bindings[{index}] has invalid target path {prim_path!r}"
                        )
                    prim = self._stage.GetPrimAtPath(prim_path)
                    if not prim or not prim.IsValid():
                        raise ValueError(f"bindings[{index}] target does not exist: {prim_path}")
                    if not prim.IsActive():
                        raise ValueError(f"bindings[{index}] target is inactive: {prim_path}")
                    if UsdShade.Material(prim):
                        raise ValueError(
                            f"bindings[{index}] target is itself a material: {prim_path}"
                        )
                    authored_path = self._authored_bind_path(prim_path, False)
                    previous_owner = target_owner.setdefault(authored_path, material_path)
                    if previous_owner != material_path:
                        raise ValueError(
                            "material plan assigns conflicting materials to authored target "
                            f"{authored_path}: {previous_owner}, {material_path}"
                        )
                    normalized_paths.append((prim_path, authored_path))
                    target_count += 1
                    if target_count > 4096:
                        raise ValueError("material plan exceeds 4096 target paths")
                normalized.append(
                    {
                        "material_name": material_name,
                        "material_path": material_path,
                        "paths": normalized_paths,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            return self._err("material.apply", exc)

        imported: dict[str, str] = {}
        applied_authored: set[str] = set()
        group_receipts: list[dict] = []
        default_prim = self._stage.GetDefaultPrim()
        default_path = default_prim.GetPath().pathString if default_prim else ""
        looks = self._stage.GetPrimAtPath(f"{default_path}/Looks")
        local_material_active = (
            {
                child.GetPath().pathString: child.IsActive()
                for child in looks.GetAllChildren()
            }
            if looks.IsValid()
            else {}
        )
        self.history.begin()
        try:
            for group in normalized:
                material_path = group["material_path"]
                local_material = imported.get(material_path)
                if local_material is None:
                    local_material = materials.import_library_material(
                        self._stage,
                        library_path=library,
                        material_name=group["material_name"],
                        source_prim_path=material_path,
                    )
                    if not local_material_active.get(local_material, False):
                        self._record(
                            "material.apply",
                            f"import {local_material}",
                            {"kind": "set_active", "path": local_material, "active": False},
                            {"kind": "set_active", "path": local_material, "active": True},
                        )
                        local_material_active[local_material] = True
                    imported[material_path] = local_material
                authored_paths: list[str] = []
                for prim_path, authored_path in group["paths"]:
                    if authored_path not in applied_authored:
                        previous = materials.bound_material(self._stage, authored_path)
                        materials.bind_material(self._stage, local_material, prim_path)
                        previous_direct = (
                            previous["bound_material_path"]
                            if previous["binding_type"] == "direct"
                            else None
                        )
                        self._record(
                            "material.apply",
                            f"bind {authored_path}",
                            self._material_bind_undo(authored_path, previous_direct),
                            {
                                "kind": "bind",
                                "path": authored_path,
                                "material": local_material,
                            },
                        )
                        applied_authored.add(authored_path)
                    authored_paths.append(authored_path)
                group_receipts.append(
                    {
                        "material_name": group["material_name"],
                        "material_source_path": material_path,
                        "bound_material_path": local_material,
                        "prim_paths": [path for path, _authored in group["paths"]],
                        "authored_paths": authored_paths,
                    }
                )
            self.history.commit(
                f"material apply {len(applied_authored)} prim(s), {len(imported)} material(s)"
            )
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("material.apply", exc)
        self._index_prims()
        return Response(
            command="material.apply",
            summary={
                "targets": target_count,
                "authored": len(applied_authored),
                "materials": len(imported),
                "groups": len(group_receipts),
            },
            data={
                "groups": group_receipts,
                "imported_material_paths": imported,
                "bound_paths": sorted(applied_authored),
                "text": (
                    f"{target_count} exact target(s) ← {len(imported)} material(s) "
                    "in one transaction"
                ),
            },
        )

    def _material_unbind_many(self, targets: list[str]) -> Response:
        """Bulk `--unbind`: clear the direct binding on every selected prim (one txn)."""
        from usd_core import materials
        cleared, seen = [], set()
        self.history.begin()
        try:
            for target in targets:
                done_path = self._authored_bind_path(target, False)
                if done_path in seen:
                    continue
                seen.add(done_path)
                prev = materials.bound_material(self._stage, done_path)
                if prev["binding_type"] != "direct":
                    continue
                materials.unbind_material(self._stage, done_path)
                self._record("material", f"unbind {done_path}",
                             {"kind": "bind", "path": done_path,
                              "material": prev["bound_material_path"]},
                             {"kind": "unbind", "path": done_path})
                cleared.append(done_path)
            self.history.commit(f"material unbind {len(cleared)} prim(s)")
        except Exception:
            self._abort_txn()
            raise
        return Response(command="material",
                        summary={"unbound": len(cleared), "matched": len(targets)},
                        data={"paths": cleared,
                              "text": f"unbound {len(cleared)} direct binding(s)"})

    def _material_unbind(self, ref: str, path: str, prev: dict) -> Response:
        """Remove a direct material binding (`material <ref> --unbind`).

        On an instance proxy this clears the binding authored on its instanceable root
        (mirroring how bind redirects there). A binding inherited from a regular ancestor
        is left alone — unbinding here would be a silent no-op, so we say where it lives.
        """
        from usd_core import materials
        materials.unbind_material(self._stage, path)
        after = materials.bound_material(self._stage, path)
        if after["bound_material_path"] == prev["bound_material_path"]:
            note = ("no material bound" if not prev["bound_material_path"] else
                    "binding is inherited from an ancestor — unbind that prim instead")
            return Response(command="material", summary={"ref": ref, "unbound": False},
                            data={"binding": after, "text": f"{ref}: {note}"},
                            issues=[Issue("info", note)])
        prev_path = prev["bound_material_path"]
        self._record("material", f"unbind {ref}",
                     {"kind": "bind", "path": path, "material": prev_path},
                     {"kind": "unbind", "path": path})
        label = self.refs.ref_for_path(prev_path) or prev_path.rsplit("/", 1)[-1]
        summary = {"ref": ref, "unbound": label}
        data = {"binding": after, "text": f"{ref}: unbound material {label}"}
        if after["bound_material_path"]:  # a weaker (referenced/baked) binding resurfaced
            now = after["bound_material_path"].rsplit("/", 1)[-1]
            summary["now_bound"] = now
            data["text"] += f" (weaker binding {now} now applies)"
        return Response(command="material", summary=summary, data=data)

    def _mat_label(self, path: str | None) -> str:
        if not path:
            return "(none)"
        return self.refs.ref_for_path(path) or path.rsplit("/", 1)[-1]

    def verify(self, file: str | None = None, strict: bool = False,
               **_ignored) -> Response:
        """One-shot deliverable verification — the macro every benchmark agent
        hand-rolled in 3-5 throwaway sessions: open the file FRESH from disk
        (never the live layer), then info + sublayers + validate + material
        audit, composed into a single verdict. Read-only by construction."""
        from pxr import Usd, UsdGeom

        from usd_core.audit import material_audit as _audit
        from usd_core.validate import validate_stage as _validate

        target = file or self._stage_path
        if not target:
            return self._err("verify", RuntimeError(
                "nothing to verify — pass a file or open a stage first"))
        target = os.path.abspath(target)
        if not os.path.exists(target):
            return self._err("verify", FileNotFoundError(target))
        stage = side = None
        try:
            # Verification must judge the bytes on disk. A plain
            # `Usd.Stage.Open` would reuse the process-global cached layer, so a
            # peer session's *unsaved* edits could turn an unfixed deliverable
            # into a PASS. `_open_unshared` reads a side copy in the same
            # directory instead, which keeps relative refs anchored. Strictly
            # read-only either way — nothing below authors anything.
            stage, side = self._open_unshared(target, tag="verify")
            n_prims = n_hidden = n_gprims = 0
            for _p in stage.Traverse(Usd.TraverseInstanceProxies()):
                n_prims += 1
                if _p.IsA(UsdGeom.Gprim):
                    n_gprims += 1
                    if (UsdGeom.Imageable(_p).ComputeVisibility()
                            == UsdGeom.Tokens.invisible):
                        n_hidden += 1
            up_axis = str(UsdGeom.GetStageUpAxis(stage))
            root = stage.GetRootLayer()
            subs = list(root.subLayerPaths)
            from pxr import Sdf
            dead_subs = [s for s in subs
                         if not (Sdf.Layer.FindOrOpenRelativeToLayer(root, s)
                                 or Sdf.Layer.FindOrOpen(s))]
            vreport = _validate(stage, fix=False)
            areport = _audit(stage)
            c = areport["counts"]
            # Portability: verify PASSed a deliverable with 10 live reference
            # arcs into ../shared/ (round 8) — dead_sublayers alone cannot see
            # refs/payloads/assets. External deps don't fail the verdict (shared
            # libraries are legitimate in-place), but they must be visible.
            ext_refs: list[str] = []
            try:
                from pxr import Ar, UsdUtils

                layers, assets, unresolved = UsdUtils.ComputeAllDependencies(target)
                here = os.path.realpath(target)
                deps = [lyr.realPath for lyr in layers if lyr.realPath]
                deps += [a for a in assets if a]
                base = os.path.dirname(here)

                def _dependency_root(path: str) -> str:
                    """Return the filesystem owner of a dependency identifier."""
                    if Ar.IsPackageRelativePath(path):
                        package, _member = Ar.SplitPackageRelativePathOuter(path)
                        return package
                    return path

                ext_refs = sorted(
                    {
                        os.path.relpath(os.path.realpath(_dependency_root(d)), base)
                        for d in deps
                        if os.path.realpath(_dependency_root(d)) != here
                    }
                )
                ext_refs += sorted(
                    f"{u} (unresolved)"
                    for u in unresolved
                    if os.path.realpath(_dependency_root(str(u))) != here
                )
            except Exception:  # noqa: BLE001 — best-effort; never sink a verify
                ext_refs = []
        except Exception as exc:  # noqa: BLE001
            return self._err("verify", exc)
        finally:
            stage = None  # release the side stage before removing its file
            self._discard_side_copy(side)
        ok = (vreport["ok"] and not dead_subs and c["unbound"] == 0
              and not areport["invalid_bindings"])
        # Round 9, task-03: the agent read "NOT self-contained … needs save
        # --flatten" twice and shipped anyway — a footnote never beats the
        # verdict line. Warnings now ride ON the verdict line, and --strict
        # demotes them to FAIL outright.
        warn_tags: list[str] = []
        if ext_refs:
            warn_tags.append(f"{len(ext_refs)} external deps")
        if n_hidden:
            warn_tags.append(f"{n_hidden} hidden renderables")
        strict_demoted = strict and ok and bool(warn_tags)
        if strict_demoted:
            ok = False
        summary = {"ok": ok, "file": os.path.basename(target), "prims": n_prims,
                   "renderables": c["renderables"], "unbound": c["unbound"],
                   "invalid_bindings": len(areport["invalid_bindings"]),
                   "validate_errors": vreport["counts"].get("errors", 0),
                   "validate_warnings": vreport["counts"].get("warnings", 0),
                   "dead_sublayers": len(dead_subs),
                   "up_axis": up_axis}
        if ext_refs:
            summary["external_deps"] = len(ext_refs)
        if n_hidden:
            summary["hidden_renderables"] = n_hidden
        if strict:
            summary["strict"] = True
        if strict_demoted:
            verdict = f"FAIL (strict: {', '.join(warn_tags)})"
        elif ok and warn_tags:
            verdict = (f"PASS with warnings ({', '.join(warn_tags)}) "
                       "— --strict treats these as FAIL")
        else:
            verdict = "PASS" if ok else "FAIL"
        lines = [f"{verdict} — {target}"]
        if n_hidden:
            # round 8: verify PASSed a deliverable in which leaked isolate
            # opinions hid 8018 of 8037 visible meshes — binding/validate
            # checks cannot see intent, but a hidden-count is a loud signal
            lines.append(f"  note: {n_hidden} of {n_gprims} renderables are "
                         "hidden by authored visibility — intended? "
                         "(`isolate --restore` / `show` if not)")
        if ext_refs:
            shown = ", ".join(ext_refs[:3])
            more = f" (+{len(ext_refs) - 3} more; --json lists all)" if len(ext_refs) > 3 else ""
            lines.append(f"  external deps: {shown}{more} — NOT self-contained; "
                         "a deliverable that must travel alone needs save --flatten")
        if dead_subs:
            lines.append(f"  dead sublayers: {', '.join(dead_subs[:5])}")
        for i in vreport["issues"][:10]:
            lines.append(f"  {i['severity'].upper()} [{i['check']}] {i['path']}: "
                         f"{i['message']}")
        if len(vreport["issues"]) > 10:
            lines.append(f"  … +{len(vreport['issues']) - 10} more issues (validate for all)")
        if not ok:
            summary["error_type"] = "verification"
        data: dict = {"text": "\n".join(lines)}
        if ext_refs:
            data["external_deps"] = ext_refs
        return Response(command="verify", ok=ok, summary=summary, data=data)

    def stats(self, **_ignored) -> Response:
        """Cheap scene overview: prim counts by type (instance-aware), instance and
        mesh totals, world bounds — the numbers agents previously computed with
        raw-pxr escape hatches on big scenes."""
        from pxr import Usd, UsdGeom

        self._require_stage()
        by_type: dict[str, int] = {}
        n_prims = n_instances = n_meshes = 0
        for prim in self._stage.Traverse(Usd.TraverseInstanceProxies()):
            n_prims += 1
            tp = str(prim.GetTypeName()) or "(untyped)"
            by_type[tp] = by_type.get(tp, 0) + 1
            if prim.IsInstance():
                n_instances += 1
            if prim.IsA(UsdGeom.Mesh):
                n_meshes += 1
        top = sorted(by_type.items(), key=lambda kv: -kv[1])[:15]
        lines = [f"{n:7}  {tp}" for tp, n in top]
        if len(by_type) > 15:
            lines.append(f"      …  {len(by_type) - 15} more types (--json for all)")
        summary = {"prims": n_prims, "meshes": n_meshes, "instances": n_instances,
                   "types": len(by_type)}
        try:
            bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                     [UsdGeom.Tokens.default_]) \
                .ComputeWorldBound(self._stage.GetPseudoRoot()).ComputeAlignedBox()
            size = bbox.GetSize()
            summary["size"] = [round(float(size[i]), 3) for i in range(3)]
        except Exception:  # noqa: BLE001 — bounds are a bonus
            pass
        return Response(command="stats", summary=summary,
                        data={"text": "\n".join(lines), "counts_by_type": by_type})

    def material_audit(self, effective: bool = False, include_subsets: bool = False,
                       under: str | None = None,
                       **_ignored) -> Response:
        """Complete binding audit: bound/unbound renderables, inherited bindings, subset
        coverage, invalid binding targets, unused materials (`material audit`)."""
        from usd_core.audit import material_audit
        self._require_stage()
        self._index_prims()
        try:
            report = material_audit(self._stage, effective=effective,
                                    include_subsets=include_subsets,
                                    under=str(self._path_of(under)) if under else None)
        except Exception as exc:  # noqa: BLE001
            return self._err("material-audit", exc)
        c = report["counts"]
        # JSON consumers get {ref, path} pairs, not bare path arrays — audit was
        # the single largest ref-free path emitter in the round-7 traces (2,068
        # lines), directly feeding path-typed follow-up commands
        report["unbound"] = [{"ref": self.refs.ref_for_path(pp), "path": pp}
                             for pp in report.get("unbound_paths", [])
                             if not isinstance(pp, str) or pp.startswith("/")]
        lines = [f"renderables={c['renderables']} direct={c['bound_direct']} "
                 f"inherited={c['bound_inherited']} unbound={c['unbound']} "
                 f"unused_materials={c['materials_unused']}"]
        if report["unbound_paths"]:
            labels = [self.refs.ref_for_path(p) or p.rsplit("/", 1)[-1]
                      for p in report["unbound_paths"][:20]]
            lines.append("  unbound: " + ", ".join(labels)
                         + (" …" if len(report["unbound_paths"]) > 20 else ""))
        # aggregate invalid bindings by (problem, target): validate/audit used to
        # re-print the identical line hundreds of times ("target prim does not
        # exist" x242 in one round-7 run)
        groups: dict = {}
        markers = []
        for bad in report["invalid_bindings"]:
            if isinstance(bad, str):  # "... N more" cap marker (audit._capped)
                markers.append(bad)
                continue
            key = (bad.get("problem", "?"), bad.get("target", "?"))
            entry = groups.setdefault(key, {"n": 0, "first": bad})
            entry["n"] += 1
        for (problem, target), entry in sorted(groups.items(),
                                               key=lambda kv: -kv[1]["n"]):
            first = self._mat_label(entry["first"]["path"])
            more = f" (+{entry['n'] - 1} more prims)" if entry["n"] > 1 else ""
            lines.append(f"  INVALID x{entry['n']}: → {target} ({problem}) — "
                         f"e.g. {first}{more}")
        for m in markers:
            lines.append(f"  INVALID: {m}")
        if report["unused_materials"]:
            lines.append("  unused: " + ", ".join(
                self._mat_label(m) for m in report["unused_materials"][:20]))
        for fam in report.get("subset_families", []):
            if isinstance(fam, str):  # "... N more" cap marker (audit._capped)
                lines.append(f"  subsets: {fam}")
                continue
            lines.append(
                f"  subsets on {self._mat_label(fam['mesh'])}: {fam['subsets']} in "
                f"materialBind ({fam['family_type']}), faces "
                f"{fam['faces_covered']}/{fam.get('mesh_face_count', '?')}"
                + ("" if fam["valid"] else f" — INVALID: {fam.get('reason')}"))
        report["text"] = "\n".join(lines)
        summary = {"renderables": c["renderables"], "unbound": c["unbound"],
                   "invalid": c["invalid_binding_targets"],
                   "unused_materials": c["materials_unused"]}
        if include_subsets:
            summary["subset_problems"] = c.get("subset_problems", 0)
        return Response(command="material-audit", ok=report["ok"], summary=summary,
                        data=report)

    def appearance_audit(self, **_ignored) -> Response:
        """Report effective material, shader, and display-primvar appearance."""
        from usd_core.appearance import audit_appearance

        self._require_stage()
        try:
            report = audit_appearance(self._stage)
        except Exception as exc:  # noqa: BLE001
            return self._err("appearance-audit", exc)
        counts = report["counts"]
        text = (
            f"clear={str(report['clear']).lower()} "
            f"bindings={counts['effective_material_bindings']} "
            f"shaders={counts['effective_shader_appearances']} "
            f"direct_shader_outputs={counts['direct_shader_outputs']} "
            f"display_values={counts['display_values']} "
            f"instance_proxies={counts['instance_proxies']}"
        )
        return Response(
            command="appearance-audit",
            summary={
                "clear": report["clear"],
                "overlay_active": report["overlay_active"],
                **counts,
            },
            data={**report, "text": text},
        )

    @_mutating
    def appearance_clear(self, **_ignored) -> Response:
        """Mask all composed appearance in the strongest, undoable session layer."""
        from usd_core.appearance import clear_appearance, restore_session_layer

        self._require_stage()
        try:
            result = clear_appearance(self._stage)
            self._index_prims()
            self._record(
                "appearance.clear",
                "clear composed appearance",
                {
                    "kind": "session_layer_snapshot",
                    "layer": result.before_layer,
                    "edit_target": result.previous_edit_target,
                },
                {
                    "kind": "session_layer_snapshot",
                    "layer": result.after_layer,
                    "edit_target": "session",
                },
            )
        except Exception as exc:  # noqa: BLE001
            # clear_appearance rolls back mutation failures. This second guard
            # covers failures while re-indexing or recording the undo entry.
            if "result" in locals():
                try:
                    restore_session_layer(
                        self._stage, result.before_layer, result.previous_edit_target
                    )
                    self._index_prims()
                except Exception as rollback_exc:  # noqa: BLE001
                    return self._err(
                        "appearance-clear",
                        RuntimeError(
                            f"appearance clear failed: {exc}; "
                            f"rollback also failed: {rollback_exc}"
                        ),
                    )
            return self._err("appearance-clear", exc)

        before = result.before_audit["counts"]
        after = result.after_audit["counts"]
        summary = {
            "clear": result.after_audit["clear"],
            "bindings_masked": result.blocked_bindings,
            "display_attributes_masked": result.blocked_display_attributes,
            "direct_shader_outputs_masked": result.blocked_shader_outputs,
            "shader_appearances_masked": before["effective_shader_appearances"],
            "deinstanced_roots": len(result.deinstanced_roots),
            "edit_layer": "session",
            "save_requires_flatten": True,
        }
        text = (
            "composed appearance is clear in the session layer; the source "
            "layer and file are unchanged. New material decisions now author "
            "above the mask. Persist only to a derivative with "
            "`save OUTPUT.usd --flatten`."
        )
        return Response(
            command="appearance-clear",
            summary=summary,
            data={
                "before": before,
                "after": after,
                "deinstanced_roots": list(result.deinstanced_roots),
                "audit": result.after_audit,
                "text": text,
            },
        )

    def validate_stage(self, fix: bool = False, **_ignored) -> Response:
        """General USD validation (`validate [--fix]`): composition, material bindings,
        subset families, unresolved assets, invalid shaders, renderer compatibility."""
        from usd_core.validate import validate_stage
        self._require_stage()
        if fix and self.read_only:
            # plain `validate` stays available to readers; --fix authors repairs
            # onto the writer's shared live layer
            return self._err("validate", RuntimeError(
                f"session '{self.name}' is read-only (opened with --read-only) — "
                "--fix would mutate the shared layer; run `validate` without --fix, "
                "or fix from the writer session"))
        try:
            report = validate_stage(self._stage, fix=fix)
            if fix and report.get("fixed"):
                self._mutation_epoch += 1  # repairs are unsaved edits too
                self._viewer_revision += 1
                self._invalidate_viewer_line_geometry()
        except Exception as exc:  # noqa: BLE001
            return self._err("validate", exc)
        lines = []
        for f in report["fixed"]:
            lines.append(f"fixed: {f}")
        # group repeated issues: one round-7 stage re-printed the identical
        # "binding target does not exist" line 111x on EVERY validate call
        grouped: dict = {}
        for i in report["issues"]:
            key = (i["severity"], i["check"], i["message"])
            grouped.setdefault(key, []).append(i["path"])
        for (severity, check, message), paths in grouped.items():
            ref0 = self.refs.ref_for_path(paths[0]) or paths[0]
            if len(paths) == 1:
                lines.append(f"{severity.upper()} [{check}] {ref0}: {message}")
            else:
                lines.append(f"{severity.upper()} x{len(paths)} [{check}] {message} "
                             f"— e.g. {ref0} (+{len(paths) - 1} more; --json lists all)")
        for row in report["renderer_compat"]:
            if row["shader_kind"] != "UsdPreviewSurface":  # only the noteworthy ones
                backends = ", ".join(f"{k}: {v}" for k, v in row["backends"].items())
                lines.append(f"compat {self._mat_label(row['material'])} "
                             f"[{row.get('mdl_module') or row['shader_kind']}] — {backends}")
        if not lines:
            lines = ["stage is clean — no issues found"]
        report["text"] = "\n".join(lines)
        return Response(command="validate", ok=report["ok"],
                        summary={"ok": report["ok"], **report["counts"],
                                 "fixed": len(report["fixed"])},
                        data=report)

    def subsets(self, ref: str, family: str | None = None, **_ignored) -> Response:
        """List + validate the GeomSubset face assignments under a mesh (or subtree)."""
        from pxr import Usd, UsdGeom
        from usd_core.subsets import MATERIAL_BIND_FAMILY, list_subsets, validate_family
        self._require_stage()
        self._index_prims()
        try:
            root_path = self._path_of(ref)
            root = self._stage.GetPrimAtPath(root_path)
            meshes = [p.GetPath().pathString
                      for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                      if p.IsA(UsdGeom.Mesh) and p.IsActive()]
            rows, families, lines, problems = [], [], [], 0
            for mesh_path in meshes:
                subs = list_subsets(self._stage, mesh_path, family=family)
                if not subs:
                    continue
                fam = validate_family(self._stage, mesh_path,
                                      family or MATERIAL_BIND_FAMILY)
                families.append(fam)
                mesh_label = self.refs.ref_for_path(mesh_path) or mesh_path.rsplit("/", 1)[-1]
                lines.append(f"{mesh_label}: {len(subs)} subset(s), faces "
                             f"{fam['faces_covered']}/{fam.get('mesh_face_count', '?')} "
                             f"covered ({fam['family_type']})"
                             + ("" if fam["valid"] else f" — INVALID: {fam.get('reason')}"))
                for s in subs:
                    rows.append(s)
                    problems += len(s["problems"])
                    lines.append(f"  {s['name']} [{s['family'] or 'no family'}] "
                                 f"{s['face_count']} faces → "
                                 f"{self._mat_label(s['bound_material_path'])}"
                                 + (f"  ⚠ {'; '.join(s['problems'])}" if s["problems"] else ""))
        except Exception as exc:  # noqa: BLE001
            return self._err("subsets", exc)
        if not rows:
            lines = [f"{ref}: no GeomSubsets" + (f" in family '{family}'" if family else "")]
        return Response(command="subsets",
                        summary={"ref": ref, "subsets": len(rows), "problems": problems},
                        data={"subsets": rows, "families": families,
                              "text": "\n".join(lines)})

    @_mutating
    def show(self, refs: list[str] | None = None, **_ignored) -> Response:  # PR-4.7
        return self._set_vis(refs, True, "show")

    @_mutating
    def hide(self, refs: list[str] | None = None, **_ignored) -> Response:  # PR-4.7
        return self._set_vis(refs, False, "hide")

    def _set_vis(self, refs, visible: bool, command: str) -> Response:
        from usd_core import edit
        from usd_core.query import _visible
        self._require_stage()
        done = []
        try:
            self.history.begin()
            for r in refs or []:
                path = self._edit_path_of(r)
                prev = _visible(self._stage.GetPrimAtPath(path))
                edit.set_visibility(self._stage, path, visible)
                self._record(command, f"{command} {r}",
                             {"kind": "set_vis", "path": path, "visible": prev},
                             {"kind": "set_vis", "path": path, "visible": visible})
                done.append(path)
            self.history.commit(f"{command} {len(done)}")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err(command, exc)
        return Response(command=command, summary={command: len(done)}, data={"paths": done})

    def select(self, refs: list[str] | None = None, **_ignored) -> Response:  # PR-4.8
        self._require_stage()
        try:
            for r in refs or []:
                path = self._path_of(r)
                if path not in self._selection:
                    self._selection.append(path)
        except Exception as exc:  # noqa: BLE001
            return self._err("select", exc)
        return self.selection()

    def deselect(self, refs: list[str] | None = None, **_ignored) -> Response:  # PR-4.8
        if not refs:
            self._selection = []
        else:
            drop = {self._path_of(r) for r in refs}
            self._selection = [p for p in self._selection if p not in drop]
        return self.selection()

    def selection(self, **_ignored) -> Response:  # PR-4.8
        rows = [{"ref": self.refs.ref_for_path(p), "path": p} for p in self._selection]
        return Response(command="selection", summary={"count": len(rows)},
                        data={"selection": rows,
                              "refs": [r["ref"] for r in rows if r["ref"]]})

    @_mutating
    def isolate(self, refs: list[str] | None = None, restore: bool = False,
                **_ignored) -> Response:
        """Show only the given prims (and their ancestors); hide the rest. Visual
        scene operation. `--restore` removes exactly the opinions isolate authored:
        a save while isolation is live bakes the hidden state into the file
        (round 8: temporary Jack-isolation opinions shipped in a deliverable and
        cost a 65-minute checkpoint-restore + full re-authoring replay)."""
        from usd_core import edit
        from pxr import Sdf, UsdGeom
        self._require_stage()
        if restore:
            return self._isolate_restore()
        try:
            keep_paths = [self._path_of(r) for r in (refs or self._selection)]
            if not keep_paths:
                raise ValueError("isolate needs refs or a current selection")
            keep = set()
            for p in keep_paths:
                parts = p.strip("/").split("/")
                for i in range(len(parts)):
                    keep.add("/" + "/".join(parts[: i + 1]))
                # descendants stay visible too
            self.history.begin()
            layer = self._stage.GetEditTarget().GetLayer()
            for prim in self._stage.Traverse():
                if not prim.IsA(UsdGeom.Imageable):
                    continue
                p = prim.GetPath().pathString
                under_keep = any(p == k or p.startswith(k + "/") for k in keep_paths)
                want = under_keep or p in keep
                from usd_core.query import _visible
                prev = _visible(prim)
                if prev != want:
                    if p not in self._isolation:
                        # first touch: remember the edit layer's PRIOR opinion
                        spec = layer.GetObjectAtPath(
                            Sdf.Path(p).AppendProperty("visibility"))
                        self._isolation[p] = (
                            str(spec.default) if spec is not None
                            and spec.HasDefaultValue() else None)
                    edit.set_visibility(self._stage, p, want)
                    self._record("isolate", f"isolate vis {p}",
                                 {"kind": "set_vis", "path": p, "visible": prev},
                                 {"kind": "set_vis", "path": p, "visible": want})
            self.history.commit("isolate")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("isolate", exc)
        return Response(command="isolate", summary={"isolated": len(keep_paths)},
                        data={"paths": keep_paths},
                        issues=[Issue("info",
                                      "temporary view: run `isolate --restore` "
                                      "before saving a deliverable")])

    def _isolate_restore(self) -> Response:
        """Remove the visibility opinions `isolate` authored, restoring any prior
        authored tokens; prims that had NO opinion get their spec deleted (a
        `show`/`hide` would author fresh litter instead)."""
        from pxr import Sdf, UsdGeom

        from usd_core.query import _visible
        if not self._isolation:
            return Response(command="isolate", summary={"restored": 0},
                            data={"text": "no isolate opinions to restore"})
        layer = self._stage.GetEditTarget().GetLayer()
        restored = 0
        self.history.begin()
        try:
            for p, prev in list(self._isolation.items()):
                prim = self._stage.GetPrimAtPath(p)
                if not prim or not prim.IsValid():
                    self._isolation.pop(p)
                    continue
                was = _visible(prim)
                if prev is None:
                    spec = layer.GetObjectAtPath(
                        Sdf.Path(p).AppendProperty("visibility"))
                    if spec is not None:
                        layer.GetPrimAtPath(p).RemoveProperty(spec)
                else:
                    UsdGeom.Imageable(prim).CreateVisibilityAttr().Set(prev)
                self._record("isolate", f"isolate restore vis {p}",
                             {"kind": "set_vis", "path": p, "visible": was},
                             {"kind": "set_vis", "path": p,
                              "visible": _visible(prim)})
                self._isolation.pop(p)
                restored += 1
            self.history.commit("isolate")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("isolate", exc)
        return Response(command="isolate", summary={"restored": restored},
                        data={"text": f"removed isolate visibility opinions on "
                                      f"{restored} prim(s)"})

    @_mutating
    def set(self, ref: str, attr: str, value: str, **_ignored) -> Response:  # PR-4.11
        from usd_core import edit
        self._require_stage()
        # Stage-level metadata: `set / upAxis Z` / `set / metersPerUnit 0.01`. These live on
        # the pseudo-root (not an attribute), so route them to the stage metadata API instead
        # of erroring with "cannot create attribute spec on the pseudo-root".
        if ref in ("/", "") and attr in ("upAxis", "metersPerUnit"):
            from pxr import UsdGeom
            try:
                if attr == "upAxis":
                    tok = UsdGeom.Tokens.z if str(value).lower().startswith("z") else UsdGeom.Tokens.y
                    old = UsdGeom.GetStageUpAxis(self._stage)
                    UsdGeom.SetStageUpAxis(self._stage, tok)
                    new = str(tok)
                else:
                    old = UsdGeom.GetStageMetersPerUnit(self._stage)
                    UsdGeom.SetStageMetersPerUnit(self._stage, float(value))
                    new = float(value)
            except Exception as exc:  # noqa: BLE001
                return self._err("set", exc)
            return Response(command="set", summary={"stage_metadata": attr},
                            data={"old": str(old), "new": str(new)})
        if attr == "instanceable":
            try:
                path = self._path_of(ref)
                prim = self._stage.GetPrimAtPath(path)
                if not prim or not prim.IsValid():
                    raise ValueError(f"no prim at {path}")
                normalized = str(value).strip().lower()
                if normalized not in {"true", "false"}:
                    raise ValueError("instanceable metadata requires true or false")
                new = normalized == "true"
                had = prim.HasAuthoredMetadata("instanceable")
                old = prim.IsInstanceable()
                prim.SetInstanceable(new)
                self._index_prims()
            except Exception as exc:  # noqa: BLE001
                return self._err("set", exc)
            self._record(
                "set",
                f"set {ref}.instanceable={normalized}",
                {
                    "kind": "set_instanceable",
                    "path": path,
                    "value": old,
                    "clear": not had,
                },
                {
                    "kind": "set_instanceable",
                    "path": path,
                    "value": new,
                    "clear": False,
                },
            )
            return Response(
                command="set",
                summary={"ref": ref, "metadata": "instanceable"},
                data={"old": old, "new": new},
            )
        try:
            path = self._edit_path_of(ref)
            prim = self._stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                raise ValueError(f"no prim at {path}")
            if prim.GetRelationship(attr):
                # schema relationships (physics:body0, material:binding, …) take prim-path
                # targets — authoring an attribute would silently corrupt the prim.
                old_t, new_t = edit.set_relationship(self._stage, path, attr, value)
                self._record("set", f"set {ref}.{attr}={value}",
                             {"kind": "rel_targets", "path": path, "attr": attr,
                              "targets": old_t},
                             {"kind": "rel_targets", "path": path, "attr": attr,
                              "targets": new_t})
                return Response(command="set", summary={"ref": ref, "attr": attr,
                                                        "relationship": True},
                                data={"old": old_t, "new": new_t})
            a = prim.GetAttribute(attr)
            had = bool(a and a.HasAuthoredValue())
            raw_old = a.Get() if had else None
            old_js, new_js, created = edit.set_attribute(self._stage, path, attr, value)
            raw_new = self._stage.GetPrimAtPath(path).GetAttribute(attr).Get()
        except Exception as exc:  # noqa: BLE001
            return self._err("set", exc)
        self._record("set", f"set {ref}.{attr}={value}",
                     {"kind": "raw_attr", "path": path, "attr": attr, "value": raw_old,
                      "unset": not had, "remove": created},
                     {"kind": "raw_attr", "path": path, "attr": attr, "value": raw_new})
        return Response(command="set", summary={"ref": ref, "attr": attr},
                        data={"old": old_js, "new": new_js})

    @_mutating
    def import_(self, source: str, at=None, name=None, **_ignored) -> Response:  # PR-4.12
        """Import an external asset as a referenced prim under the default parent."""
        from usd_core import edit
        self._require_stage()
        try:
            parent = self._default_parent()
            base = name or Path(source).stem.replace(" ", "_")
            path = edit.unique_child_path(self._stage, parent, base)
            prim = self._stage.DefinePrim(path, "Xform")
            prim.GetReferences().AddReference(source)
            if at:
                edit.set_translate(self._stage, path, at)
        except Exception as exc:  # noqa: BLE001
            return self._err("import", exc)
        self._index_prims()
        self._record("import", f"import {source}",
                     {"kind": "set_active", "path": path, "active": False},
                     {"kind": "set_active", "path": path, "active": True})
        return Response(command="import",
                        summary={"ref": self.refs.ref_for_path(path), "name": path.rsplit("/", 1)[-1]},
                        data={"path": path})

    @_mutating
    def align(self, ref: str, to: str, axis: str = "x", mode: str = "center", **_ignored) -> Response:  # PR-4.13
        from usd_core import edit
        from usd_core.spatial import get_world_bbox
        self._require_stage()
        try:
            src_p, dst_p = self._edit_path_of(ref), self._path_of(to)
            sb, db = get_world_bbox(self._stage, src_p), get_world_bbox(self._stage, dst_p)
            if not sb or not db:
                raise RuntimeError("both prims need geometry bounds to align")
            ax = {"x": 0, "y": 1, "z": 2}[axis]
            key = {"min": "min", "center": "center", "max": "max"}[mode]
            delta = db[key][ax] - sb[key][ax]
            d = [None, None, None]
            d[ax] = {"mode": "relative", "value": delta}
            before, after = edit.apply_transform(self._stage, src_p,
                                                 **{f"t{axis}": d[ax]})
        except Exception as exc:  # noqa: BLE001
            return self._err("align", exc)
        self._record("align", f"align {ref} to {to}",
                     {"kind": "set_trs", "path": src_p, "trs": before},
                     {"kind": "set_trs", "path": src_p, "trs": after})
        return Response(command="align", summary={"ref": ref, "to": to, "axis": axis,
                        "mode": mode}, data={"after": after})

    @_mutating
    def scatter(self, ref: str, on: str, count: int = 10, seed=None, **_ignored) -> Response:  # PR-4.14
        import random
        from usd_core import edit
        from usd_core.spatial import get_world_bbox
        self._require_stage()
        try:
            src_p, surf_p = self._edit_path_of(ref), self._path_of(on)
            sb = get_world_bbox(self._stage, surf_p)
            if not sb:
                raise RuntimeError(f"{on} has no bounds to scatter on")
            rng = random.Random(seed)
            mn, mx = sb["min"], sb["max"]
            top_y = mx[1]
            made = []
            self.history.begin()
            for _ in range(int(count)):
                x = rng.uniform(mn[0], mx[0]); z = rng.uniform(mn[2], mx[2])
                dst = edit.duplicate_prim(self._stage, src_p, at=[x, top_y, z])
                self._record("scatter", f"scatter copy {dst}",
                             {"kind": "set_active", "path": dst, "active": False},
                             {"kind": "set_active", "path": dst, "active": True})
                made.append(dst)
            self.history.commit(f"scatter {len(made)}")
        except Exception as exc:  # noqa: BLE001
            self._abort_txn()
            return self._err("scatter", exc)
        self._index_prims()
        return Response(command="scatter", summary={"created": len(made)}, data={"paths": made})

    def _default_parent(self) -> str:
        dp = self._stage.GetDefaultPrim()
        return dp.GetPath().pathString if dp and dp.IsValid() and dp.GetPath().pathString != "/" else ""

    # -- camera pan/zoom (PR-6.6/6.7) --------------------------------------------
    @_mutating
    def camera_pan(self, delta, **_ignored) -> Response:
        from usd_core.camera import author_camera
        from pxr import Gf, UsdGeom
        self._require_stage()
        if not self._active_cam:
            self.camera_fit(None)
        prim = self._stage.GetPrimAtPath(self._active_cam)
        m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        right = Gf.Vec3d(m[0][0], m[0][1], m[0][2]).GetNormalized()
        up = Gf.Vec3d(m[1][0], m[1][1], m[1][2]).GetNormalized()
        pos = m.ExtractTranslation()
        dx, dy = (delta or [0, 0])[:2]
        new_pos = pos + right * float(dx) + up * float(dy)
        # keep current look direction
        fwd = -Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetNormalized()
        author_camera(self._stage, self._active_cam, list(new_pos), list(new_pos + fwd))
        return Response(command="camera.pan", summary={"camera": self._active_cam, "delta": [dx, dy]})

    @_mutating
    def camera_zoom(self, factor: float, **_ignored) -> Response:
        from usd_core.camera import author_camera
        from pxr import Gf, UsdGeom
        self._require_stage()
        if not self._active_cam:
            self.camera_fit(None)
        prim = self._stage.GetPrimAtPath(self._active_cam)
        m = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
        pos = m.ExtractTranslation()
        fwd = -Gf.Vec3d(m[2][0], m[2][1], m[2][2]).GetNormalized()
        # move along forward by (1 - 1/factor) of distance to a notional target one unit ahead
        rng = self._target_range(None)
        center = (rng.GetMin() + rng.GetMax()) / 2.0
        to_center = Gf.Vec3d(*center) - pos
        new_pos = pos + to_center * (1.0 - 1.0 / max(1e-3, float(factor)))
        author_camera(self._stage, self._active_cam, list(new_pos), list(new_pos + fwd))
        return Response(command="camera.zoom", summary={"camera": self._active_cam, "factor": factor})

    # -- spatial (PR-5) ----------------------------------------------------------
    def _center(self, path: str):
        from usd_core.spatial import get_world_bbox
        b = get_world_bbox(self._stage, path)
        return b["center"] if b else None

    def _is_object(self, prim, type_=None) -> bool:
        """A spatial-query candidate: an explicit type match, else actual geometry (Gprim)
        or a native-instance root (its Gprims hide behind instance proxies, and the whole
        instance is the semantic "object" anyway).

        Without this, container Xforms/Scopes (whose bounds enclose their children) and the
        scene root dominate every result, which isn't what 'nearest/within/raycast object'
        means."""
        from pxr import UsdGeom
        if type_:
            return str(prim.GetTypeName()).lower() == type_.lower()
        return prim.IsA(UsdGeom.Gprim) or prim.IsInstance()

    def _rows_text(self, rows) -> str:
        """Readable listing keyed on refs — `@ref [Type] "name"` (+distance). Full SdfPaths
        stay in the JSON `results`/`resolve`, never the default text (usd-cli operates on refs)."""
        out = []
        for r in rows:
            ref = r.get("ref") or "-"
            extra = f"  d={r['distance']}" if r.get("distance") is not None else ""
            out.append(f'{ref:>5} [{r.get("type", "")}] "{r.get("name", "")}"{extra}')
        return "\n".join(out) if out else "(none)"

    def _row(self, path, distance=None) -> dict:
        prim = self._stage.GetPrimAtPath(path)
        r = {"ref": self.refs.ref_for_path(path), "path": path,
             "type": str(prim.GetTypeName()) if prim else "", "name": prim.GetName() if prim else ""}
        if distance is not None:
            r["distance"] = round(distance, 4)
        return r

    def _result(self, command, summary, rows) -> Response:
        summary = {**summary, "count": len(rows)}
        return Response(command=command, summary=summary,
                        data={"results": rows, "refs": [r["ref"] for r in rows if r["ref"]],
                              "text": self._rows_text(rows)})

    def distance(self, a: str, b: str, **_ignored) -> Response:  # PR-5.7
        self._require_stage()
        try:
            pa, pb = self._path_of(a), self._path_of(b)
            ca, cb = self._center(pa), self._center(pb)
            if ca is None or cb is None:
                raise RuntimeError("both prims need geometry bounds")
            d = sum((ca[i] - cb[i]) ** 2 for i in range(3)) ** 0.5
        except Exception as exc:  # noqa: BLE001
            return self._err("distance", exc)
        return Response(command="distance", summary={"a": a, "b": b, "distance": round(d, 4)},
                        data={"distance": d, "a_center": ca, "b_center": cb,
                              "text": f"{a} → {b}: {round(d, 4)} (centers {[round(x,3) for x in ca]} "
                                      f"→ {[round(x,3) for x in cb]})"})

    def _ranked_by_distance(self, ref, type_=None):
        base = self._center(self._path_of(ref))
        if base is None:
            raise RuntimeError(f"{ref} has no bounds")
        ref_path = self._path_of(ref)
        rows = []
        for prim in self._stage.Traverse():
            p = prim.GetPath().pathString
            if not prim.IsActive() or p == ref_path or not self._is_object(prim, type_):
                continue
            c = self._center(p)
            if c is None:
                continue
            d = sum((base[i] - c[i]) ** 2 for i in range(3)) ** 0.5
            rows.append((d, p))
        rows.sort(key=lambda x: x[0])
        return rows

    def nearest(self, ref: str, type=None, count: int = 1, **_ignored) -> Response:  # PR-5.3
        self._require_stage(); self._index_prims()
        try:
            rows = self._ranked_by_distance(ref, type)[: int(count)]
        except Exception as exc:  # noqa: BLE001
            return self._err("nearest", exc)
        return self._result("nearest", {"ref": ref},
                            [self._row(p, d) for d, p in rows])

    def within(self, ref: str, radius: float, **_ignored) -> Response:  # PR-5.4
        self._require_stage(); self._index_prims()
        try:
            rows = [(d, p) for d, p in self._ranked_by_distance(ref) if d <= float(radius)]
        except Exception as exc:  # noqa: BLE001
            return self._err("within", exc)
        return self._result("within", {"ref": ref, "radius": radius},
                            [self._row(p, d) for d, p in rows])

    def overlapping(self, ref: str, **_ignored) -> Response:  # PR-5.5
        from usd_core.spatial import get_world_bbox
        self._require_stage(); self._index_prims()
        try:
            tp = self._path_of(ref)
            tb = get_world_bbox(self._stage, tp)
            if not tb:
                raise RuntimeError(f"{ref} has no bounds")
            hits = []
            for prim in self._stage.Traverse():
                p = prim.GetPath().pathString
                if not prim.IsActive() or p == tp or not self._is_object(prim):
                    continue
                b = get_world_bbox(self._stage, p)
                if b and self._aabb_overlap(tb, b):
                    hits.append(p)
        except Exception as exc:  # noqa: BLE001
            return self._err("overlapping", exc)
        return self._result("overlapping", {"ref": ref}, [self._row(p) for p in hits])

    @staticmethod
    def _aabb_overlap(a, b) -> bool:
        return all(a["min"][i] <= b["max"][i] and a["max"][i] >= b["min"][i] for i in range(3))

    def _directional(self, name: str, ref: str, type_=None, count=None):
        """Objects positioned in a cardinal direction relative to a ref (PR-5.6)."""
        axis_sign = {"right": (0, 1), "left": (0, -1), "above": (1, 1), "below": (1, -1),
                     "front": (2, 1), "behind": (2, -1)}[name]
        self._require_stage(); self._index_prims()
        try:
            ref_path = self._path_of(ref)
            base = self._center(ref_path)
            if base is None:
                raise RuntimeError(f"{ref} has no bounds")
            ax, sign = axis_sign
            rows = []
            for prim in self._stage.Traverse():
                p = prim.GetPath().pathString
                if not prim.IsActive() or p == ref_path or not self._is_object(prim, type_):
                    continue
                c = self._center(p)
                if c is None or (c[ax] - base[ax]) * sign <= 0:
                    continue
                rows.append((abs(c[ax] - base[ax]), p))
            rows.sort(key=lambda x: x[0])
        except Exception as exc:  # noqa: BLE001
            return self._err(name, exc)
        if count:
            rows = rows[: int(count)]
        return self._result(name, {"ref": ref}, [self._row(p, d) for d, p in rows])

    def above(self, ref, type=None, count=None, **_): return self._directional("above", ref, type, count)
    def below(self, ref, type=None, count=None, **_): return self._directional("below", ref, type, count)
    def left(self, ref, type=None, count=None, **_): return self._directional("left", ref, type, count)
    def right(self, ref, type=None, count=None, **_): return self._directional("right", ref, type, count)
    def front(self, ref, type=None, count=None, **_): return self._directional("front", ref, type, count)
    def behind(self, ref, type=None, count=None, **_): return self._directional("behind", ref, type, count)

    def raycast(self, origin=None, direction=None, screen=None, **_ignored) -> Response:  # PR-5.1
        """Ray vs scene AABBs (slab test) — returns the nearest hit ref/point/distance."""
        self._require_stage(); self._index_prims()
        from usd_core.spatial import get_world_bbox
        try:
            if screen:
                o, dvec = self._screen_ray(screen)
            elif origin and direction:
                o, dvec = [float(x) for x in origin], [float(x) for x in direction]
            else:
                raise ValueError("raycast needs origin+direction, or --screen u,v with an active camera")
            best = None
            for prim in self._stage.Traverse():
                # only actual geometry — container Xforms/Scopes enclose everything and would
                # always win the nearest-bbox test (returning e.g. the scene root).
                if not prim.IsActive() or not self._is_object(prim):
                    continue
                b = get_world_bbox(self._stage, prim.GetPath().pathString)
                if not b:
                    continue
                t = self._ray_aabb(o, dvec, b["min"], b["max"])
                if t is not None and (best is None or t < best[0]):
                    best = (t, prim.GetPath().pathString)
        except Exception as exc:  # noqa: BLE001
            return self._err("raycast", exc)
        if best is None:
            return Response(command="raycast", summary={"hit": False},
                            data={"text": "no hit"})
        t, p = best
        ref = self.refs.ref_for_path(p)
        # float() strips numpy scalars — screen rays come back as np.float64 and
        # their list repr leaked `np.float64(239.4127)` into the hit line
        dist = round(float(t), 4)
        hit_pt = [round(float(o[i] + dvec[i] * t), 4) for i in range(3)]
        prim = self._stage.GetPrimAtPath(p)
        return Response(command="raycast", summary={"hit": True, "ref": ref},
                        data={"ref": ref, "path": p, "type": str(prim.GetTypeName()),
                              "name": prim.GetName(), "distance": dist, "point": hit_pt,
                              "text": f'hit {ref or "-"} [{prim.GetTypeName()}] '
                                      f'"{prim.GetName()}"  d={dist} at {hit_pt}'})

    def _screen_ray(self, screen):
        """Build a world-space ray from a screen pixel (u,v) through the active camera —
        the 'pick' primitive for visual grounding."""
        import numpy as np
        from usd_core.camera import camera_params
        if not self._active_cam:
            raise RuntimeError("no active camera for screen-space raycast (run camera fit first)")
        w, h = [int(x) for x in str(self.config.render.get("resolution", "1024x1024")).lower().split("x")]
        p = camera_params(self._stage, self._active_cam, w, h)
        u, v = float(screen[0]), float(screen[1])
        # camera-space direction (USD camera looks down -Z; image v grows downward)
        d_cam = np.array([(u - p["cx"]) / p["fx"], -(v - p["cy"]) / p["fy"], -1.0])
        # w2c is USD row-vector (point' = point · M); so camera→world is the inverse with
        # the camera origin in row 3 and directions transformed by right-multiply.
        c2w = np.linalg.inv(p["w2c"])
        origin = c2w[3, :3]
        d_world = d_cam @ c2w[:3, :3]
        n = np.linalg.norm(d_world) or 1.0
        return list(origin), list(d_world / n)

    @staticmethod
    def _ray_aabb(o, d, mn, mx):
        tmin, tmax = -1e30, 1e30
        for i in range(3):
            if abs(d[i]) < 1e-9:
                if o[i] < mn[i] or o[i] > mx[i]:
                    return None
                continue
            t1 = (mn[i] - o[i]) / d[i]
            t2 = (mx[i] - o[i]) / d[i]
            t1, t2 = min(t1, t2), max(t1, t2)
            tmin, tmax = max(tmin, t1), min(tmax, t2)
            if tmin > tmax:
                return None
        return tmin if tmin >= 0 else (tmax if tmax >= 0 else None)

    # -- empty space -------------------------------------------------------------
    # The spatial verbs above locate OBJECTS from their bounds; these locate the
    # ABSENCE of objects from actual triangles. Read-only, no @_mutating, no history
    # entry — same class of command as `nearest`/`within`.

    #: Cost ceiling for one query's grid. `nx*ny` cells at `cell` metres; a careless
    #: `--cell 0.001` over a warehouse is a multi-GB allocation inside a Warp kernel,
    #: which is a hang, not an error. Enforced before any geometry is ingested.
    _SPACE_MAX_CELLS = 4_000_000

    #: Bytes the solid heightfield allocates per cell: `heightfield.MAX_SOLID` (256)
    #: layer slots x (2 float32 + 5 int32). Kept in sync by
    #: `test_space.py::test_cell_cost_matches_the_heightfield_allocation`.
    _SPACE_BYTES_PER_CELL = 256 * 28

    #: The real ceiling. Bounding cells alone does not bound memory: 4M cells is
    #: ~16 GB at 4 KiB/cell, which is exactly the hang `_SPACE_MAX_CELLS` exists to
    #: prevent — and a 100x100 m warehouse at `--cell 0.05` reaches it without
    #: anyone doing anything unusual.
    _SPACE_MAX_BYTES = 4 * 1024 ** 3

    def _shared_writer_epoch(self) -> int | None:
        """The mutation epoch of the session that OWNS this reader's root layer.

        `open --read-only` exists precisely to watch a writer's live, unsaved layer —
        USD hands both sessions the same cached `SdfLayer`, which is the whole point of
        "parallel inspection". So a reader's stage changes when the writer edits, while
        its OWN epoch never moves: it runs no mutating command, by definition. Keying a
        cache on the reader's epoch alone therefore pins the geometry as it stood at the
        reader's first query, and a repeated query answers from a stage that no longer
        exists — reporting a deck as clear after the writer has loaded it, for as long
        as the edit leaves the scope bounds alone.

        The owner's epoch is the missing half of the key. `None` when this session is
        not a reader, or holds a layer nobody else owns.
        """
        key = getattr(self, "_reader_key", None)
        if key is None:
            return None
        with _ROOT_LAYER_LOCK:
            entry = _ROOT_LAYER_OWNERS.get(key)
            owner = entry[1]() if entry else None
            if owner is None or getattr(owner, "_layer_key", None) != key:
                return None
            return int(owner._mutation_epoch)

    def _space_scene(self, scope=None, scope_pad=0.0):
        """Ingest (or reuse) the detector-frame triangle soup for this stage.

        Cached on the Session and keyed by the mutation epoch: re-ingesting 37M
        triangles per query is not viable, and the epoch is exactly the signal that
        the geometry changed (it is what `snapshot -D` and the unsaved-edit guard
        already use) — plus the epoch of a writer sharing this layer, see
        `_shared_writer_epoch`. The scope is part of the key because a scoped ingest is
        deliberately partial — reusing it for a wider query would silently under-report
        geometry and over-report free space.
        """
        from usd_core.space.adapter import scene_geometry

        key = (
            self._mutation_epoch,
            self._shared_writer_epoch(),
            None if scope is None else (tuple(map(float, scope[0])), tuple(map(float, scope[1]))),
            float(scope_pad),
        )
        cached = getattr(self, "_space_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        scene = scene_geometry(self._stage, self.refs.ref_for_path,
                               scope=scope, scope_pad=scope_pad)
        self._space_cache = (key, scene)
        return scene

    @staticmethod
    def _container_refs(container) -> list[str]:
        """Normalise `--container` into a flat ref list.

        Comma-separated is the primary spelling, because every other multi-value
        argument in this CLI is (`--size 0.3,0.3,0.3`, `--scope minx,...,maxz`) and a
        container is routinely four or five component prims — repeating the flag once
        per panel is not a shape anyone wants to type or read. The repeated form is
        still accepted, so both of these name the same cabinet:

            --container @l,@r,@back,@base
            --container @l --container @r --container @back --container @base
        """
        if container is None:
            return []
        values = [container] if isinstance(container, str) else list(container)
        out = []
        for value in values:
            out += [part.strip() for part in str(value).split(",") if part.strip()]
        return out

    def _space_window(self, container_paths, scope, frame, obj, interior: bool):
        """Resolve the search window: `--container` bounds ∩ `--scope`.

        One rule covers every combination, because the two inputs answer different
        halves of the same question and compose:

        * neither → the whole stage
        * `--container` only → the union world bbox of its components
        * `--scope` only → exactly the box given
        * both → their intersection ("only the cabinet's lower bay")

        A **derived** window (stage bounds, or `--container`) is grown upward by the
        object's own height. A bounding box's top face is by construction the top of
        the tallest thing inside it, so an ungrown window leaves exactly zero room
        above the surface you most want to load — an open-top container (pallet deck,
        table, flat shelf) would report nothing at all. `--interior` turns that off:
        it says the question is about the space the container *encloses*, not the
        space above it, which is what you want for a bookshelf whose top is reachable
        but which nobody stacks on.

        An explicit `--scope` is never grown: the caller stated the window they meant,
        and silently raising its ceiling would make `--scope` unable to express "do
        not look above this line".
        """
        import numpy as np
        from usd_core.spatial import combined_bbox

        head = 0.0 if interior else float(obj[2])

        def _grow(hi_arr):
            """Raise the ceiling by `head`, plus the float noise at this scale.

            The clearance test is `span > object_height` (with a fixed 1e-6 slack),
            and a derived ceiling is exactly `surface + object_height` — so the space
            above an open-top container sits precisely on the threshold and whether it
            survives is decided by float32 rounding in the bbox. It did survive on a
            pallet deck at z=0.2 and did not on a bookcase top at z=1.6, which is the
            same query giving opposite answers because of the coordinate's magnitude.
            Same relative epsilon `scene_geometry` uses for its scope cull.
            """
            if head <= 0.0:
                return hi_arr
            out = hi_arr.copy()
            out[2] += head + 1e-6 * (abs(float(out[2])) + abs(head) + 1.0)
            return out

        lo = hi = None
        if container_paths:
            rng = combined_bbox(self._stage, container_paths)
            if rng is None:
                raise RuntimeError(
                    "--container names no geometry to measure — pass the container's "
                    "mesh components, not an empty Xform or Scope")
            lo, hi = frame.to_detector_bbox(list(rng.GetMin()), list(rng.GetMax()))
            hi = _grow(hi)
        if scope:
            vals = [float(v) for v in scope] if not isinstance(scope, str) else \
                [float(v) for v in str(scope).replace(" ", ",").split(",") if v]
            if len(vals) != 6:
                raise ValueError(
                    "--scope takes 6 numbers: minx,miny,minz,maxx,maxy,maxz")
            s_lo, s_hi = frame.to_detector_bbox(vals[:3], vals[3:])
            if lo is None:
                lo, hi = s_lo, s_hi
            else:
                lo, hi = np.maximum(lo, s_lo), np.minimum(hi, s_hi)
                if (lo >= hi).any():
                    raise ValueError(
                        "--scope does not overlap --container, so there is nothing "
                        "to search")
        if lo is None:
            rng = self._target_range(None)
            lo, hi = frame.to_detector_bbox(list(rng.GetMin()), list(rng.GetMax()))
            hi = _grow(hi)
        return lo, hi

    def _space_isolate(self, scene, container_paths, exclude_others: bool):
        """Drop everything that is not the container, so capacity is measurable.

        `--exclude-others` is the difference between "what fits around what is in
        there now" and "what this thing could hold" -- the second needs the contents
        gone. It is meaningless without a container to be other *than*, so that
        combination is refused rather than silently ignored.
        """
        if not exclude_others:
            return scene
        if not container_paths:
            raise ValueError(
                "--exclude-others needs --container: it drops everything that is not "
                "the container, so without one there is nothing to keep")
        from usd_core.space.geometry import _under, _under_any, mask_out

        # A parent Xform is an assembly boundary, not evidence of which child meshes
        # are walls/decks and which are removable contents. Keeping every descendant
        # would silently retain a loaded bin's cargo and label remaining space as
        # capacity. There is no sound generic USD convention for that split, so require
        # structural mesh/component paths instead. A selected geometry prim is already
        # unambiguous even when it has authored descendants.
        object_paths = set(scene.object_paths)
        assemblies = [path for path in container_paths if path not in object_paths
                      and any(_under(candidate, path) for candidate in object_paths)]
        if assemblies:
            joined = ", ".join(assemblies)
            raise ValueError(
                "--exclude-others cannot infer structure from container assembly "
                f"{joined}; pass the container's wall/deck mesh components explicitly "
                "so contents can be removed safely")
        drop = [p for p in scene.object_paths if not _under_any(p, container_paths)]
        return mask_out(scene, drop)

    def _space_validate_capacity_container(self, container_paths, exclude_others: bool):
        """Reject an ambiguous capacity assembly before expensive scene ingestion.

        A container root Xform with geometric descendants cannot tell us whether a
        child mesh is structure or cargo.  Validate from USD hierarchy before building
        the triangle soup: capacity errors must be cheap even for a huge assembly.
        ``_space_isolate`` repeats the equivalent check against ingested object paths
        as a defence for its direct callers.
        """
        if not exclude_others or not container_paths:
            return
        from pxr import Usd, UsdGeom

        assemblies = []
        for path in container_paths:
            prim = self._stage.GetPrimAtPath(path)
            if prim and not prim.IsA(UsdGeom.Gprim):
                if any(child.IsA(UsdGeom.Gprim)
                       for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
                    assemblies.append(path)
        if assemblies:
            joined = ", ".join(assemblies)
            raise ValueError(
                "--exclude-others cannot infer structure from container assembly "
                f"{joined}; pass the container's wall/deck mesh components explicitly "
                "so contents can be removed safely")

    @staticmethod
    def _space_size(size, frame):
        """Parse `--size` into a detector-frame (sx, sy, sz) in metres."""
        vals = [float(v) for v in size] if not isinstance(size, str) else \
            [float(v) for v in str(size).replace(" ", ",").split(",") if v]
        if len(vals) == 1:
            vals = vals * 3
        if len(vals) != 3 or any(v <= 0 for v in vals):
            raise ValueError(
                "--size takes 3 positive numbers 'x,y,z' (or one for a cube), in stage units")
        return frame.to_detector_size(vals)

    def _space_guard(self, lo, hi, cell, frame):
        """Reject a grid that would be a hang rather than an answer.

        Two ceilings, because the cell count alone does not bound the allocation: the
        heightfield reserves `_SPACE_BYTES_PER_CELL` for every cell, so the byte limit
        binds first on any grid large enough to matter.

        The resolution and the window are the caller's to choose — nobody is stopped
        from scanning a warehouse at a nanometre — so this only refuses the ones that
        would be a hang, and says what to change. `frame` is here because the number it
        names has to be in the units the caller types: `lo`/`hi`/`cell` are detector
        metres by this point, and a suggestion left in metres reads as 0.000251 on a
        millimetre stage, which is 200x *finer* than the `--cell 0.05` just rejected.
        Following the advice would fail again, with a larger number.
        """
        if cell <= 0:
            raise ValueError("--cell must be positive")
        # Match `build_spans_raster` exactly — a guard that counts cells differently
        # from the allocation it guards is not a guard.
        nx = max(1, int(math.ceil((hi[0] - lo[0]) / cell - 1e-9)))
        ny = max(1, int(math.ceil((hi[1] - lo[1]) / cell - 1e-9)))
        area = (hi[0] - lo[0]) * (hi[1] - lo[1])
        budget = min(self._SPACE_MAX_CELLS,
                     self._SPACE_MAX_BYTES // self._SPACE_BYTES_PER_CELL)
        if nx * ny > budget:
            from usd_core.space.errors import SpaceOutOfRange
            suggested = ((area / budget) ** 0.5 if budget else 0.0) / frame.meters_per_unit
            gib = nx * ny * self._SPACE_BYTES_PER_CELL / 1024 ** 3
            raise SpaceOutOfRange(
                f"grid would be {nx}x{ny} = {nx * ny:,} cells (~{gib:.1f} GiB of "
                f"heightfield; limit {budget:,} cells). "
                f"Raise --cell to about {suggested:.3g} or narrow the scope with --container/--scope")
        return nx, ny

    @staticmethod
    def _space_volume(size_m, frame) -> float:
        """A detector-frame extent (metres) → a volume in stage units.

        Every other reported field crosses back through a `Frame.to_stage_*`, so a raw
        m^3 volume next to a stage-unit `size` reads as a contradiction: on the shipped
        millimetre PCB a region would report `size: [5, 5, 3]` beside `volume: 7.5e-08`.
        Ranking is unaffected (the scale factor is positive and monotone) — this is
        purely about the numbers meaning what the rest of the record means.
        """
        mpu = frame.meters_per_unit
        return float(size_m[0] * size_m[1] * size_m[2]) / (mpu * mpu * mpu)

    @staticmethod
    def _space_area(area_m2, frame) -> float:
        """A detector-frame area (m^2) → stage units. See `_space_volume`."""
        mpu = frame.meters_per_unit
        return float(area_m2) / (mpu * mpu)

    def _mint_space_refs(self, kind: str, regions: list[dict]) -> list[dict]:
        """Assign `@s1…@sN` to ranked regions, best first.

        Renumbers from `@s1` on every query: these are result-set handles, so `@s1`
        must always mean "the best candidate from the last space query".

        The handle names a region; it does not carry hidden geometry. Detection is a
        read-only measurement and the response IS the answer — `aabb` (the key), a
        `floor` + `holes` (the support surface), and for `free` a `ceiling` surface.
        Anything that consumes a region reads those, so there is nothing to keep back. `_rank`
        is dropped because it only ever ordered this list.
        """
        self.refs.reset_namespace("s")
        out = []
        for i, region in enumerate(regions, start=1):
            ref = self.refs.assign("s", f"/__space__/{kind}/{i}", kind, f"{kind}{i}")
            out.append({k: v for k, v in {**region, "ref": ref, "kind": kind}.items()
                        if k != "_rank"})
        return out

    @staticmethod
    def _space_overflow_issue(result) -> list[Issue]:
        """Warn when rasterised or synthesized per-column span buffers overflow.

        The rasteriser keeps at most `heightfield.MAX_SOLID` solid layers per column.
        A column pierced by more is treated as **fully solid**, which is fail-safe —
        the answer can lose a resting surface but can never invent one — and until now
        that was the end of it: a query over a scene with thousands of dropped layers
        came back `ok: true` with nothing to distinguish it from a clean one.

        Deliberately an issue and not a summary field: `issues` is already the
        variable-length channel for exactly this ("a non-fatal observation surfaced to
        the agent"), while the summary is a fixed shape describing the query. A
        structured `solid_overflow_cells` / `degraded` field belongs with the rest of
        the reporting surface in the follow-up.

        The advice deliberately does not mention `--cell`. Overflow counts geometry
        layers stacked at one XY location, not triangles per unit area, so the cell
        size does not change it: on a 300-plate column the deepest cell holds 2404
        layers at every `--cell` from 0.2 down to 0.005 m — only the number of affected
        cells moves (3 to 1681).
        """
        solid = getattr(result, "solid_overflow_cells", 0)
        free = getattr(result, "free_overflow_cells", 0)
        gapfill = getattr(result, "gapfill_overflow_cells", 0)
        if not solid and not free and not gapfill:
            return []
        # Lazy: `heightfield` pulls in warp, and this module must stay importable
        # without the optional `[space]` extra.
        from usd_core.space.heightfield import MAX_FREE, MAX_SOLID

        total = int(result.nx) * int(result.ny)
        issues = []
        if solid:
            issues.append(Issue(
                "warn",
                f"{solid} of {total} grid cells hold more solid layers than the "
                f"rasteriser keeps ({int(MAX_SOLID)}); each was treated as fully solid",
                "Conservative: no free space is invented, but a real resting surface in "
                "those cells is missing from this answer. They are normally dense "
                "structure cores (uprights, brace intersections) that are solid anyway. "
                "`--cell` does not affect this — the limit is on layers stacked at one "
                "spot. Use --scope or --container to measure away from that structure."))
        if free:
            # The other direction, and the dangerous one: this drops answers rather
            # than hiding surfaces behind extra solid, and the levels it drops are the
            # topmost ones — a rack with more shelves than the buffer holds simply
            # stops reporting its upper bays, with nothing else in the response saying so.
            issues.append(Issue(
                "warn",
                f"{free} of {total} grid columns have more usable levels than the "
                f"rasteriser keeps ({int(MAX_FREE)}); the highest ones were dropped",
                "NOT conservative: whole regions are missing from this answer, not just "
                "detail. It takes a column with many stacked surfaces — a tall rack, a "
                "deep shelving unit — so narrow the question with --scope or --container "
                "to a few bays at a time, which also puts the levels back."))
        if gapfill:
            issues.append(Issue(
                "warn",
                f"{gapfill} of {total} empty grid columns matched more supported levels "
                f"than the detector keeps ({int(MAX_FREE)}); the highest ones were dropped",
                "Whole upper deck regions can be missing from this answer. Narrow the "
                "query with --scope or --container to the affected rack bays."))
        return issues

    #: Regions listed in the human-readable text. The payload is never truncated, so
    #: this is a reading limit and nothing else: a warehouse scope yields hundreds of
    #: regions and dumping them all is the anti-pattern the README calls out.
    _SPACE_TEXT_LINES = 10

    def _space_response(self, command: str, summary: dict, regions: list[dict],
                        issues: list[Issue] | None = None) -> Response:
        """Every region in `data`; the first few in the text.

        There is no knob. `--top` used to be one and it could only mislead: the text and
        the payload disagreed about what it meant, `--top 0` was documented as "all" and
        silently returned the default ten (the CLI turned 0 into None and `drop_none`
        then dropped the key, so the session default won), while the same 0 sent
        straight to the daemon returned *nothing*. A machine consumer reads `data` and
        gets everything; a human reads the text and gets the best ten with `count`
        stating the true total, so a listing never reads as exhaustive.
        """
        def _row(r):
            lo, hi = r["aabb"]["min"], r["aabb"]["max"]
            span = [round(hi[i] - lo[i], 3) for i in range(len(lo))]
            holes = len(r.get("holes") or [])
            bits = [f"{r['ref']}", f"{[round(v, 3) for v in lo]}..{[round(v, 3) for v in hi]}",
                    f"span={span}"]
            # The hole count is the one thing the footprint cannot say, and it is the
            # difference between "a 3.6 x 3.6 deck" and "a deck around a pillar".
            if holes:
                bits.append(f"holes={holes}")
            return "  ".join(bits)

        listed = regions[:self._SPACE_TEXT_LINES]
        summary = {**summary, "count": len(regions), "listed": len(listed)}
        if listed:
            head = f"{len(regions)} region(s); best {_row(listed[0])}"
        else:
            head = "no region fits — try a smaller --size, a coarser --cell, or a wider scope"
        lines = [head] + [f"  {_row(r)}" for r in listed]
        if len(regions) > len(listed):
            lines.append(f"  … {len(regions) - len(listed)} more (all of them are in --json)")
        return Response(command=command, summary=summary,
                        data={"regions": regions, "refs": [r["ref"] for r in regions],
                              "text": "\n".join(lines)},
                        issues=issues or [])

    def _space_common(self, size, container, scope, cell, interior, exclude_others,
                      device, frame):
        """Everything `free` and `support` do identically, in one place.

        Both verbs are the same pipeline over the same unit -- a surface and the
        clearance above it -- so window, grid, ingest and isolation are resolved once
        here and only the extraction differs.
        """
        from usd_core.space.device import resolved_device

        # Resolve auto -> cpu/cuda ONCE, before any work: the resolved value goes into
        # the summary, so which device produced a result is never a guess.
        device = resolved_device(device)
        obj = self._space_size(size, frame)
        paths = [self._path_of(r) for r in self._container_refs(container)]
        self._space_validate_capacity_container(paths, exclude_others)
        lo, hi = self._space_window(paths, scope, frame, obj, interior)
        cell_m = self._space_cell(cell, obj, frame)
        self._space_guard(lo, hi, cell_m, frame)
        scene = self._space_scene(scope=(lo, hi), scope_pad=cell_m)
        scene = self._space_isolate(scene, paths, exclude_others)
        return device, obj, paths, lo, hi, cell_m, scene

    def _space_summary(self, size, container, scope, exclude_others, interior,
                       cell_m, frame, device, margin=0.0, merge_gap=None) -> dict:
        summary = {
            "size": [round(v, 4) for v in self._as_list(size)],
            "container": self._container_refs(container) or None,
            "scope": "explicit" if scope else ("container" if container else "stage"),
            "contents": "excluded" if exclude_others else "included",
            "extent": "interior" if interior else "including top",
            "cell": round(cell_m / frame.meters_per_unit, 6),
            "device": device,
        }
        # Echoed only when set. A margin is the one argument that can empty the answer
        # without appearing anywhere in it: "no region fits" reads like the scene has
        # no room, when what happened is that the inset ate it.
        if float(margin):
            summary["margin"] = float(margin)
        # Same rule as `margin`: echoed only when the caller set it, because an
        # explicit merge gap changes which surfaces exist at all and "count: 0" would
        # otherwise carry no trace of the argument that produced it.
        if merge_gap is not None:
            summary["merge_gap"] = float(merge_gap)
        return summary

    @staticmethod
    def _corner_height(gx: int, gy: int, floors, base: float) -> float:
        """The surface height at one grid corner: the highest of the (up to four)
        region cells meeting there, so the polygon sits ON the surface rather than
        sunk into it. `base` covers a corner no region cell touches."""
        near = [floors[c] for c in ((gx - 1, gy - 1), (gx, gy - 1),
                                    (gx - 1, gy), (gx, gy)) if c in floors]
        return max(near) if near else base

    @staticmethod
    def _loop_on_surface(loop, floors, base: float, cs: float, ox: float, oy: float):
        """A traced loop re-sampled onto the surface: `(points, heights)`.

        The loop arrives collinear-simplified, which is right in XY — a straight run of
        boundary is two vertices — but it is exactly wrong in z once a region may be
        sloped: the ramp's whole descent lives on one such run, and simplifying it away
        leaves a polygon pinned to its endpoints, hovering over the middle. Recast has
        the same problem and answers it with a separate detail mesh; here the loop is
        walked cell by cell and a vertex is kept wherever the surface height actually
        moves. A flat deck is untouched (every intermediate height matches, so nothing
        is re-inserted); a slope gets one vertex per cell of rise, and no more.
        """
        if not floors:
            return [list(p) for p in loop], base
        pts, zs = [], []
        n = len(loop)
        for i in range(n):
            x0, y0 = float(loop[i][0]), float(loop[i][1])
            x1, y1 = float(loop[(i + 1) % n][0]), float(loop[(i + 1) % n][1])
            g0 = (int(round((x0 - ox) / cs)), int(round((y0 - oy) / cs)))
            g1 = (int(round((x1 - ox) / cs)), int(round((y1 - oy) / cs)))
            steps = max(abs(g1[0] - g0[0]), abs(g1[1] - g0[1])) or 1
            dx = (g1[0] - g0[0]) // steps
            dy = (g1[1] - g0[1]) // steps
            prev = None
            for k in range(steps):  # [start, end) — the next edge contributes the end
                gx, gy = g0[0] + dx * k, g0[1] + dy * k
                z = Session._corner_height(gx, gy, floors, base)
                # Keep the first vertex of the edge always; after that only where the
                # surface moves, which is what makes a flat run stay two points.
                if k == 0 or (prev is not None and abs(z - prev) > 1.0e-9):
                    pts.append([ox + gx * cs, oy + gy * cs])
                    zs.append(z)
                prev = z
        return pts, zs

    def _support_regions(self, result, frame, cap=None,
                         report_height: bool = False) -> list[dict]:
        """`SupportRegionCells` → region records, for both queries.

        `free` and `support` are the same measurement — a surface and the clearance
        above it — so they share one detector and one conversion. `support` reports
        the floor surface; `free` reports that floor plus an explicit ceiling surface.

        The cap is applied here, on the way out, so the clearance is measured before
        it is truncated. Capping inside the detector would lower `ceiling_z` itself and
        the reported height would collapse to whatever was asked for — losing the
        distinction between "exactly one fits" and "this was capped at one". (The
        detector used to take a `max_stack_height` for this and nothing ever passed
        it; the parameter is gone.)
        """
        from usd_core.space.extract import prisms_from_region_cells

        cs, (ox, oy) = result.cell_size, result.origin
        out = []
        for reg in result.regions:
            if not reg.cells:
                continue
            cells = sorted(reg.cells)
            # The AABB is the region's EXTENT — where this space is and how far it
            # reaches. A shelf deck with two pillars through it is still that deck:
            # reporting the largest pillar-free rectangle instead would call a
            # 2.0 x 1.0 m level "1.85 x 0.85", which is not what the space is.
            # The holes in the floor are where a caller sees that this extent is
            # not solid; nothing else needs to restate it.
            xs = [c[0] for c in cells]
            ys = [c[1] for c in cells]
            ax0, ay0 = ox + min(xs) * cs, oy + min(ys) * cs
            ax1, ay1 = ox + (max(xs) + 1) * cs, oy + (max(ys) + 1) * cs
            clearance = float(reg.ceiling_z - reg.support_z)
            height = clearance if cap is None else min(clearance, float(cap))
            base = float(reg.support_z)
            floors = reg.floors or {}
            # `base` is the detector's conservative rest height (its highest floor) and
            # `floor_lo` its lowest. They differ exactly when the surface is not flat:
            # cells join when the step between NEIGHBOURS is under the climb tolerance,
            # so a continuous ramp is deliberately one region and its floors span the
            # whole rise. Reporting only `base` there would put the polygon at the top
            # of the slope with the surface metres below it.
            floor_lo = min(floors.values()) if floors else base
            prisms = prisms_from_region_cells(
                [(set(cells), base, float(reg.ceiling_z))], cs, (ox, oy))
            prism = prisms[0] if prisms else None
            lo, hi = frame.to_stage_bbox((ax0, ay0, floor_lo), (ax1, ay1, base + height))
            up = 1 if frame.up_axis == "Y" else 2
            # The AABB is the KEY — an address for a shape, not a description of it.
            # It encloses the real 3D floor and, for `free`, its ceiling. A flat
            # support surface is degenerate on the up axis; a continuous slope spans
            # its true low and high floor points.
            amin, amax = list(lo), list(hi)
            if not report_height:
                # A support AABB encloses its surface: degenerate for a flat deck, and
                # spanning the rise when the floors vary.
                amax[up] = max(amin[up], float(frame.to_stage_point((ax0, ay0, base))[up]))
            floor_loop, floor_zs = (
                self._loop_on_surface(prism.outline, floors, base, cs, ox, oy)
                if prism else ([], []))
            floor = frame.to_stage_xy_polygon(floor_loop, floor_zs) if prism else []
            record = {
                "aabb": {"min": amin, "max": amax},
                # The shape is the value: every point in `floor` (holes excluded) is
                # a valid rest point for the requested footprint.
                # Every point inside it (holes excluded) is a valid resting spot for
                # an object of `--size`, because the cells it was traced from are the
                # eroded valid-CENTRE cells, not the raw deck. `kind` says how to read
                # it: `support` returns only this surface; `free` adds its ceiling.
                "floor": floor,
                "holes": ([frame.to_stage_xy_polygon(
                    *self._loop_on_surface(h, floors, base, cs, ox, oy))
                    for h in (prism.holes or [])] if prism else []),
                # Ranking key, dropped by `_mint_space_refs` before the response.
                # `free` asks "how much room is left", so it ranks by usable volume;
                # `support` asks "where does one go", so the height is fixed and only
                # the area distinguishes candidates.
                "_rank": len(cells) * cs * cs * (height if report_height else 1.0),
            }
            if report_height:
                # Without a cap, the conservative lowest ceiling is horizontal. A
                # stack cap is a slab above each local floor point, so its ceiling
                # follows a slope instead of spuriously adding the slope's rise.
                ceiling_zs = ([float(reg.ceiling_z)] * len(floor_zs) if cap is None
                              else [z + height for z in floor_zs])
                record["ceiling"] = (frame.to_stage_xy_polygon(floor_loop, ceiling_zs)
                                     if prism else [])
            out.append(record)
        return out

    def space_free(self, size, container=None, exclude_others: bool = False,
                   interior: bool = False, scope=None, cell: float | None = None,
                   stack_height: float | None = None, tau: float = 1.0,
                   margin: float = 0.0, smooth: bool = True,
                   merge_gap: float | None = None,
                   device: str = "auto", **_ignored) -> Response:
        """How much room is left, ranked by volume.

        The same measurement as `support`, reported to the top of the clearance
        instead of to the top of one object. With `--container --exclude-others` it
        is a container's capacity; there is no separate verb for that.
        """
        from usd_core.space.adapter import Frame
        self._require_stage(); self._index_prims()
        try:
            from usd_core.space.support import detect_support_regions

            frame = Frame.of(self._stage)
            device, obj, paths, lo, hi, cell_m, scene = self._space_common(
                size, container, scope, cell, interior, exclude_others, device, frame)
            cap = None if stack_height is None else float(stack_height) * frame.meters_per_unit
            if cap is not None and cap <= 0:
                raise ValueError("--stack-height must be positive")
            result = detect_support_regions(
                scene, lo, hi, cell_size=cell_m, object_size=obj,
                tau=float(tau), support_margin=self._space_margin(margin, frame),
                merge_gap=self._space_merge_gap(merge_gap, frame),
                smooth=bool(smooth), device=device)
            regions = self._support_regions(result, frame, cap=cap,
                                            report_height=True)
            regions.sort(key=lambda r: r["_rank"], reverse=True)
            regions = self._mint_space_refs("free", regions)
        except Exception as exc:  # noqa: BLE001 — bad size/scope/cell and a missing
            # `[space]` extra all become the ordinary ok:false envelope
            return self._err("space.free", exc)
        summary = self._space_summary(size, container, scope, exclude_others, interior,
                                      cell_m, frame, device, margin, merge_gap)
        summary["stack"] = "to ceiling" if stack_height is None else stack_height
        summary["tau"] = tau
        return self._space_response(
            "space.free", summary, regions,
            self._space_overflow_issue(result)
            + self._space_margin_issue(margin, cell_m, frame)
            + self._space_merge_gap_issue(merge_gap, obj, frame))

    def space_support(self, size, container=None, exclude_others: bool = False,
                      interior: bool = False, scope=None, cell: float | None = None,
                      tau: float = 1.0, margin: float = 0.0, smooth: bool = True,
                      merge_gap: float | None = None,
                      device: str = "auto", **_ignored) -> Response:
        """Where one object of `size` can rest stably, ranked by support area.

        Same detector as `free`; this one reports a single object's slab and ranks by
        support area, because the question is "where does this go" rather than "how
        much is left".
        """
        from usd_core.space.adapter import Frame
        self._require_stage(); self._index_prims()
        try:
            from usd_core.space.support import detect_support_regions

            frame = Frame.of(self._stage)
            device, obj, paths, lo, hi, cell_m, scene = self._space_common(
                size, container, scope, cell, interior, exclude_others, device, frame)
            # smooth: sliding-window gap bridging. A slatted deck (pallet, grating)
            # has columns whose floor fell through a gap to the ground far below;
            # bridging harmonises them to the surrounding slat level over the object's
            # footprint, so a box wider than a gap rests across the slats. Off, a real
            # pallet reports no usable deck at all.
            result = detect_support_regions(
                scene, lo, hi, cell_size=cell_m, object_size=obj,
                tau=float(tau), support_margin=self._space_margin(margin, frame),
                merge_gap=self._space_merge_gap(merge_gap, frame),
                smooth=bool(smooth), device=device)
            regions = self._support_regions(result, frame)
            regions.sort(key=lambda r: r["_rank"], reverse=True)
            regions = self._mint_space_refs("support", regions)
        except Exception as exc:  # noqa: BLE001
            return self._err("space.support", exc)
        summary = self._space_summary(size, container, scope, exclude_others, interior,
                                      cell_m, frame, device, margin, merge_gap)
        summary["tau"] = tau
        return self._space_response(
            "space.support", summary, regions,
            self._space_overflow_issue(result)
            + self._space_margin_issue(margin, cell_m, frame)
            + self._space_merge_gap_issue(merge_gap, obj, frame))

    @staticmethod
    def _as_list(size) -> list[float]:
        vals = [float(v) for v in size] if not isinstance(size, str) else \
            [float(v) for v in str(size).replace(" ", ",").split(",") if v]
        return vals * 3 if len(vals) == 1 else vals

    @staticmethod
    def _space_margin(margin, frame) -> float:
        """`--margin` in stage units → detector metres.

        Every length these verbs take is in stage units and is converted: `--size`
        through `to_detector_size`, `--cell` and `--stack-height` by `meters_per_unit`,
        `--scope` through `to_detector_bbox`. This one was passed through raw, so a
        2 mm inset on a millimetre stage arrived as a 2 **metre** inset, eroded every
        region to nothing, and the query answered `count: 0` — advising a smaller
        `--size` or a coarser `--cell`, neither of which is the problem.
        """
        return float(margin) * frame.meters_per_unit

    @staticmethod
    def _space_merge_gap(merge_gap, frame) -> float | None:
        """`--merge-gap` in stage units → detector metres; `None` keeps the default.

        Stage units for the same reason as `--cell`, `--margin` and `--size`: every
        length these verbs accept is typed in the units the stage is authored in. On
        the millimetre PCB the detector default of 1e-3 m is `--merge-gap 1.0`.
        """
        if merge_gap is None:
            return None
        return float(merge_gap) * frame.meters_per_unit

    @staticmethod
    def _space_merge_gap_issue(merge_gap, obj, frame) -> list:
        """A `warn` when the merge gap is at least as tall as the object.

        The rasteriser fuses two solid layers with less air than the gap between them,
        and the free-span filter drops anything shorter than the object — so a fused
        gap is only ever *reachable* by an object shorter than the gap itself. In that
        regime the merge deletes real resting surfaces: two plates 0.9 mm apart report
        one surface at the 1 mm default and two at `--merge-gap 0`.

        The default is silent here on purpose. Nothing warns when a 5 mm object meets
        the 1 mm default, because that combination provably cannot lose anything; the
        warning fires exactly when the object could have fitted in a gap that was
        merged away.
        """
        # Lazy for the same reason as the overflow issue above: `heightfield` pulls in
        # warp, and this module stays importable without the optional `[space]` extra.
        from usd_core.space.heightfield import DEFAULT_MERGE_GAP

        gap = Session._space_merge_gap(merge_gap, frame)
        gap = DEFAULT_MERGE_GAP if gap is None else gap
        if gap < float(obj[2]):
            return []
        stage_gap = gap / frame.meters_per_unit
        return [Issue(
            "warn",
            f"--merge-gap {stage_gap:g} is at least as tall as the object "
            f"({float(obj[2]) / frame.meters_per_unit:g} in stage units), so gaps the "
            f"object would have fitted in were fused into solid",
            f"Resting surfaces are missing from this answer. Pass --merge-gap below "
            f"the object height (it only has to absorb float noise, so a value orders "
            f"of magnitude smaller is fine) and re-run.")]

    @staticmethod
    def _space_margin_issue(margin, cell_m, frame) -> list:
        """Warn when a requested support margin rounds to zero detector cells.

        The support gate is morphological and operates on whole grid cells.  Silently
        accepting a positive value that quantizes to zero is worse than no margin: the
        response echoes the caller's value while returning perch candidates as though
        the anti-perch gate had run.
        """
        requested = float(margin)
        if requested <= 0.0:
            return []
        resolved = Session._space_margin(requested, frame)
        if int(round(resolved / float(cell_m))) > 0:
            return []
        half_cell = 0.5 * float(cell_m) / frame.meters_per_unit
        return [Issue(
            "warn",
            f"--margin {requested:g} does not reach one grid cell at --cell "
            f"{float(cell_m) / frame.meters_per_unit:g}, so it rounded to zero and "
            f"did not apply a support-margin gate",
            f"Use --margin greater than {half_cell:g}, choose a smaller --cell, or "
            f"pass --margin 0 to request no anti-perch margin explicitly.")]

    @staticmethod
    def _space_cell(cell, obj_size, frame) -> float:
        """Grid resolution in detector metres; defaults to a fraction of the object.

        The grid must resolve the object's own footprint or every candidate is either
        missed or over-reported, so the default scales with `--size` rather than being
        a fixed number that is wrong at both warehouse and tabletop scale.
        """
        if cell is not None:
            resolved = float(cell) * frame.meters_per_unit
            if resolved <= 0:
                raise ValueError("--cell must be positive")
            return resolved
        return max(float(min(obj_size[0], obj_size[1])) / 4.0, 1e-4)

    # -- physics ------------------------------------------------------------------
    def physics_inspect(self, root=None, **_ignored) -> Response:
        from usd_core import query
        self._require_stage(); self._index_prims()
        try:
            cands = query.mesh_candidates(self._stage, root=self._path_of(root) if root else None)
        except Exception as exc:  # noqa: BLE001
            return self._err("physics.inspect", exc)
        for c in cands:
            c["ref"] = self.refs.ref_for_path(c["path"])
            if c.get("instance_root"):
                # the editable authoring target for proxy candidates — physics apply
                # rejects proxies (USD forbids authoring inside an instance)
                c["instance_root_ref"] = self.refs.ref_for_path(c["instance_root"])
        return Response(command="physics.inspect", summary={"candidates": len(cands)},
                        data={"candidates": cands})

    def physics_topology(self, root=None, **_ignored) -> Response:
        """Return authored physics topology for the open stage."""
        from usd_core.physics_topology import inspect_topology

        self._require_stage()
        source = self._stage.GetRootLayer().realPath
        if not source:
            return self._err("physics.topology", ValueError("stage has no file path"))
        try:
            report = inspect_topology(
                source,
                root_prim_path=self._path_of(root) if root else None,
                path_space="source",
                stage=self._stage,
            )
        except Exception as exc:  # noqa: BLE001
            return self._err("physics.topology", exc)
        return Response(
            command="physics.topology",
            summary={
                "rigid_bodies": report.get("enabled_rigid_body_count", 0),
                "colliders": report.get("enabled_collider_count", 0),
                "joints": len(report.get("joints", [])),
            },
            data=report,
        )

    def _resolved_physics_path(self, ref_or_path: str) -> str:
        """Resolve one physics-patch path field (@ref or literal) and prove the target
        prim exists and is editable BEFORE any authoring touches the stage."""
        path = self._edit_path_of(ref_or_path)
        prim = self._stage.GetPrimAtPath(path)
        if not prim.IsValid() or not prim.IsActive():
            raise ValueError(f"physics decision references a missing prim: "
                             f"{ref_or_path} -> {path}")
        return path

    @_mutating
    def physics_apply(self, operations=None, **_ignored) -> Response:
        from usd_core import physics
        self._require_stage()
        try:
            if not isinstance(operations, dict):
                raise ValueError(
                    "physics.apply needs an explicit operations object; workflow decisions "
                    "must be translated before invoking usd-cli"
                )
            staged = dict(operations)
            for key in ("rigid_bodies", "colliders"):
                if key in staged:
                    staged[key] = [
                        {**row, "path": self._resolved_physics_path(row["path"])}
                        for row in staged[key]
                    ]
            if "bindings" in staged:
                staged["bindings"] = [
                    {
                        **row,
                        "target_path": self._resolved_physics_path(row["target_path"]),
                    }
                    for row in staged["bindings"]
                ]
            # Apply the whole patch transactionally: snapshot the root layer and roll
            # back on ANY failure, so a mid-apply error can never leave the stage
            # partially mutated (same rollback mechanism undo/redo use).
            root = self._stage.GetRootLayer()
            before = root.ExportToString()
            try:
                resolved = physics.apply_operations(self._stage, staged)
            except Exception as apply_exc:
                root.ImportFromString(before)
                self._index_prims()
                raise RuntimeError(
                    f"physics.apply failed and the stage was rolled back: {apply_exc}"
                ) from apply_exc
        except Exception as exc:  # noqa: BLE001
            return self._err("physics.apply", exc)
        self._index_prims()
        line = self._physics_authored_line(resolved)
        data = {"authored": resolved}
        if line:
            data["text"] = line
        return Response(command="physics.apply", summary={"operations": sum(
                        len(resolved.get(key) or [])
                        for key in ("scene", "rigid_body", "collision", "material", "binding")
                    )},
                        data=data)

    @staticmethod
    def _physics_authored_line(rec: dict) -> str | None:
        """One-line digest of a physics.apply authored record, e.g.
        `authored: /Robot (PhysicsRigidBodyAPI+PhysicsMassAPI), 33 colliders updated`."""
        apis = rec.get("authored_apis") or {}
        bits = []
        for body in rec.get("rigid_body") or []:
            names = "+".join(apis.get(body, []))
            bits.append(f"{body} ({names})" if names else body)
        n_coll = len(rec.get("collision") or [])
        if n_coll:
            bits.append(f"{n_coll} collider{'s' if n_coll != 1 else ''} updated")
        if rec.get("material"):
            bits.append(f"{len(rec['material'])} physics material(s) updated")
        return ("authored: " + ", ".join(bits)) if bits else None

    def physics_validate(self, **_ignored) -> Response:
        from usd_core import physics
        self._require_stage()
        try:
            report = physics.validate_schema(self._stage)
        except Exception as exc:  # noqa: BLE001
            return self._err("physics.validate", exc)
        return Response(command="physics.validate", ok=report["ok"],
                        summary={"ok": report["ok"], **report["checks"]}, data=report)

    def physics_simulate(self, scene: str, body: str, rest_position: list[float],
                         world_up: list[float], engine: str = "ovphysx", duration: float = 1.0,
                         dt: float = 1.0 / 240.0, fps: int = 30, output=None,
                         body_pattern: str | None = None,
                         relax_ovphysx_address_space_limit: bool = False,
                         **_ignored) -> Response:
        """Simulate an explicit pre-authored scene and record trajectory facts. Uses the
        real ovphysx solver — locally on Linux or Windows, otherwise on a
        configured managed OVRTX adapter with the physics capability; errors
        when neither is available. Play the recording with
        `render-frames --scene <recording.usda>` for
        behavior review."""
        from usd_core import physics_runtime
        self._require_stage()
        try:
            out_arg = Path(output) if output else self._session_dir("physics")
            # `-o recording.usda` means "write the recording THERE", not "make a
            # directory named recording.usda" — physics_runtime owns that suffix
            # handling; here we only need the containing dir for staging, and we
            # must not mkdir the file path (that pre-empted the fix with a dir).
            out_dir = out_arg
            if out_dir.suffix.lower() in {".usd", ".usda", ".usdc"}:
                out_dir = out_dir.parent if str(out_dir.parent) not in ("", ".") else Path(".")
            out_dir.mkdir(parents=True, exist_ok=True)
            remote = physics_runtime.resolve_remote_physics_backend(
                self.config.render,
                verify=not physics_runtime.ovphysx_platform_supported(),
            )
            body_path = self._path_of(body)
            report = physics_runtime.simulate_scene(
                scene, str(out_arg), body_path=body_path,
                body_pattern=body_pattern or body_path,
                rest_position=rest_position, world_up=world_up, engine=engine,
                duration_s=float(duration), dt=float(dt), sample_fps=int(fps), remote=remote,
                relax_address_space_limit=relax_ovphysx_address_space_limit)
        except Exception as exc:  # noqa: BLE001
            return self._err("physics.simulate", exc)
        m = report.get("metrics", {})
        arts = []
        for key, label in (("recording_usda", "recording"), ("scene_usd", "scene"),
                           ("trajectory_jsonl", "trajectory"), ("report_path", "report")):
            if report.get(key):
                arts.append(Artifact(path=report[key], kind="physics", label=label))
        facts = report.get("simulation_facts", {})
        lines = [f"engine={report['engine']} executor={report.get('executor', 'local')} "
                 f"bodies={report['n_bodies']} "
                 f"samples={facts.get('trajectory_sample_count')} "
                 f"finite={facts.get('trajectory_finite')}"]
        if m:
            lines.append(f"  settle_distance={m.get('settle_distance')} "
                         f"final_speed={round(m.get('final_linear_speed', 0), 3)} "
                         f"max_speed={round(m.get('max_linear_speed', 0), 3)}")
        if report.get("recording_usda"):
            lines.append(f"  play: render-frames --scene {report['recording_usda']}")
        return Response(command="physics.simulate",
                        summary={"engine": engine, "executor": report.get("executor", "local"),
                                 "settle_distance": m.get("settle_distance")},
                        data={**report, "text": "\n".join(lines)}, artifacts=arts)

    # -- conversion (convert-to-usd: shell out to external converters) -----------
    def convert(self, source: str, output=None, output_format=None, **_ignored) -> Response:
        """Route a source asset to USD via an external converter (urdf/mujoco/cad).

        USD inputs pass through. Conversion itself lives in third-party packages; this
        probes the format and shells out when a converter is available.
        """
        import shutil
        import subprocess
        self_path = Path(source)
        suffix = self_path.suffix.lower()
        usd_suffixes = {".usd", ".usda", ".usdc", ".usdz"}
        if suffix in usd_suffixes:
            out = str(output) if output else (
                str(self_path.with_suffix("." + str(output_format).lstrip(".")))
                if output_format else None)
            # realpath, not abspath: a symlink alias of the source is the same file
            if out is None or os.path.realpath(out) == os.path.realpath(source):
                return Response(command="convert",
                                summary={"route": "passthrough", "output": source},
                                data={"output_usd_path": source, "status": "passed"})
            # An explicit output path/format was requested — honor it. This used to
            # print 'route: passthrough' and silently never write the file.
            out_suffix = Path(out).suffix.lower()
            if out_suffix not in usd_suffixes:
                return Response(
                    command="convert", ok=False,
                    summary={"route": "usd-export", "status": "blocked"},
                    issues=[Issue("error",
                                  f"cannot convert USD to '{out_suffix or out}' here — "
                                  "supported outputs are .usd/.usda/.usdc/.usdz "
                                  "(use `export` / an external converter for other "
                                  "formats)")])
            # USD→USD goes through the SAME verified-atomic publish pipeline as
            # save/export (per-destination lock, refuse another session's open
            # file, temp work file + read-back verify + os.replace + dir fsync) —
            # the old bare `layer.Export(out)` could clobber another session's live
            # destination mid-write, and `_rebase_asset_paths(out, …)` after the
            # fact could FindOrOpen that session's live layer and Save() over it.
            try:
                info = self._convert_usd_to_usd(source, out, out_suffix)
            except Exception as exc:  # noqa: BLE001
                return self._err("convert", exc)
            return Response(command="convert",
                            summary={"route": "usd-export", "status": "passed",
                                     "output": out, **info},
                            data={"output_usd_path": out, "status": "passed", **info})
        out = output or str(self_path.with_suffix("." + (output_format or "usd")))
        if suffix == ".urdf":
            tool, args = "urdf-usd-converter", [source, out]
        elif suffix in (".mjcf", ".xml"):
            tool, args = "mujoco-usd-converter", [source, out]
        else:  # CAD/mesh family
            tool, args = "usd-convert", [source, out]
        if not shutil.which(tool):
            return Response(command="convert", ok=False,
                            issues=[Issue("error", f"converter '{tool}' not installed for "
                                          f"'{suffix}'. Install it (e.g. `uv pip install "
                                          f"{tool}`) — conversion is an external dependency.")],
                            summary={"route": tool, "status": "blocked"})
        try:
            proc = subprocess.run([tool, *args], capture_output=True, text=True, timeout=1800)
        except Exception as exc:  # noqa: BLE001
            return self._err("convert", exc)
        ok = proc.returncode == 0 and Path(out).exists()
        return Response(command="convert", ok=ok,
                        summary={"route": tool, "status": "passed" if ok else "failed",
                                 "output": out},
                        data={"output_usd_path": out if ok else None,
                              "stdout": proc.stdout[-2000:], "stderr": proc.stderr[-2000:]},
                        issues=[] if ok else [Issue("error", f"{tool} failed ({proc.returncode})")])

    # -- history (PR-7) ----------------------------------------------------------
    @_mutating
    def undo(self, n: int = 1) -> Response:  # PR-7.1
        self._require_stage()
        ops = self.history.peek_undo(n)
        if not ops:
            return Response(command="undo", data={"undone": 0, "note": "nothing to undo"})
        before = self._stage_layer_snapshot()
        try:
            for op in ops:
                self._apply_compound(op, "undo")
        except Exception as exc:  # noqa: BLE001 — a failing inverse shouldn't crash the daemon
            try:
                self._restore_stage_layer_snapshot(before)
            except Exception as rollback_exc:  # noqa: BLE001
                self._index_prims()
                return self._err("undo", RuntimeError(
                    f"undo failed: {exc}; rollback also failed: {rollback_exc}"))
            return self._err("undo", RuntimeError(f"undo failed and was rolled back: {exc}"))
        self.history.commit_undo(len(ops))
        self._index_prims()
        skipped = self._non_undoable_in(ops)
        summary = {"undone": len(ops)}
        if skipped:
            summary["not_undoable"] = len(skipped)
        return Response(command="undo", summary=summary,
                        data={"undone": len(ops), "ops": [o.label or o.command for o in ops]},
                        issues=[Issue("warn", f"this operation is not undoable ({r}) — "
                                      "nothing was restored for it") for r in skipped])

    @_mutating
    def redo(self, n: int = 1) -> Response:  # PR-7.2
        self._require_stage()
        ops = self.history.peek_redo(n)
        if not ops:
            return Response(command="redo", data={"redone": 0, "note": "nothing to redo"})
        before = self._stage_layer_snapshot()
        try:
            for op in ops:
                self._apply_compound(op, "redo")
        except Exception as exc:  # noqa: BLE001
            try:
                self._restore_stage_layer_snapshot(before)
            except Exception as rollback_exc:  # noqa: BLE001
                self._index_prims()
                return self._err("redo", RuntimeError(
                    f"redo failed: {exc}; rollback also failed: {rollback_exc}"))
            return self._err("redo", RuntimeError(f"redo failed and was rolled back: {exc}"))
        self.history.commit_redo(len(ops))
        self._index_prims()
        skipped = self._non_undoable_in(ops)
        summary = {"redone": len(ops)}
        if skipped:
            summary["not_redoable"] = len(skipped)
        return Response(command="redo", summary=summary,
                        data={"redone": len(ops), "ops": [o.label or o.command for o in ops]},
                        issues=[Issue("warn", f"this operation is not redoable ({r}) — "
                                      "nothing was re-applied for it") for r in skipped])

    @staticmethod
    def _non_undoable_in(ops: list[Op]) -> list[str]:
        """Reasons for every non-undoable entry among `ops` (recursing into compounds) —
        undo/redo must WARN about these instead of implying the stage was restored."""
        return [c.non_undoable for op in ops
                for c in (op.children if op.is_compound else [op]) if c.non_undoable]

    def _apply_compound(self, op: Op, direction: str) -> None:
        """Apply an op's undo/redo change, recursing into compound (chain/transaction) ops."""
        children = op.children if op.is_compound else [op]
        seq = reversed(children) if direction == "undo" else children
        for child in seq:
            if child.inverse and direction in child.inverse:
                self._apply_change(child.inverse[direction])

    def op_history(self, n: int | None = None) -> Response:  # PR-7.3
        entries = self.history.entries(n)
        return Response(command="history",
                        summary={"undo_depth": len(self.history.entries()),
                                 "can_redo": self.history.can_redo},
                        data={"ops": [e.label or e.command for e in entries]})

    # -- checkpoints (PR-7.4–7.6) ------------------------------------------------
    def checkpoint_save(self, name: str, full: bool = False) -> Response:  # PR-7.4
        self._require_stage()
        from usd_core.appearance import appearance_clear_active

        if appearance_clear_active(self._stage) and not full:
            return self._err(
                "checkpoint.save",
                RuntimeError(
                    "appearance clear is active in the session layer; a "
                    "default checkpoint exports only the root layer and "
                    "would silently omit the clean-slate mask. Use "
                    "`checkpoint save NAME --full`."
                ),
            )
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                self.config.state_dir.chmod(0o700)
            except OSError:
                pass
            self._session_dir("checkpoints").mkdir(parents=True, exist_ok=True, mode=0o700)
            target = self.checkpoint_path(name)
            # Export to a sibling temporary file and atomically publish it. This avoids
            # exposing a truncated checkpoint if the USD writer or process fails. The
            # uuid keeps concurrent same-process sessions (same pid!) from colliding.
            tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.usd")
            # --full flattens the composition into one self-contained layer (payloads /
            # references inlined) so the checkpoint reopens faithfully from anywhere; the
            # default exports just the root layer. Note: Stage.Flatten() returns an Sdf.Layer.
            if full:
                ok = self._stage.Flatten().Export(str(tmp))
            else:
                ok = self._stage.GetRootLayer().Export(str(tmp))
            if ok is False or not tmp.exists():
                raise RuntimeError("USD checkpoint export failed")
            tmp.replace(target)
        except Exception as exc:  # noqa: BLE001
            if "tmp" in locals():
                tmp.unlink(missing_ok=True)
            return self._err("checkpoint.save", exc)
        return Response(command="checkpoint.save", summary={"name": name, "path": str(target),
                        "full": full}, artifacts=[Artifact(path=str(target), kind="checkpoint",
                                                           label=name)])

    @_mutating
    def checkpoint_load(self, name: str) -> Response:  # PR-7.5
        self._require_stage()
        target = self.checkpoint_path(name)
        if not target.exists():
            return self._err("checkpoint.load", FileNotFoundError(f"no checkpoint '{name}'"))
        try:
            self._open(str(target))  # reopen the checkpoint as the live stage (resets refs/history)
        except Exception as exc:  # noqa: BLE001
            return self._err("checkpoint.load", exc)
        return Response(command="checkpoint.load", summary={"name": name, "path": str(target)})

    def checkpoint_list(self) -> Response:  # PR-7.6
        d = self._session_dir("checkpoints")
        names = sorted(p.stem for p in d.glob("*.usd*")) if d.exists() else []
        return Response(command="checkpoint.list",
                        summary={"count": len(names), "dir": str(d)},
                        data={"checkpoints": names})

    def checkpoint_delete(self, name: str) -> Response:
        target = self.checkpoint_path(name)
        existed = target.exists()
        if existed:
            target.unlink()
        return Response(command="checkpoint.delete", summary={"name": name, "deleted": existed})

    def checkpoint_path(self, name: str):
        import re

        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", name):
            raise ValueError("checkpoint name must be 1-64 letters, digits, '.', '_' or '-' and "
                             "must start with a letter or digit")
        if name in {".", ".."} or "/" in name or "\\" in name:
            raise ValueError("checkpoint name must not contain path separators")
        root = self._session_dir("checkpoints").resolve()
        target = (root / f"{name}.usd").resolve()
        if target.parent != root:
            raise ValueError("checkpoint path escapes the checkpoint directory")
        return target

    # -- io (PR-8) ---------------------------------------------------------------
    @staticmethod
    def _rebase_asset_paths(exported: str, old_anchor: str) -> int:
        """Rewrite relative asset paths in an exported layer for its new directory.

        `Export` copies asset paths verbatim, so a `./Payload/…` or `../Textures/…`
        reference silently stops resolving when the file lands in another directory —
        reopening then drops geometry/textures with no error. Returns #rewritten.
        """
        import os

        from pxr import Sdf, UsdUtils

        new_anchor = os.path.dirname(os.path.abspath(exported))
        if not old_anchor or os.path.abspath(old_anchor) == new_anchor:
            return 0
        layer = Sdf.Layer.FindOrOpen(str(exported))
        if not layer:
            return 0
        count = [0]

        def _rebase(asset_path: str) -> str:
            if not asset_path or "://" in asset_path or os.path.isabs(asset_path):
                return asset_path
            resolved = os.path.normpath(os.path.join(old_anchor, asset_path))
            if not os.path.exists(resolved):
                return asset_path  # already broken — leave it alone
            count[0] += 1
            return os.path.relpath(resolved, new_anchor)

        UsdUtils.ModifyAssetPaths(layer, _rebase)
        if count[0]:
            layer.Save()
        return count[0]

    @staticmethod
    def _strip_managed_cameras(exported: str) -> int:
        """Remove tool-managed cameras from a saved deliverable."""
        from pxr import Sdf

        layer = Sdf.Layer.FindOrOpen(str(exported))
        if not layer:
            return 0
        removed = 0

        def _walk(prim_spec):
            nonlocal removed
            for child in list(prim_spec.nameChildren):
                data = dict(child.customData or {})
                # "ovManagedCamera"/"dscManagedCamera" are the pre-rename keys —
                # stages saved by older builds still carry them and must strip
                # identically.
                if child.typeName == "Camera" and (
                        data.get("usdManagedCamera") or data.get("ovManagedCamera")
                        or data.get("dscManagedCamera")):
                    del prim_spec.nameChildren[child.name]
                    removed += 1
                else:
                    _walk(child)

        _walk(layer.pseudoRoot)
        if removed:
            layer.Save()
        return removed

    @staticmethod
    def _fsync_path(path: str) -> None:
        """Flush a written file's bytes to disk before anything depends on them."""
        # Windows' ``FlushFileBuffers`` rejects a handle opened for read-only
        # access with ``ERROR_INVALID_HANDLE`` (surfaced by ``os.fsync`` as
        # ``EBADF``).  Publish work files are owned by this session and remain
        # writable here, so request a writable handle on Windows.  Keep the
        # read-only POSIX open so publishing still works in directories where an
        # existing file is readable but not writable through this descriptor.
        flags = os.O_RDWR if os.name == "nt" else os.O_RDONLY
        fd = os.open(str(path), flags)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    @staticmethod
    def _atomic_export(root_layer, dest: str) -> None:
        """Export a layer to `dest` atomically: O_EXCL-reserve a temp sibling (same
        suffix, so USD picks the right crate/usda writer), export, fsync, then
        os.replace. Guarantees `dest` is never a half-written/corrupt file, even if
        the write is interrupted. The uuid keeps concurrent same-process sessions
        (same pid!) from sharing a temp name; O_EXCL makes a collision fail loudly
        instead of silently interleaving two writers."""
        import os
        dest = str(dest)
        tmp = (f"{dest}.dsctmp{os.getpid()}-{uuid.uuid4().hex[:8]}"
               f"{os.path.splitext(dest)[1]}")
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
        try:
            if root_layer.Export(tmp) is False:
                raise RuntimeError(f"USD export failed for {dest}")
            Session._fsync_path(tmp)
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # Full-value read-back is O(prims): read everything up to this cap, then sample.
    _DEEP_READ_FULL_PRIMS = 20000
    _DEEP_READ_SAMPLE_STRIDE = 97

    @classmethod
    def _deep_read_stage(cls, stage) -> int:
        """Force-read `stage` deeply enough to trip lazily-materialized corruption;
        returns the prim count.

        A damaged crate can pass a plain prim-index traversal (structure sections
        intact) and explode only when an attribute value, time-sample block, or
        relationship-target section is first decompressed. For every prim — all of
        the first 20k, then every 97th, so huge stages stay bounded — this reads
        each AUTHORED property: the attribute's default value plus its first/last
        time samples, and relationship targets. Raises whatever pxr raises on a
        damaged range; also reads the defaultPrim."""
        from pxr import Usd
        count = 0
        for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
            count += 1
            if count > cls._DEEP_READ_FULL_PRIMS and \
                    count % cls._DEEP_READ_SAMPLE_STRIDE:
                continue
            for prop in prim.GetAuthoredProperties():
                if isinstance(prop, Usd.Relationship):
                    prop.GetTargets()
                    continue
                prop.Get()
                times = prop.GetTimeSamples()
                if times:
                    prop.Get(times[0])
                    prop.Get(times[-1])
        stage.GetDefaultPrim()
        return count

    @staticmethod
    def _verify_usd_file(path: str) -> int:
        """Prove a just-written USD file actually reads back: reopen it from its BYTES
        (force-reload past any cached in-memory layer), then force-read every prim's
        authored attribute values, first/last time samples, and relationship targets
        (`_deep_read_stage`; bounded sampling on huge stages). Returns the prim count.
        Catches both truncated/mis-written crates ('Corrupt asset …: zero-copy data
        range out-of-bounds') and damaged value/time-sample blocks that a plain
        prim-index traversal reads straight past."""
        from pxr import Sdf, Usd
        try:
            cached = Sdf.Layer.Find(str(path))
            if cached is not None and cached.Reload(force=True) is False:
                # verify what's on disk, not what's in RAM — a refused reload means
                # we would be validating stale in-memory bytes
                raise RuntimeError("force-reload of the written file was refused")
            stage = Usd.Stage.Open(str(path))
            if not stage:
                raise RuntimeError("could not reopen the written file")
            return Session._deep_read_stage(stage)
        except Exception as exc:  # noqa: BLE001 — pxr raises Tf errors
            raise RuntimeError(
                f"verification of the written file failed ({exc})") from exc

    @staticmethod
    def _fsync_dir(path: str) -> None:
        """Fsync a directory so a just-renamed entry survives power loss (best
        effort: some filesystems refuse directory fsync)."""
        try:
            dfd = os.open(path or ".", os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except OSError:
            pass

    def _refuse_foreign_destination(self, real_dest: str, dest: str) -> None:
        """Refuse to publish onto a file ANOTHER live session has open read-write.

        `os.replace` would split that session's live stage (whose mmap pins the old
        inode) from what's on disk: its unsaved edits would no longer correspond to
        the file, and its next in-place save would silently clobber this publish.
        `real_dest` is the canonical (realpath) destination — the same keying the
        ownership registry uses."""
        with _ROOT_LAYER_LOCK:
            owner = _root_layer_owner(real_dest, self)
        if owner is not None:
            raise RuntimeError(
                f"'{dest}' is open (read-write) in session '{owner}' — refusing to "
                "publish over another session's live file; save from that session, "
                "close it there first, or publish to a new path")

    def _post_process_work(self, work: str, rebase_from: str | None,
                           strip_cameras: bool, info: dict) -> None:
        """Work-file post-passes shared by every publish flavor: asset-path rebase
        and managed-camera stripping."""
        if rebase_from:
            rebased = self._rebase_asset_paths(work, rebase_from)
            if rebased:
                info["rebased_asset_paths"] = rebased
        if strip_cameras:
            cameras = self._strip_managed_cameras(work)
            if cameras:
                info["stripped_render_cameras"] = cameras

    def _publish_file(self, dest: str, write_work) -> dict:
        """The verified-atomic publish pipeline shared by save/export/convert.

        `write_work(work_path, info)` writes the candidate bytes to the uniquely
        named sibling WORK file; then: fsync, read-back verify (deep value read),
        os.replace onto the destination, and fsync of the destination DIRECTORY so
        the rename itself is durable before success is reported.

        Cross-session safety: publishing onto a file another live session has open
        read-write is refused (see `_refuse_foreign_destination`), and a module-
        level per-destination lock serializes concurrent publishes to one path —
        two racing save-as calls used to interleave their export/verify/replace
        windows.

        Symlink policy: the destination is resolved to its canonical target
        (realpath) and the replace targets THAT file — publishing through a symlink
        writes through it and PRESERVES the link, keeping the ownership registry
        (also realpath-keyed), the bytes on disk, and any session's live layer all
        referring to one file. The resolved target is reported in
        `info["destination"]` when it differs from the argument.
        """
        dest = str(dest)
        real_dest = os.path.realpath(dest)
        info: dict = {}
        if real_dest != os.path.abspath(dest):
            info["destination"] = real_dest  # symlink preserved; its target replaced
        with _publish_dest_lock(real_dest):
            self._refuse_foreign_destination(real_dest, dest)
            work = (f"{real_dest}.dscsave{os.getpid()}-{uuid.uuid4().hex[:8]}"
                    f"{os.path.splitext(real_dest)[1]}")
            try:
                write_work(work, info)
                self._fsync_path(work)
                info["verified_prims"] = self._verify_usd_file(work)
                self._apply_dest_mode(work, real_dest)
                os.replace(work, real_dest)
                self._fsync_dir(os.path.dirname(real_dest))
            finally:
                if os.path.exists(work):
                    os.unlink(work)
        return info

    @staticmethod
    def _apply_dest_mode(work: str, real_dest: str) -> None:
        """Give the work file the permissions the destination should end with:
        the existing destination's mode when overwriting, else a fresh-file mode
        honoring the umask. USD's safe-output path writes mkstemp-style 0600
        temps, and os.replace carries that onto the deliverable — a saved asset
        nobody else on the host could read (benchmark round 5)."""
        try:
            if os.path.exists(real_dest):
                mode = stat.S_IMODE(os.stat(real_dest).st_mode)
            else:
                mode = 0o666 & ~_PROCESS_UMASK
            os.chmod(work, mode)
        except OSError:  # permissions are cosmetic next to a verified publish
            pass

    def _write_usdz_work(self, layer, work: str, *, rebase_from: str | None,
                         strip_cameras: bool, info: dict) -> None:
        """Write a stage-publish work file in usdz form. Sdf cannot write usdz
        directly in this build: export the layer to a temp sibling, post-process
        it, then let UsdUtils package it plus its resolved dependencies."""
        from pxr import Sdf, UsdUtils
        side = f"{work}.stage.usdc"
        try:
            if layer.Export(side) is False:
                raise RuntimeError(f"USD export failed for {work}")
            self._post_process_work(side, rebase_from, strip_cameras, info)
            if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(side), work):
                raise RuntimeError("usdz packaging failed")
        finally:
            if os.path.exists(side):
                os.unlink(side)

    @staticmethod
    def _disk_source_for_convert(source: str, scratch: list[str]) -> str:
        """A path whose bytes are guaranteed to be the ON-DISK `source`.

        USD caches SdfLayers process-wide, so `FindOrOpen(source)` returns the
        LIVE layer when any session in this daemon has the file open — with its
        unsaved in-memory edits (round 5: a convert leaked another session's
        transient render cameras into its output). Whenever a cached layer
        exists (dirty OR clean — a clean one can go dirty between the check and
        the open), copy the file's bytes to a temp sibling IN THE SAME DIRECTORY
        (relative references and usdz dependency walks must keep resolving) and
        convert that; the caller deletes paths appended to `scratch`.

        FAILS CLOSED: if the copy cannot be made, the conversion errors with
        guidance instead of silently converting live in-memory state. The
        snapshot is UNCONDITIONAL — gating it on Sdf.Layer.Find raced another
        session opening (and dirtying) the source between the check and the
        FindOrOpen. The read goes through an O_NOFOLLOW descriptor on the
        resolved path, so a symlink swapped in during a queue wait cannot
        redirect the copy (shared-daemon allowed_roots was validated against
        the original pathname)."""
        real = os.path.realpath(source)
        src_dir = os.path.dirname(real)
        tmp = None
        try:
            src_fd = os.open(real, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                fd, tmp = tempfile.mkstemp(
                    suffix=os.path.splitext(real)[1],
                    prefix=".ovconvert_", dir=src_dir)
                scratch.append(tmp)
                with os.fdopen(fd, "wb") as dst, os.fdopen(src_fd, "rb") as srcf:
                    src_fd = -1  # ownership moved to the context manager
                    shutil.copyfileobj(srcf, dst)
            finally:
                if src_fd >= 0:
                    os.close(src_fd)
        except OSError as exc:
            raise RuntimeError(
                f"'{source}' is open in a live session, and a disk-state copy for "
                f"the conversion could not be made ({exc}) — converting the live "
                "in-memory state would leak unsaved edits. Save the session first, "
                "or make the source's directory writable") from exc
        return tmp

    def _convert_usd_to_usd(self, source: str, out: str, out_suffix: str) -> dict:
        """USD→USD conversion body: export (or usdz-package) `source` to `out`
        through the verified-atomic publish pipeline. Returns the publish info.

        Converts read the file AS IT IS ON DISK (see _disk_source_for_convert) —
        never another session's live in-memory state. The snapshot copy is made
        for every USD source."""
        from pxr import Sdf
        scratch: list[str] = []
        try:
            disk_source = self._disk_source_for_convert(source, scratch)
            if out_suffix == ".usdz":
                from pxr import UsdUtils

                def write_work(work: str, info: dict) -> None:
                    # UsdUtils packages the source layer PLUS its resolved dependencies
                    # (textures/payloads) — self-contained, so no asset-path rebase.
                    if not UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(disk_source), work):
                        raise RuntimeError(f"usdz packaging of {source} failed")

                return self._publish_file(out, write_work)
            layer = Sdf.Layer.FindOrOpen(disk_source)
            if layer is None:
                raise RuntimeError(f"failed to open USD source: {source}")

            def write_work(work: str, info: dict) -> None:
                if layer.Export(work) is False:
                    raise RuntimeError(f"USD export to {out} failed")
                # keep relative payloads/references/textures resolving from the new dir
                self._post_process_work(
                    work, os.path.dirname(os.path.abspath(source)), False, info)

            return self._publish_file(out, write_work)
        finally:
            for tmp in scratch:
                try:
                    from pxr import Sdf as _Sdf
                    stale = _Sdf.Layer.Find(tmp)
                    if stale is not None:
                        del stale  # drop our reference before removing the file
                except Exception:  # noqa: BLE001 — cache hygiene only
                    pass
                if os.path.exists(tmp):
                    os.unlink(tmp)

    def _publish_stage(self, dest: str, *, flatten: bool = False,
                       rebase_from: str | None = None,
                       strip_cameras: bool = False) -> dict:
        """Atomically publish the live stage to `dest`, with read-back verification.

        The whole pipeline happens on a uniquely-named sibling WORK file: export
        (root layer, or the flattened composition), asset-path rebase, managed-camera
        strip, fsync, deep read-back verify — and only then os.replace onto the
        canonical destination (see `_publish_file` for the locking, cross-session
        refusal, symlink, and durability rules). Post-processing the work file
        (never `dest`) is what fixes the in-place-save corruption:
        `Sdf.Layer.FindOrOpen(dest)` returns the LIVE root layer when dest is the
        session's own open file, and `Save()`ing that rewrites the crate under its
        own mmap — the written file then reads back as 'Corrupt asset … zero-copy
        data range out-of-bounds'. Saving over the session's own root is safe here:
        the open stage's mmap keeps the old inode alive while the replace swaps in
        the verified new one.
        """
        layer = self._stage.Flatten() if flatten else self._stage.GetRootLayer()

        def write_work(work: str, info: dict) -> None:
            if Path(work).suffix.lower() == ".usdz":
                self._write_usdz_work(layer, work, rebase_from=rebase_from,
                                      strip_cameras=strip_cameras, info=info)
                return
            self._atomic_export(layer, work)
            self._post_process_work(work, rebase_from, strip_cameras, info)

        return self._publish_file(dest, write_work)

    @staticmethod
    def _dir_snapshot(target) -> set[str] | None:
        """The names present in `target`'s directory, taken immediately before a save
        attempt. Failed-save cleanup may only touch files that appeared AFTER this
        snapshot — that is the ownership proof."""
        if not target:
            return None
        try:
            return {p.name for p in Path(target).parent.iterdir()}
        except OSError:
            return set()

    @staticmethod
    def _cleanup_save_temps(target, before: set[str] | None) -> None:
        """Remove USD-writer temp litter after a failed save/export — by OWNERSHIP.

        A failed TfSafeOutputFile close leaves an mkstemp-style sibling next to the
        target (observed: `recording.VxRCU3` beside `recording.usda`). Only files that
        are NEW since the pre-save `before` snapshot AND match the mkstemp
        `<stem>.??????` / `<name>.??????` shape are deleted. A user's pre-existing
        look-alike (`model.2024Q1`) or another session's live temp — both present
        before our attempt — are never touched (the old name+age heuristic unlinked
        them).
        """
        if not target or before is None:
            return
        t = Path(target)
        pat = re.compile(
            "^(" + re.escape(t.stem) + "|" + re.escape(t.name) + ")"
            r"\.(?=[A-Za-z0-9]{6}$)[a-z]*[A-Z0-9][A-Za-z0-9]*$")
        try:
            siblings = list(t.parent.iterdir())
        except OSError:
            return
        for p in siblings:
            if p.name == t.name or p.name in before or not pat.match(p.name):
                continue
            try:
                if p.is_file():
                    p.unlink()
            except OSError:
                pass

    @staticmethod
    def _force_reload_layer(layer) -> None:
        """Force re-read a layer from disk; raises when pxr refuses or fails (kept
        as a separate seam so post-commit failure handling is testable)."""
        if layer.Reload(force=True) is False:
            raise RuntimeError("Reload(force=True) reported failure")

    def _reload_own_root_after_publish(self) -> list[Issue]:
        """Re-read the session's own root layer after an in-place publish, then
        rebuild the session caches. The publish already COMMITTED (the verified
        bytes replaced the file), so a reload failure here is a WARNING with a
        reopen hint — it used to be reported as 'save failed' although the file on
        disk was the good, verified save. Returns issues for the response."""
        try:
            self._force_reload_layer(self._stage.GetRootLayer())
        except Exception as exc:  # noqa: BLE001 — pxr raises Tf errors
            return [Issue(
                "warn",
                f"the save committed to disk, but reloading the live stage failed "
                f"({exc}) — the file is the verified save; run "
                f"`open {self._stage_path}` to reopen the stage from it")]
        # the reload may have dropped prims the publish stripped (managed render
        # cameras): re-index the refs and re-point the active camera at a live prim
        self._index_prims()
        if self._active_cam and \
                not self._stage.GetPrimAtPath(self._active_cam).IsValid():
            self._active_cam = next(iter(self._camera_paths()), None)
        return []

    @_mutating
    def save(self, path: str | None = None, flatten: bool = False,
             **_ignored) -> Response:  # PR-8.2
        self._require_stage()
        from usd_core.appearance import appearance_clear_active

        if appearance_clear_active(self._stage):
            out = str(path) if path else self._stage_path
            same_as_source = bool(out and self._stage_path) and os.path.realpath(
                out
            ) == os.path.realpath(self._stage_path)
            if same_as_source:
                return self._err(
                    "save",
                    RuntimeError(
                        "appearance clear leaves the opened source immutable; "
                        "save the composed result to a different derivative "
                        "path"
                    ),
                )
            if not flatten:
                return self._err(
                    "save",
                    RuntimeError(
                        "appearance clear is active in the session layer; "
                        "a normal save exports only the root layer and would "
                        "silently omit it. Use `save OUTPUT.usd --flatten` "
                        "with a derivative path."
                    ),
                )
        summary: dict = {}
        issues: list[Issue] = []
        if self._isolation:
            issues.append(Issue(
                "warning",
                f"isolate visibility opinions on {len(self._isolation)} prim(s) "
                "are being SAVED into the file — if the isolation was a "
                "temporary view, run `isolate --restore` and save again"))
        # snapshot the target directory NOW: on failure, cleanup may delete only
        # writer temps that appeared after this point (see _cleanup_save_temps)
        pre_save = self._dir_snapshot(path or self._stage_path)
        same_file = False
        try:
            out = str(path) if path else self._stage_path
            if out:
                root = self._stage.GetRootLayer()
                old_anchor = os.path.dirname(getattr(root, "realPath", "") or "") or None
                same_file = bool(self._stage_path) and \
                    os.path.realpath(out) == os.path.realpath(self._stage_path)
                # --flatten localizes every composition arc (sublayers/references/
                # payloads — incl. `material --library` arcs) into ONE layer; a
                # flattened layer has no external arcs left to rebase. External
                # texture/asset FILES stay referenced, not embedded (a .usdz target
                # packages them).
                info = self._publish_stage(
                    out, flatten=flatten,
                    rebase_from=None if flatten else old_anchor,
                    strip_cameras=True)
                summary.update(info)
                if flatten:
                    summary["flattened"] = True
                    issues.append(Issue(
                        "info",
                        "composition-flattened: sublayers/references/payloads were "
                        "localized into this one layer; external texture/asset "
                        "files remain referenced on disk, not embedded (save to "
                        ".usdz to package them)"))
            else:
                if flatten:
                    raise ValueError("this stage has no file yet — pass a target: "
                                     "save <file.usd> --flatten")
                self._stage.GetRootLayer().Save()
        except Exception as exc:  # noqa: BLE001
            self._cleanup_save_temps(path or self._stage_path, pre_save)
            return self._err("save", exc)
        if same_file:
            # the os.replace inside the publish was the COMMIT POINT; the live
            # stage still mmaps the replaced file's OLD inode — re-read so the
            # session serves what is now on disk. Failures past the commit are
            # warnings, never a 'save failed'.
            issues += self._reload_own_root_after_publish()
        if same_file or not path:
            # the open file now matches the session: nothing unsaved remains
            # (a save to a DIFFERENT path leaves the live stage's edits unsaved)
            self._saved_op_serial = self.history.total_recorded
            self._saved_mutation_epoch = self._mutation_epoch
        summary["path"] = out
        return Response(command="save", summary=summary, issues=issues,
                        artifacts=[Artifact(path=str(out), kind="export", label="save")])

    def viewer_snapshot(
        self,
        path: str,
        since_revision: int = -1,
        line_geometry: str | None = None,
    ) -> Response:
        """Publish a verified, immutable observation of the composed live stage."""
        self._require_stage()
        revision = int(self._viewer_revision)
        destination = Path(path.replace("{revision}", str(revision)))
        destination.parent.mkdir(parents=True, exist_ok=True)
        geometry_destination = (
            Path(line_geometry.replace("{revision}", str(revision)))
            if line_geometry else None
        )
        if geometry_destination is not None:
            geometry_destination.parent.mkdir(parents=True, exist_ok=True)
        geometry_key = (
            "\0".join(
                (
                    str(geometry_destination.parent.resolve()),
                    Path(line_geometry).name,
                )
            )
            if geometry_destination is not None and line_geometry is not None
            else None
        )
        cached_geometry = (
            self._viewer_line_geometry_publications.get(geometry_key)
            if geometry_key is not None else None
        )
        cached_geometry_available = bool(
            cached_geometry
            and isinstance(cached_geometry.get("path"), str)
            and Path(cached_geometry["path"]).is_file()
        )
        if (revision == int(since_revision) and destination.is_file()
                and (geometry_destination is None
                     or geometry_destination.is_file()
                     or cached_geometry_available)):
            return Response(command="viewer.snapshot", summary={
                "path": str(destination), "revision": revision,
                "generation": self._viewer_generation, "changed": False,
            })

        from pxr import Usd, UsdGeom, UsdLux
        from usd_core.spatial import get_world_bbox

        info = self._publish_stage(str(destination), flatten=True, strip_cameras=False)
        stage_bounds = get_world_bbox(self._stage, "/")
        light_count = sum(
            1 for prim in Usd.PrimRange(
                self._stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())
            if not prim.IsPseudoRoot() and prim.HasAPI(UsdLux.LightAPI)
        )
        try:
            size_bytes = destination.stat().st_size
            hasher = hashlib.sha256()
            with destination.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        except OSError as exc:
            return self._err("viewer.snapshot", exc)

        line_geometry_data = None
        if geometry_destination is not None:
            try:
                from usd_core.viewer_geometry import (
                    capture_line_geometry, publish_line_geometry_capture)
                if cached_geometry_available:
                    line_geometry_data = dict(cached_geometry)
                else:
                    if self._viewer_line_geometry_capture is None:
                        self._viewer_line_geometry_capture = capture_line_geometry(
                            self._stage)
                    line_geometry_data = publish_line_geometry_capture(
                        self._viewer_line_geometry_capture, geometry_destination)
                    self._viewer_line_geometry_publications[geometry_key] = dict(
                        line_geometry_data)
            except (OSError, RuntimeError, ValueError) as exc:
                return self._err("viewer.snapshot", exc)

        data = {
            "path": str(destination), "revision": revision,
            "generation": self._viewer_generation, "changed": True,
            "sha256": digest, "size_bytes": size_bytes,
            "up_axis": str(UsdGeom.GetStageUpAxis(self._stage)),
            "source_light_count": light_count, "bounds": stage_bounds,
        }
        if line_geometry_data is not None:
            data["line_geometry"] = line_geometry_data
        artifacts = [Artifact(
            path=str(destination), kind="viewer-snapshot",
            label=f"live-stage-r{revision}")]
        if line_geometry_data is not None:
            artifacts.append(Artifact(
                path=str(line_geometry_data["path"]), kind="viewer-line-geometry",
                label=f"live-lines-r{revision}"))
        return Response(
            command="viewer.snapshot", summary={**info, **data}, data=data,
            artifacts=artifacts)

    def sublayers(self, drop_dead: bool = False, **_ignored) -> Response:
        """List the root layer's sublayers; with drop_dead, remove ones that don't resolve
        (e.g. dead `omniverse://` or missing-file arcs that keep a repaired stage dirty)."""
        self._require_stage()
        if drop_dead and self.read_only:
            return self._err("sublayers", RuntimeError(
                f"session '{self.name}' is read-only (opened with --read-only) — "
                "--drop-dead would mutate the shared layer"))
        from pxr import Sdf
        root = self._stage.GetRootLayer()
        subs = list(root.subLayerPaths)
        resolved = []
        for s in subs:
            lyr = Sdf.Layer.FindOrOpenRelativeToLayer(root, s) or Sdf.Layer.FindOrOpen(s)
            resolved.append((s, bool(lyr)))
        dead = [s for s, ok in resolved if not ok]
        if drop_dead and dead:
            root.subLayerPaths = [s for s, ok in resolved if ok]
            self._mutation_epoch += 1  # a mutation outside the undo history
            self._viewer_revision += 1
            self._invalidate_viewer_line_geometry()
        return Response(command="sublayers",
                        summary={"total": len(subs), "dead": len(dead),
                                 "dropped": len(dead) if drop_dead else 0},
                        data={"sublayers": [{"path": s, "resolves": ok} for s, ok in resolved],
                              "dropped": dead if drop_dead else []})

    def export(self, format: str, path: str | None = None, **_ignored) -> Response:  # PR-8.3
        """Export the stage. USD formats use the USD writer; obj/glb/fbx are unsupported
        here and require a dedicated converter."""
        self._require_stage()
        fmt = format.lower()
        from usd_core.appearance import appearance_clear_active

        if appearance_clear_active(self._stage) and fmt in (
            "usd",
            "usda",
            "usdc",
            "usdz",
        ):
            return self._err(
                "export",
                RuntimeError(
                    "appearance clear is active in the session layer; "
                    "`export` would omit it. Use "
                    "`save OUTPUT.usd --flatten` instead."
                ),
            )
        out = path or (
            str(Path(self._stage_path).with_suffix("." + fmt))
            if self._stage_path
            else f"export.{fmt}"
        )
        pre_save = self._dir_snapshot(out)
        issues: list[Issue] = []
        same_file = False
        try:
            if fmt in ("usd", "usda", "usdc", "usdz"):
                # verified atomic publish (same pipeline as `save`): a bare
                # `Export(out)` over the session's own open crate corrupted it
                same_file = bool(self._stage_path) and \
                    os.path.realpath(out) == os.path.realpath(self._stage_path)
                if same_file and self.read_only:
                    # the post-replace reload would re-read the SHARED live layer,
                    # discarding the writer session's unsaved edits
                    raise RuntimeError(
                        f"session '{self.name}' is read-only (opened with "
                        "--read-only) — exporting over the open file would reload "
                        "the shared live layer out from under the writer session; "
                        "export to a new path instead")
                summary_info = self._publish_stage(out)
            else:
                return Response(command="export", ok=False,
                                issues=[Issue("error", f"export to '{fmt}' needs an external "
                                              "converter; USD formats (usd/usda/usdc/usdz) are "
                                              "supported natively")])
        except Exception as exc:  # noqa: BLE001
            self._cleanup_save_temps(out, pre_save)
            return self._err("export", exc)
        if same_file:
            # committed on disk; reload failures past this point are warnings
            issues += self._reload_own_root_after_publish()
        return Response(command="export",
                        summary={"format": fmt, "path": out, **summary_info},
                        issues=issues,
                        artifacts=[Artifact(path=out, kind="export", label=fmt)])

    def new(self, up: str | None = None, meters_per_unit: float | None = None,
            **_ignored) -> Response:  # PR-8.4
        """Create an empty in-memory stage. `up` = Y (default) or Z — set it to Z when you
        will reference Z-up source assets, so their orientation and gravity match."""
        from pxr import Usd, UsdGeom
        up_tok = UsdGeom.Tokens.z if str(up or "y").lower() == "z" else UsdGeom.Tokens.y
        stage = Usd.Stage.CreateInMemory()
        world = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world.GetPrim())
        UsdGeom.SetStageUpAxis(stage, up_tok)
        if meters_per_unit is not None:
            UsdGeom.SetStageMetersPerUnit(stage, float(meters_per_unit))
        # No baked-in light: a featureless authored dome washes metals out to flat
        # white AND suppresses the render-time default rig (gradient dome + key/fill),
        # which only engages when the stage has no lights of its own.
        self._stage = stage
        self._stage_path = None
        self.read_only = False  # a fresh private stage is always writable
        with _ROOT_LAYER_LOCK:  # the previous file (if any) is no longer held here
            self._release_layer_key_locked()
            self._release_reader_key_locked()
        self.refs.reset(); self.history.reset_baseline(); self._selection = []
        self._space_cache = None
        self._active_cam = None
        self._viewer_generation += 1
        self._invalidate_viewer_line_geometry()
        self.camera_create()
        n = self._index_prims()
        return Response(command="new", summary={"prims": n, "default_prim": "/World",
                                                "up_axis": str(up_tok)})

    def info(self, **_ignored) -> Response:  # PR-8.5
        from usd_core import query
        self._require_stage()
        try:
            data = query.stage_info(self._stage, self._stage_path)
        except Exception as exc:  # noqa: BLE001
            return self._err("info", exc)
        return Response(command="info", summary={"prims": data["prim_count"],
                        "up_axis": data["up_axis"]}, data=data)

    # -- parsing helpers for find ------------------------------------------------
    @staticmethod
    def _parse_box(in_box):
        if not in_box:
            return None
        vals = [[float(x) for x in str(item).split(",")] for item in in_box]
        if len(vals) != 2 or any(len(v) != 3 for v in vals):
            raise ValueError("--in-box expects two 'x,y,z' corners, e.g. --in-box 0,0,0 --in-box 5,5,5")
        return (vals[0], vals[1])

    @staticmethod
    def _parse_sphere(in_sphere):
        if not in_sphere:
            return None
        if len(in_sphere) != 2:
            raise ValueError("--in-sphere expects 'cx,cy,cz' then a radius")
        center = [float(x) for x in str(in_sphere[0]).split(",")]
        if len(center) != 3:
            raise ValueError("--in-sphere center must be 'cx,cy,cz'")
        return (center, float(in_sphere[1]))

    # -- escape hatch ------------------------------------------------------------
    def eval(self, code: str) -> Response:  # composition escape hatch (cli-design §0.8)
        raise NotImplementedError(
            "AST-sandboxed code-as-action over the ref namespace "
            "(reuse SpatialClaw kernel/safety.py + 3DAL AST remote exec)"
        )
