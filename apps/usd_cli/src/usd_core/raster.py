# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Analytic multimodal AOVs — depth / normals / segmentation / wireframe — via a small
numpy CPU rasterizer over the scene's geometry and the active camera's intrinsics.

Why analytic instead of renderer AOVs: it is backend-independent, deterministic, and gives
pixel-exact segmentation masks —
exactly what an agent's perception loop wants. The photoreal/beauty pass still goes through
the render backend; these structured passes come from geometry + the camera matrix.

Geometry sources: `UsdGeom.Mesh` prims (the vast majority of SimReady geometry) *and* the
analytic Gprims `Cube`/`Sphere`/`Cylinder`/`Cone`/`Capsule`/`Plane` — those are tessellated
here from their schema attributes so a shape `create`d as a Gprim shows up in the AOVs
exactly as it does in the beauty pass (which tessellates them via Hydra).

Scale: geometry is streamed one prim at a time (never the whole scene at once) and
rasterized in bulk numpy batches — triangulation, projection, coverage and z-testing are
all vectorized, with per-batch memory capped by `_PAIR_BUDGET`. Multi-million-triangle
scenes rasterize in seconds within a few hundred MB, instead of minutes of per-triangle
Python (which also ballooned memory on instance-heavy scenes).
"""

from __future__ import annotations

import math

import numpy as np


def _stable_color(i: int) -> tuple[int, int, int]:
    """Distinct, deterministic RGB per index via golden-ratio hue spacing."""
    import colorsys

    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


# ── geometry tessellation ─────────────────────────────────────────────────────────
_RADIAL = 48   # segments around a revolved primitive (sphere/cylinder/cone/capsule)
_STACKS = 24   # rings along the sweep (sphere latitude / capsule hemisphere)


def _axis(token) -> str:
    a = str(token or "Z").upper()
    return a if a in ("X", "Y", "Z") else "Z"


def _orient(pts: np.ndarray, axis: str) -> np.ndarray:
    """Rotate a Z-aligned point set so its Z axis maps to `axis` (cyclic, handedness-preserving).

    For radially symmetric primitives (and Plane's normal), only the axis mapping matters —
    roll about it is irrelevant. Cube/Sphere pass axis='Z' (no-op).
    """
    if axis == "Z":
        return pts
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    if axis == "X":  # Z -> X
        return np.stack([z, x, y], axis=1)
    return np.stack([y, z, x], axis=1)  # axis == "Y": Z -> Y


def _revolve(profile, radial: int = _RADIAL):
    """Surface of revolution about Z from a profile of (radius, z) samples.

    A profile point with radius≈0 is a pole (a single vertex); this makes sphere/cylinder/
    cone/capsule (with end caps) fall out of one triangulator. Consecutive rings are stitched
    as a quad strip (or a fan at a pole)."""
    pts: list[tuple[float, float, float]] = []
    rings: list[list[int]] = []
    for r, z in profile:
        if r <= 1e-9:
            rings.append([len(pts)])
            pts.append((0.0, 0.0, float(z)))
        else:
            ring = []
            for j in range(radial):
                phi = 2.0 * math.pi * j / radial
                ring.append(len(pts))
                pts.append((r * math.cos(phi), r * math.sin(phi), float(z)))
            rings.append(ring)
    tris: list[tuple[int, int, int]] = []
    for a, b in zip(rings[:-1], rings[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:  # pole -> ring (fan)
            for j in range(radial):
                # The first pole is the bottom cap.  Its outward normal points
                # down, the inverse of the side-strip winding below.
                tris.append((a[0], b[(j + 1) % radial], b[j]))
        elif len(b) == 1:  # ring -> pole (fan)
            for j in range(radial):
                tris.append((a[j], a[(j + 1) % radial], b[0]))
        else:  # ring -> ring (quad strip)
            for j in range(radial):
                a0, a1 = a[j], a[(j + 1) % radial]
                b0, b1 = b[j], b[(j + 1) % radial]
                tris.append((a0, a1, b1))
                tris.append((a0, b1, b0))
    return np.array(pts, dtype=np.float64), np.array(tris, dtype=np.int64)


def _cube(size: float):
    h = float(size) / 2.0
    v = np.array([(-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
                  (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)], dtype=np.float64) * h
    # Counter-clockwise when viewed from OUTSIDE.  The rasteriser preserves a
    # horizontal face's orientation as contact provenance: a crate's bottom face
    # seals its cavity when it rests coplanar on a floor.  Inward-facing triangles
    # reverse that fact and make the sealed interior look like usable clearance.
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 5, 4), (0, 1, 5),
         (1, 6, 5), (1, 2, 6), (2, 7, 6), (2, 3, 7), (3, 4, 7), (3, 0, 4)]
    return v, np.array(f, dtype=np.int64)


def _sphere(radius: float):
    r = float(radius)
    profile = [(r * math.sin(math.pi * i / _STACKS), r * -math.cos(math.pi * i / _STACKS))
               for i in range(_STACKS + 1)]  # pole (i=0, z=-r) .. pole (i=_STACKS, z=+r)
    return _revolve(profile)


def _cylinder(radius: float, height: float, axis: str):
    r, hz = float(radius), float(height) / 2.0
    pts, tris = _revolve([(0, -hz), (r, -hz), (r, hz), (0, hz)])  # capped
    return _orient(pts, axis), tris


def _cone(radius: float, height: float, axis: str):
    r, hz = float(radius), float(height) / 2.0
    pts, tris = _revolve([(0, -hz), (r, -hz), (0, hz)])  # base cap + side to apex
    return _orient(pts, axis), tris


def _capsule(radius: float, height: float, axis: str):
    r, hz = float(radius), float(height) / 2.0
    n = max(2, _STACKS // 2)
    bottom = [(r * math.cos(a), -hz + r * math.sin(a))
              for a in (-math.pi / 2 + (math.pi / 2) * k / n for k in range(n + 1))]
    top = [(r * math.cos(a), hz + r * math.sin(a))
           for a in ((math.pi / 2) * k / n for k in range(n + 1))]
    pts, tris = _revolve(bottom + top)  # seam (r,-hz)->(r,hz) is the cylinder wall
    return _orient(pts, axis), tris


def _plane(width: float, length: float, axis: str):
    hw, hl = float(width) / 2.0, float(length) / 2.0
    v = np.array([(-hw, -hl, 0.0), (hw, -hl, 0.0), (hw, hl, 0.0), (-hw, hl, 0.0)])
    return _orient(v, axis), np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)


def _mesh_geometry(prim):
    from pxr import UsdGeom

    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not indices:
        return None
    local = np.asarray(pts, dtype=np.float64)  # Vt buffer protocol — no per-point Python
    cnt = np.asarray(counts, dtype=np.int64)
    idx = np.asarray(indices, dtype=np.int64)
    # Vectorized fan triangulation: face f with c verts at offset s yields triangles
    # (idx[s], idx[s+j], idx[s+j+1]) for j = 1..c-2. Malformed faces (fewer than 3 verts,
    # or ranges/indices past the end of the buffers) are dropped rather than raising.
    ends = np.cumsum(cnt)
    starts = ends - cnt
    ntri = np.where((cnt >= 3) & (ends <= len(idx)), cnt - 2, 0)
    total = int(ntri.sum())
    if total == 0:
        return None
    face = np.repeat(np.arange(len(cnt)), ntri)
    j = np.arange(total) - np.repeat(np.cumsum(ntri) - ntri, ntri) + 1
    s = starts[face]
    tris = np.stack([idx[s], idx[s + j], idx[s + j + 1]], axis=1)
    tris = tris[(tris >= 0).all(axis=1) & (tris < len(local)).all(axis=1)]
    if len(tris) == 0:
        return None
    return local, tris


def _prim_local_geometry(prim):
    """Local-space (points[N,3], triangles[M,3]) for a Mesh or an analytic Gprim, else None."""
    from pxr import UsdGeom

    if prim.IsA(UsdGeom.Mesh):
        return _mesh_geometry(prim)
    if prim.IsA(UsdGeom.Cube):
        return _cube(UsdGeom.Cube(prim).GetSizeAttr().Get() or 2.0)
    if prim.IsA(UsdGeom.Sphere):
        return _sphere(UsdGeom.Sphere(prim).GetRadiusAttr().Get() or 1.0)
    if prim.IsA(UsdGeom.Cylinder):
        c = UsdGeom.Cylinder(prim)
        return _cylinder(c.GetRadiusAttr().Get() or 1.0, c.GetHeightAttr().Get() or 2.0,
                         _axis(c.GetAxisAttr().Get()))
    if prim.IsA(UsdGeom.Cone):
        c = UsdGeom.Cone(prim)
        return _cone(c.GetRadiusAttr().Get() or 1.0, c.GetHeightAttr().Get() or 2.0,
                     _axis(c.GetAxisAttr().Get()))
    if prim.IsA(UsdGeom.Capsule):
        c = UsdGeom.Capsule(prim)
        return _capsule(c.GetRadiusAttr().Get() or 0.5, c.GetHeightAttr().Get() or 1.0,
                        _axis(c.GetAxisAttr().Get()))
    if hasattr(UsdGeom, "Plane") and prim.IsA(UsdGeom.Plane):
        c = UsdGeom.Plane(prim)
        return _plane(c.GetWidthAttr().Get() or 1.0, c.GetLengthAttr().Get() or 1.0,
                      _axis(c.GetAxisAttr().Get()))
    return None


def _is_supported_geometry_prim(prim) -> bool:
    """Whether ``prim`` is a geometry schema this rasterizer can ingest cheaply."""
    from pxr import UsdGeom

    schemas = (UsdGeom.Mesh, UsdGeom.Cube, UsdGeom.Sphere, UsdGeom.Cylinder,
               UsdGeom.Cone, UsdGeom.Capsule)
    return any(prim.IsA(schema) for schema in schemas) or (
        hasattr(UsdGeom, "Plane") and prim.IsA(UsdGeom.Plane))


def _iter_geometry(stage, ref_for_path, *, include_prim=None):
    """Yield (path, ref, world_points[N,3], triangles[M,3]) for visible active geometry,
    one prim at a time — streaming keeps peak memory at one prim's geometry, not the scene's.

    ``include_prim``, when supplied, receives a visible active prim before its local
    geometry is read or tessellated.  Scope-aware consumers use this hook with a USD
    bounds cache, so an out-of-window dense mesh is rejected before allocating points
    or triangulating faces.  Render callers leave it unset and retain the original
    ingest path.
    """
    from pxr import Usd, UsdGeom

    xcache = UsdGeom.XformCache(Usd.TimeCode.Default())
    # descend into native-instance proxies — otherwise instanced geometry rasterizes blank
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsActive():
            continue
        if not _is_supported_geometry_prim(prim):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if include_prim is not None and not include_prim(prim):
            continue
        geo = _prim_local_geometry(prim)
        if geo is None:
            continue
        local, tris = geo
        if local.shape[0] == 0 or tris.shape[0] == 0:
            continue
        # local -> world (row-vector convention)
        l2w = xcache.GetLocalToWorldTransform(prim)
        m = np.array([[l2w[r][c] for c in range(4)] for r in range(4)], dtype=np.float64)
        ph = np.concatenate([local, np.ones((len(local), 1))], axis=1)
        world = (ph @ m)[:, :3]
        # The detector uses the cross-product normal to retain upper/lower face
        # provenance.  USD may reverse that normal in two independent ways: a
        # left-handed Gprim changes the declared front-face winding, and a mirrored
        # local-to-world transform reverses handedness.  Normalize to outward world
        # winding before any consumer (AOV or heightfield) sees these triangles.
        # Analytic Gprims are tessellated above with our own already-outward index
        # order.  Only Mesh face indices carry authored USD orientation semantics.
        orientation = UsdGeom.Gprim(prim).GetOrientationAttr().Get()
        left_handed = (prim.IsA(UsdGeom.Mesh)
                       and orientation == UsdGeom.Tokens.leftHanded)
        mirrored = np.linalg.det(m[:3, :3]) < 0.0
        if left_handed != mirrored:
            tris = tris[:, [0, 2, 1]]
        path = prim.GetPath().pathString
        yield path, ref_for_path(path), world, tris


def _project(world_pts, params):
    """Project Nx3 world points to screen (u, v, depth) arrays (row-vector w2c)."""
    ph = np.concatenate([world_pts, np.ones((len(world_pts), 1))], axis=1)
    cam = ph @ params["w2c"]
    zc = cam[:, 2]
    safe = np.where(np.abs(zc) < 1e-9, -1e-9, zc)
    inv = -1.0 / safe
    u = params["fx"] * (cam[:, 0] * inv) + params["cx"]
    v = params["fy"] * (-cam[:, 1] * inv) + params["cy"]
    depth = -zc  # >0 in front of the camera
    return np.stack([u, v, depth], axis=1)


def _feature_edges(world, tris, scr, crease_cos: float = 0.70):
    """Vertex-index edges worth drawing for a wireframe of a *dense* mesh.

    Drawing every triangle edge of a 100k-tri mesh just fills the silhouette solid. A
    readable wireframe is a line drawing: an edge is kept when it is a **silhouette**
    (its two faces point opposite ways in screen space), a **boundary** (one face), or a
    **crease** (dihedral angle sharper than ~45°). Smooth tessellation (sphere facets) is
    dropped; hard structure (box edges, object outlines) stays. Fully vectorized.
    """
    T = len(tris)
    if T == 0:
        return np.empty((0, 2), dtype=np.int64)
    a, b, c = scr[tris[:, 0], :2], scr[tris[:, 1], :2], scr[tris[:, 2], :2]
    facing = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
              - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])) >= 0.0  # screen-space winding
    n = np.cross(world[tris[:, 1]] - world[tris[:, 0]], world[tris[:, 2]] - world[tris[:, 0]])
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)

    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=0)
    fid = np.tile(np.arange(T), 3)
    es = np.sort(e, axis=1)
    nv = int(world.shape[0])
    key = es[:, 0].astype(np.int64) * nv + es[:, 1].astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_s, es_s, fid_s = key[order], es[order], fid[order]
    bnd = np.nonzero(np.concatenate([[True], key_s[1:] != key_s[:-1], [True]]))[0]
    starts, counts = bnd[:-1], np.diff(bnd)

    keep = (counts == 1) | (counts > 2)          # boundary / non-manifold: always
    man = np.nonzero(counts == 2)[0]
    if len(man):
        s = starts[man]
        f0, f1 = fid_s[s], fid_s[s + 1]
        sil = facing[f0] != facing[f1]
        crease = np.einsum("ij,ij->i", n[f0], n[f1]) < crease_cos
        keep[man] = sil | crease
    return es_s[starts[keep]]


# ── vectorized z-buffered rasterization ───────────────────────────────────────────
# The z-buffer is a flat int64 image of encoded keys: (float32 depth bits << 24) | local
# triangle id. Positive-float bit patterns sort like the floats, so a scatter-min on keys
# is a z-test; the triangle id makes each candidate's key (near-)unique, so after the
# scatter `enc[pixel] == key` identifies the winning candidate and its attributes (prim
# color, face normal) are written for exactly the visible surface.
_ENC_EMPTY = np.int64(np.iinfo(np.int64).max)
_ID_BITS = 24
# Cap on candidate (triangle × covered-pixel) pairs materialized per batch — bounds peak
# memory to a few hundred MB regardless of scene size. Also caps batch triangle count
# below 2^24, so local ids always fit the key's id field.
_PAIR_BUDGET = 4_000_000


def _tri_normal_colors(world, tris, cam_origin):
    """Camera-facing unit normals per triangle, encoded as uint8 RGB [M,3]."""
    w0, w1, w2 = world[tris[:, 0]], world[tris[:, 1]], world[tris[:, 2]]
    n = np.cross(w1 - w0, w2 - w0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln < 1e-12, 1.0, ln)
    # orient toward the camera so the visible surface normal faces the viewer
    flip = np.einsum("ij,ij->i", n, cam_origin - (w0 + w1 + w2) / 3.0) < 0
    n[flip] = -n[flip]
    return ((n * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)


def _rasterize_mesh(scr, tris, enc, seg_flat, nrm_flat, color, tri_rgb,
                    width, height) -> None:
    """Scatter one prim's triangles into the encoded z-buffer + attribute images.

    scr: projected vertices [N,3] (u, v, depth); tris: vertex-index triangles [M,3];
    enc: flat encoded z-buffer [H*W]; seg_flat/nrm_flat: flat uint8 [H*W,3] or None;
    tri_rgb: per-triangle normal colors [M,3] or None (aligned with `tris`).
    """
    z = scr[:, 2]
    front = (z[tris[:, 0]] > 0) & (z[tris[:, 1]] > 0) & (z[tris[:, 2]] > 0)
    tris = tris[front]  # behind camera — skip (no near-clip split)
    if tri_rgb is not None:
        tri_rgb = tri_rgb[front]
    if len(tris) == 0:
        return

    p0, p1, p2 = scr[tris[:, 0]], scr[tris[:, 1]], scr[tris[:, 2]]
    ax, ay, az = p0[:, 0], p0[:, 1], p0[:, 2]
    bx, by, bz = p1[:, 0], p1[:, 1], p1[:, 2]
    cx, cy, cz = p2[:, 0], p2[:, 1], p2[:, 2]
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    minx, maxx = np.minimum.reduce([ax, bx, cx]), np.maximum.reduce([ax, bx, cx])
    miny, maxy = np.minimum.reduce([ay, by, cy]), np.maximum.reduce([ay, by, cy])
    ok = ((np.abs(denom) >= 1e-12) & (maxx >= 0) & (minx <= width - 1)
          & (maxy >= 0) & (miny <= height - 1))
    if not ok.any():
        return
    keep = np.nonzero(ok)[0]
    if tri_rgb is not None:
        tri_rgb = tri_rgb[keep]  # keep tri_rgb aligned with the keep-space `tid` indices
    x0 = np.clip(np.floor(minx[keep]), 0, width - 1).astype(np.int64)
    x1 = np.clip(np.ceil(maxx[keep]), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(miny[keep]), 0, height - 1).astype(np.int64)
    y1 = np.clip(np.ceil(maxy[keep]), 0, height - 1).astype(np.int64)
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    npix = bw * bh
    offs = np.concatenate([[0], np.cumsum(npix)])

    # per-triangle edge-function constants (float32: pair-level math at half the memory)
    e1x, e1y = (by - cy)[keep], (cx - bx)[keep]
    e2x, e2y = (cy - ay)[keep], (ax - cx)[keep]
    rx, ry = cx[keep], cy[keep]
    inv_den = (1.0 / denom[keep]).astype(np.float32)
    za, zb, zc = az[keep].astype(np.float32), bz[keep].astype(np.float32), cz[keep].astype(np.float32)
    e1x, e1y = e1x.astype(np.float32), e1y.astype(np.float32)
    e2x, e2y = e2x.astype(np.float32), e2y.astype(np.float32)
    rxf, ryf = rx.astype(np.float32), ry.astype(np.float32)

    start = 0
    n_tris = len(keep)
    while start < n_tris:
        end = int(np.searchsorted(offs, offs[start] + _PAIR_BUDGET, side="left"))
        end = min(max(end, start + 1), n_tris)
        total = int(offs[end] - offs[start])
        # expand each triangle's bbox into (tid, pixel) candidate pairs
        tid = np.repeat(np.arange(start, end), npix[start:end])
        k = np.arange(total, dtype=np.int64) - (offs[tid] - offs[start])
        pxi = x0[tid] + k % bw[tid]
        pyi = y0[tid] + k // bw[tid]
        dx = (pxi + 0.5).astype(np.float32) - rxf[tid]
        dy = (pyi + 0.5).astype(np.float32) - ryf[tid]
        l1 = (e1x[tid] * dx + e1y[tid] * dy) * inv_den[tid]
        l2 = (e2x[tid] * dx + e2y[tid] * dy) * inv_den[tid]
        l3 = 1.0 - l1 - l2
        inside = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
        if inside.any():
            tid, l1, l2, l3 = tid[inside], l1[inside], l2[inside], l3[inside]
            pix = pyi[inside] * width + pxi[inside]
            # Screen-space barycentrics require reciprocal interpolation to recover
            # perspective-correct camera-space depth.
            depth = 1.0 / (
                l1 / za[tid]
                + l2 / zb[tid]
                + l3 / zc[tid]
            )
            key = ((depth.view(np.uint32).astype(np.int64) << _ID_BITS)
                   | ((tid - start) & ((1 << _ID_BITS) - 1)))
            np.minimum.at(enc, pix, key)
            won = enc[pix] == key
            wpix = pix[won]
            if seg_flat is not None:
                seg_flat[wpix] = color
            if nrm_flat is not None:
                nrm_flat[wpix] = tri_rgb[tid[won]]
        start = end


def render_aovs(stage, camera_path, width, height, modalities, ref_for_path) -> dict:
    """Rasterize requested AOVs. Returns {modality: PIL.Image, ..., "legend": {...}}.

    modalities ⊆ {"depth", "normals", "segmentation", "wireframe"}.

    A requested ``depth`` pass returns both the normalized uint8 ``depth`` preview
    and a float32 ``linear_depth`` array in meters.  The preview is intentionally
    unsuitable for geometric comparisons; callers that need metric evidence must
    consume the raw array.
    """
    from PIL import Image, ImageDraw
    from pxr import UsdGeom

    from usd_core.camera import camera_params

    params = camera_params(stage, camera_path, width, height)

    want_depth = "depth" in modalities
    want_norm = "normals" in modalities
    want_seg = "segmentation" in modalities
    want_wire = "wireframe" in modalities
    want_fill = want_depth or want_seg or want_norm

    enc = np.full(height * width, _ENC_EMPTY, dtype=np.int64) if want_fill else None
    seg = np.zeros((height * width, 3), dtype=np.uint8) if want_seg else None
    nrm = np.zeros((height * width, 3), dtype=np.uint8) if want_norm else None
    legend: dict[str, list[int]] = {}
    cam_origin = np.linalg.inv(params["w2c"])[3, :3]

    wire_img = Image.new("RGB", (width, height), (12, 12, 16)) if want_wire else None
    wire_draw = ImageDraw.Draw(wire_img) if want_wire else None

    seg_candidates: list[tuple[str, tuple[int, int, int]]] = []  # (label, color) per prim
    for idx, (path, ref, world, tris) in enumerate(_iter_geometry(stage, ref_for_path)):
        scr = _project(world, params)  # [N,3] u,v,depth
        color = _stable_color(idx)
        if want_seg:
            # Defer legend entries until after rasterization: only prims that actually
            # win visible pixels belong in the legend. Emitting one per iterated prim
            # ballooned the legend to ~20k entries (830K JSON tokens) on instanced scenes.
            seg_candidates.append((ref or path, color))
        if want_wire:
            e = _feature_edges(world, tris, scr)
            if len(e):
                vis = (scr[e[:, 0], 2] > 0) & (scr[e[:, 1], 2] > 0)  # both endpoints in front
                segs = np.concatenate([scr[e[vis, 0], :2], scr[e[vis, 1], :2]], axis=1)
                for x0, y0, x1, y1 in segs.tolist():
                    wire_draw.line([(x0, y0), (x1, y1)], fill=(210, 210, 220), width=1)
        if want_fill:
            tri_rgb = _tri_normal_colors(world, tris, cam_origin) if want_norm else None
            _rasterize_mesh(scr, tris, enc, seg, nrm, color, tri_rgb, width, height)

    result: dict = {}
    if want_wire:
        result["wireframe"] = wire_img
    if want_depth:
        result["depth"] = _depth_image(enc, width, height)
        result["linear_depth"] = _linear_depth_meters(
            enc,
            width,
            height,
            meters_per_unit=float(UsdGeom.GetStageMetersPerUnit(stage)),
        )
    if want_seg:
        result["segmentation"] = Image.fromarray(seg.reshape(height, width, 3), "RGB")
        # Legend = only prims whose color survived into visible pixels.
        px = seg.reshape(-1, 3)
        visible = set(map(tuple, np.unique(px[np.any(px != 0, axis=1)], axis=0).tolist()))
        legend = {label: list(color) for label, color in seg_candidates
                  if tuple(color) in visible}
        result["legend"] = legend
    if want_norm:
        result["normals"] = Image.fromarray(nrm.reshape(height, width, 3), "RGB")
    return result


def _depth_image(enc, width, height):
    """Normalized grayscale depth from the encoded z-buffer: near = bright, far = dark,
    background = black."""
    from PIL import Image

    finite = enc != _ENC_EMPTY
    img = np.zeros(height * width, dtype=np.uint8)
    if finite.any():
        bits = ((enc[finite] >> _ID_BITS) & 0xFFFFFFFF).astype(np.uint32)
        d = bits.view(np.float32).astype(np.float64)
        lo, hi = float(d.min()), float(d.max())
        rng = (hi - lo) or 1.0
        img[finite] = (255.0 * (1.0 - (d - lo) / rng)).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img.reshape(height, width), "L")


def _linear_depth_meters(
    enc: np.ndarray,
    width: int,
    height: int,
    *,
    meters_per_unit: float,
) -> np.ndarray:
    """Decode the z-buffer into camera-space linear depth measured in meters."""

    if not np.isfinite(meters_per_unit) or meters_per_unit <= 0:
        raise ValueError("USD stage metersPerUnit must be a positive finite value")
    finite = enc != _ENC_EMPTY
    depth = np.full(width * height, np.nan, dtype=np.float32)
    if finite.any():
        bits = ((enc[finite] >> _ID_BITS) & 0xFFFFFFFF).astype(np.uint32)
        stage_units = bits.view(np.float32)
        depth[finite] = stage_units * np.float32(meters_per_unit)
    return depth.reshape(height, width)
