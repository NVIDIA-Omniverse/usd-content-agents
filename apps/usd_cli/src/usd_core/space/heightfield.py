# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rasterized multi-layer solid heightfield -> free spans (Recast-style).

This replaced ESD's single-ray-per-cell sampler (``esd/kernels.py``, since removed). That
sampler cast ONE down-ray + ONE up-ray from each cell centre, so:

  * thin structures (pallet slats, rack beams, shelf lips) that miss the exact
    centre line were invisible -> real support surfaces dropped / free space
    reported where a beam actually is (the "space not included" artefacts);
  * lateral occlusion was ignored entirely -> a column beside/through the rack
    frame reported a tall free volume that in reality is full of structure (the
    absurd ``/World`` placements).

Instead we RASTERISE every triangle into the XY grid (à la Recast's
``rcRasterizeTriangle`` -> compact heightfield):

  1. one thread per triangle projects it onto the cells its XY footprint covers
     (conservative 2D triangle-vs-cell SAT overlap, so thin/vertical faces are
     never missed) and appends a SOLID span ``[z_lo, z_hi]`` per covered cell;
  2. one thread per cell sorts its solid spans by height and MERGES overlapping/
     adjacent ones (the compact-heightfield storage saving — a rack column of
     many coincident faces collapses to a handful of layers);
  3. the FREE spans are the gaps ABOVE each solid layer (each free span therefore
     rests on a real support surface), inverted into the exact same struct-of-
     arrays ESD's sampler produced, so ``smooth``/``extract`` are unchanged.

Because a lateral beam's triangles now land a solid span in every cell they
overlap, the free-span inversion excludes those heights automatically — the same
multi-layer heightfield that fixes aliasing also gives the extractor its lateral-
collision awareness for free. Runs device-agnostically on CPU and CUDA.
"""

from __future__ import annotations

import math

import numpy as np
from ._warp import warp as _warp

wp = _warp()


# Solid layers held per cell before merging. This is a RAW per-triangle span
# budget, NOT the merged-layer count: a cell straddling the rack frame collects
# every beam/bracket/upright triangle at every shelf level, each emitting several
# near-duplicate spans, so raw counts reach the low hundreds (measured p99≈300)
# even though they merge down to ~15 real layers. 64 was far too small — the
# atomic buffer overflowed and DROPPED spans (pallets are appended last, so they
# vanished first), corrupting the free-space inversion into a jagged fringe along
# every pallet/frame edge. 256 covers p99; genuine overflow beyond it only occurs
# at dense structure cores (uprights/brace intersections), which are solid anyway
# and are handled conservatively (marked fully solid, never invents free space).
MAX_SOLID = wp.constant(256)
MAX_FREE = wp.constant(16)
#: Default coincidence tolerance for the rasteriser's solid-layer merge, in DETECTOR
#: metres. Two solid layers in one column with less air than this between them become
#: one layer, which is what makes an object *resting* on something read as contact
#: rather than as two surfaces a float-epsilon apart.
#:
#: It is a coincidence tolerance, not a clearance one, and 1 mm is generous for that:
#: float32 noise at PCB coordinates is ~1e-7 m. The physical filter is a different
#: knob — ``min_free`` drops any free span shorter than the object — which is why the
#: value is invisible in normal use: measured on the shipped PCB with a 5 mm object,
#: 1e-3, 1e-6 and 0 all return the identical 71 regions / 4112 cells.
#:
#: It becomes visible in exactly one regime: an object SHORTER than the gap. A merged
#: gap is under 1 mm by construction, so it can only have held an object under 1 mm
#: tall — and there the merge deletes a real resting surface. Two plates 0.9 mm apart
#: report one surface at the default and two at ``merge_gap=0``. Hence the parameter:
#: sub-millimetre work sets its own tolerance instead of arguing with this number.
DEFAULT_MERGE_GAP = 1.0e-3

# Slope-handling modes. A triangle whose surface tilts more than
# ``slope_threshold_deg`` from horizontal but less than ``vertical_cutoff_deg``
# (a ramp/incline, NOT a near-vertical wall) is treated per this mode:
#   IGNORE  — the ramp is skipped entirely; it contributes no solid at all, so it
#             is invisible to the detector (nothing rests on it, it occludes
#             nothing). Safest against corruption; risks placement through a ramp.
#   FLATTEN — the ramp is kept as a solid up to its PEAK z (the whole-triangle
#             slab, top at the highest vertex). The incline is reported as a flat
#             platform at its highest point. Conservative (over-blocks the wedge).
#   TERRAIN — the ramp is rasterised at its ACTUAL per-cell height (a true stepped
#             heightfield), so the free space beside/under it is preserved, not
#             destroyed. Ramp cells are flagged steep and excluded as rest
#             surfaces (nothing is placed on the incline itself). Recommended.
# Near-vertical faces (walls) and near-horizontal faces (decks) are UNAFFECTED by
# the mode — walls always occlude as a full-height slab, decks stay thin support.
SLOPE_IGNORE = wp.constant(0)
SLOPE_FLATTEN = wp.constant(1)
SLOPE_TERRAIN = wp.constant(2)
_SLOPE_MODES = {"ignore": 0, "flatten": 1, "terrain": 2}


@wp.func
def _tri_box_overlap_2d(
    ax: wp.float32, ay: wp.float32,
    bx: wp.float32, by: wp.float32,
    cx: wp.float32, cy: wp.float32,
    minx: wp.float32, miny: wp.float32,
    maxx: wp.float32, maxy: wp.float32,
) -> wp.int32:
    """2D separating-axis test: does triangle (a,b,c) overlap the AABB square?

    Conservative — returns 1 whenever the triangle touches the cell, so a thin
    slat/beam that grazes the cell is counted (never point-sampled away)."""
    # Box axes (x, y).
    if wp.max(ax, wp.max(bx, cx)) < minx or wp.min(ax, wp.min(bx, cx)) > maxx:
        return int(0)
    if wp.max(ay, wp.max(by, cy)) < miny or wp.min(ay, wp.min(by, cy)) > maxy:
        return int(0)

    # Box centre + half extents; work in box-centred coords.
    hx = 0.5 * (maxx - minx)
    hy = 0.5 * (maxy - miny)
    ox = 0.5 * (minx + maxx)
    oy = 0.5 * (miny + maxy)
    v0x = ax - ox; v0y = ay - oy
    v1x = bx - ox; v1y = by - oy
    v2x = cx - ox; v2y = cy - oy

    # Three triangle-edge normals as separating axes.
    ex = v1x - v0x; ey = v1y - v0y
    nx = -ey; ny = ex
    p0 = v0x * nx + v0y * ny
    p1 = v1x * nx + v1y * ny
    p2 = v2x * nx + v2y * ny
    r = hx * wp.abs(nx) + hy * wp.abs(ny)
    if wp.min(p0, wp.min(p1, p2)) > r or wp.max(p0, wp.max(p1, p2)) < -r:
        return int(0)

    ex = v2x - v1x; ey = v2y - v1y
    nx = -ey; ny = ex
    p0 = v0x * nx + v0y * ny
    p1 = v1x * nx + v1y * ny
    p2 = v2x * nx + v2y * ny
    r = hx * wp.abs(nx) + hy * wp.abs(ny)
    if wp.min(p0, wp.min(p1, p2)) > r or wp.max(p0, wp.max(p1, p2)) < -r:
        return int(0)

    ex = v0x - v2x; ey = v0y - v2y
    nx = -ey; ny = ex
    p0 = v0x * nx + v0y * ny
    p1 = v1x * nx + v1y * ny
    p2 = v2x * nx + v2y * ny
    r = hx * wp.abs(nx) + hy * wp.abs(ny)
    if wp.min(p0, wp.min(p1, p2)) > r or wp.max(p0, wp.max(p1, p2)) < -r:
        return int(0)

    return int(1)


@wp.kernel
def rasterize_solid_kernel(
    points: wp.array(dtype=wp.vec3),
    indices: wp.array(dtype=wp.int32),
    face_object: wp.array(dtype=wp.int32),
    origin_x: wp.float32,
    origin_y: wp.float32,
    cell_size: wp.float32,
    nx: wp.int32,
    ny: wp.int32,
    min_z: wp.float32,
    max_z: wp.float32,
    slope_mode: wp.int32,
    cos_flat: wp.float32,
    cos_wall: wp.float32,
    solid_min: wp.array(dtype=wp.float32),
    solid_max: wp.array(dtype=wp.float32),
    solid_obj: wp.array(dtype=wp.int32),
    solid_obj_top: wp.array(dtype=wp.int32),
    solid_top_down: wp.array(dtype=wp.int32),
    solid_cavity_owner: wp.array(dtype=wp.int32),
    solid_steep: wp.array(dtype=wp.int32),
    solid_cnt: wp.array(dtype=wp.int32),
):
    t = wp.tid()
    i0 = indices[3 * t + 0]
    i1 = indices[3 * t + 1]
    i2 = indices[3 * t + 2]
    p0 = points[i0]
    p1 = points[i1]
    p2 = points[i2]

    # Face normal + inclination from horizontal. cos_tilt = |n_z| / |n| is 1 for a
    # horizontal deck and 0 for a vertical wall. A face is a RAMP when it tilts
    # more than the slope threshold (cos_tilt < cos_flat) but is not near-vertical
    # (cos_tilt > cos_wall); decks and walls are handled exactly as before so
    # axis-aligned warehouse geometry is unchanged.
    n = wp.cross(p1 - p0, p2 - p0)
    nn = wp.length(n)
    if nn <= 1.0e-12:
        return  # degenerate triangle — no surface
    cos_tilt = wp.abs(n[2]) / nn
    is_ramp = int(0)
    if cos_tilt < cos_flat and cos_tilt > cos_wall:
        is_ramp = int(1)

    # IGNORE mode: a ramp contributes no geometry at all.
    if is_ramp == 1 and slope_mode == SLOPE_IGNORE:
        return
    # TERRAIN mode: a ramp is a rest surface nothing may be placed on, so its cells
    # carry a steep flag that `invert_free_kernel` uses to drop the span above them.
    terrain = int(0)
    if is_ramp == 1 and slope_mode == SLOPE_TERRAIN:
        terrain = int(1)
    # Clip the face to each cell (below) whenever its plane is well-conditioned in z,
    # i.e. anything that is not near-vertical. FLATTEN keeps the peak-height slab for
    # ramps, which is what that mode means.
    clip = int(0)
    if cos_tilt > cos_wall and not (is_ramp == 1 and slope_mode == SLOPE_FLATTEN):
        clip = int(1)

    # Triangle vertical extent, clamped to the scope. Zero-thickness (a flat deck)
    # is kept — it is a valid support surface.
    tzmin = wp.min(p0[2], wp.min(p1[2], p2[2]))
    tzmax = wp.max(p0[2], wp.max(p1[2], p2[2]))
    if tzmax < min_z or tzmin > max_z:
        return
    tzmin = wp.max(tzmin, min_z)
    tzmax = wp.min(tzmax, max_z)

    # XY cell footprint of the triangle.
    txmin = wp.min(p0[0], wp.min(p1[0], p2[0]))
    txmax = wp.max(p0[0], wp.max(p1[0], p2[0]))
    tymin = wp.min(p0[1], wp.min(p1[1], p2[1]))
    tymax = wp.max(p0[1], wp.max(p1[1], p2[1]))

    cx0 = int(wp.floor((txmin - origin_x) / cell_size))
    cx1 = int(wp.floor((txmax - origin_x) / cell_size))
    cy0 = int(wp.floor((tymin - origin_y) / cell_size))
    cy1 = int(wp.floor((tymax - origin_y) / cell_size))
    if cx0 < 0:
        cx0 = int(0)
    if cy0 < 0:
        cy0 = int(0)
    if cx1 > nx - 1:
        cx1 = nx - 1
    if cy1 > ny - 1:
        cy1 = ny - 1

    obj = face_object[t]
    # Retain the direction of a face's outward normal as provenance for exact
    # coplanar contacts.  A floor top faces up; a crate underside faces down.
    # Their z intervals are both [z, z], so height alone cannot tell which one
    # bounds the crate cavity.  This is deliberately geometry data, never an
    # object-id/traversal-order tie-break.
    top_down = int(0)
    if n[2] < 0.0:
        top_down = int(1)

    ix = cx0
    while ix <= cx1:
        bx0 = origin_x + wp.float32(ix) * cell_size
        bx1 = bx0 + cell_size
        iy = cy0
        while iy <= cy1:
            by0 = origin_y + wp.float32(iy) * cell_size
            by1 = by0 + cell_size
            if _tri_box_overlap_2d(p0[0], p0[1], p1[0], p1[1], p2[0], p2[1],
                                   bx0, by0, bx1, by1) == 1:
                # Whole-triangle z-range: the fallback for a near-vertical face,
                # where the plane cannot be evaluated in z, and for a FLATTEN ramp.
                z_lo = tzmin
                z_hi = tzmax
                steep = int(0)
                if terrain == 1:
                    steep = int(1)
                if clip == 1:
                    # Recast's `rcRasterizeTriangles` records, per column, the z range
                    # of the triangle CLIPPED to that column — not the whole triangle's
                    # range. Without it a tilted face is marked solid up to its own peak
                    # in every cell it touches, so a 14-degree ramp reports a flat
                    # surface at its high end and an object placed at the low end floats
                    # by the triangle's entire z extent. The error is unbounded: it is
                    # the size of the triangle, not of the cell.
                    #
                    # The four cell corners bound the plane over the cell (it is linear),
                    # and clamping to the triangle's own range keeps a face that only
                    # partly covers the cell from reaching outside itself. A horizontal
                    # deck is unchanged — all four corners give the same z — so
                    # axis-aligned geometry rasterises exactly as before.
                    inv = 1.0 / n[2]
                    q00 = p0[2] - (n[0] * (bx0 - p0[0]) + n[1] * (by0 - p0[1])) * inv
                    q10 = p0[2] - (n[0] * (bx1 - p0[0]) + n[1] * (by0 - p0[1])) * inv
                    q01 = p0[2] - (n[0] * (bx0 - p0[0]) + n[1] * (by1 - p0[1])) * inv
                    q11 = p0[2] - (n[0] * (bx1 - p0[0]) + n[1] * (by1 - p0[1])) * inv
                    z_lo = wp.clamp(wp.min(q00, wp.min(q10, wp.min(q01, q11))), tzmin, tzmax)
                    z_hi = wp.clamp(wp.max(q00, wp.max(q10, wp.max(q01, q11))), tzmin, tzmax)
                cell = ix * ny + iy
                slot = wp.atomic_add(solid_cnt, cell, 1)
                if slot < MAX_SOLID:
                    base = cell * MAX_SOLID + slot
                    solid_min[base] = z_lo
                    solid_max[base] = z_hi
                    solid_obj[base] = obj
                    solid_obj_top[base] = obj
                    solid_top_down[base] = top_down
                    solid_cavity_owner[base] = obj if top_down == 1 else -1
                    solid_steep[base] = steep
            iy += 1
        ix += 1


@wp.kernel
def invert_free_kernel(
    solid_min: wp.array(dtype=wp.float32),
    solid_max: wp.array(dtype=wp.float32),
    solid_obj: wp.array(dtype=wp.int32),
    solid_obj_top: wp.array(dtype=wp.int32),
    solid_top_down: wp.array(dtype=wp.int32),
    solid_cavity_owner: wp.array(dtype=wp.int32),
    solid_steep: wp.array(dtype=wp.int32),
    solid_cnt: wp.array(dtype=wp.int32),
    max_z: wp.float32,
    merge_gap: wp.float32,
    min_free: wp.float32,
    exclude_steep: wp.int32,
    span_min: wp.array(dtype=wp.float32),
    span_max: wp.array(dtype=wp.float32),
    span_count: wp.array(dtype=wp.int32),
    span_wanted: wp.array(dtype=wp.int32),
):
    cell = wp.tid()
    n = solid_cnt[cell]
    sbase = cell * MAX_SOLID
    fbase = cell * MAX_FREE

    if n == 0:
        # An empty column gets NO free span, and that is deliberate for this consumer.
        # The column really is free, but a free span here means "a surface you could
        # rest on, and the clearance above it" — an empty column has no surface. Making
        # it emit [min_z, max_z] instead (tried, reverted) floors every empty cell at
        # the scope bottom; they then share a floor, the region grouper joins them, and
        # `space support` reports a large phantom resting surface hanging at the bottom
        # of the scope wherever the scene is simply empty.
        #
        # Bridging a slat gap is therefore NOT this function's job: it must synthesize
        # the span at the height the NEIGHBOURS agree on, never at the scope floor.
        # See `usd_core.space.gapfill`, which runs before the smoothing pass.
        span_count[cell] = 0
        return

    # True overflow (more raw spans than the buffer held): some solids were
    # dropped, so we can't trust the layering. Be conservative — treat the cell as
    # fully solid (no free spans) rather than invent free space from a corrupted
    # partial list. This only fires at dense structure cores (uprights / brace
    # intersections), which are solid anyway.
    if n > MAX_SOLID:
        span_count[cell] = 0
        return

    # Insertion-sort this cell's solid spans by z_lo (n <= MAX_SOLID, cheap).
    i = int(1)
    while i < n:
        kmin = solid_min[sbase + i]
        kmax = solid_max[sbase + i]
        kobj = solid_obj[sbase + i]
        kobjt = solid_obj_top[sbase + i]
        kdown = solid_top_down[sbase + i]
        kcavity = solid_cavity_owner[sbase + i]
        kstp = solid_steep[sbase + i]
        j = i - 1
        # Total order on (z_lo, z_max, object), not just z_lo. Slots are handed out by
        # `wp.atomic_add`, so coincident spans — a crate's underside and the floor it
        # stands on, at the same z — arrive in thread order, which differs between CPU
        # and CUDA. Sorting on z_lo alone leaves their relative order to that race, and
        # the merged layer's face ownership then depends on the device. The tie-breakers
        # make the sorted column a function of the geometry alone.
        while j >= 0 and (
            solid_min[sbase + j] > kmin
            or (solid_min[sbase + j] == kmin and solid_max[sbase + j] > kmax)
            or (solid_min[sbase + j] == kmin and solid_max[sbase + j] == kmax
                and solid_obj[sbase + j] > kobj)
        ):
            solid_min[sbase + j + 1] = solid_min[sbase + j]
            solid_max[sbase + j + 1] = solid_max[sbase + j]
            solid_obj[sbase + j + 1] = solid_obj[sbase + j]
            solid_obj_top[sbase + j + 1] = solid_obj_top[sbase + j]
            solid_top_down[sbase + j + 1] = solid_top_down[sbase + j]
            solid_cavity_owner[sbase + j + 1] = solid_cavity_owner[sbase + j]
            solid_steep[sbase + j + 1] = solid_steep[sbase + j]
            j -= 1
        solid_min[sbase + j + 1] = kmin
        solid_max[sbase + j + 1] = kmax
        solid_obj[sbase + j + 1] = kobj
        solid_obj_top[sbase + j + 1] = kobjt
        solid_top_down[sbase + j + 1] = kdown
        solid_cavity_owner[sbase + j + 1] = kcavity
        solid_steep[sbase + j + 1] = kstp
        i += 1

    # Merge overlapping/adjacent solid spans in place (compact heightfield). A merged
    # layer keeps the owner of its BOTTOM face in `solid_obj` and of its TOP face in
    # `solid_obj_top`, because those are the two the cavity test asks about — is the
    # surface under this gap the same object as the surface over it — and a merged
    # layer's interior owner is not a question anyone has.
    #
    # Collapsing both to `-1` for a mixed layer, as this used to, breaks that test on
    # the most ordinary input there is: an object RESTING on something is coplanar with
    # it by definition, so a crate on a floor merges, the merged layer reports no owner,
    # and the crate's sealed interior comes back as placeable. The steep flag already
    # tracked the top face for the same reason — this is the same rule for ownership.
    w = int(0)
    r = int(1)
    while r < n:
        if solid_min[sbase + r] <= solid_max[sbase + w] + merge_gap:
            # At an exact coplanar contact, choose the down-facing boundary as
            # the upper-face owner.  A floor top and crate underside have the
            # same degenerate z interval, so neither z nor object id can identify
            # the cavity boundary.  The face normal can: the crate underside faces
            # down.  This remains correct if stage traversal reverses their ids.
            contact = (solid_obj[sbase + r] != solid_obj_top[sbase + w]
                       and wp.abs(solid_min[sbase + r] - solid_max[sbase + w])
                           <= 1.0e-6)
            replace_top = (
                solid_max[sbase + r] > solid_max[sbase + w]
                or (solid_max[sbase + r] == solid_max[sbase + w]
                    and solid_min[sbase + r] > solid_min[sbase + w])
            )
            extends_top = solid_max[sbase + r] > solid_max[sbase + w]
            if contact == 1 and solid_top_down[sbase + r] != solid_top_down[sbase + w]:
                replace_top = solid_top_down[sbase + r] == 1
                # The up-facing mate of this contact is the layer's lower
                # boundary.  Preserve it as `solid_obj` even if an id-sorted
                # down-facing crate underside arrived first.  The pair then
                # represents floor -> crate rather than crate -> floor and
                # closes both the floor slab and the crate cavity.
                if solid_top_down[sbase + r] == 0:
                    solid_obj[sbase + w] = solid_obj[sbase + r]
            # Keep the down-facing contact owner separately from the merged
            # layer's ordinary top owner.  At one exact z there can be a floor
            # bottom, floor top, and crate bottom; the up-facing floor top may
            # legitimately be visited after the crate bottom.  It must not erase
            # the crate boundary that seals the cavity above it.
            if contact == 1 and solid_top_down[sbase + r] == 1:
                solid_cavity_owner[sbase + w] = solid_obj[sbase + r]
            if replace_top:
                solid_max[sbase + w] = solid_max[sbase + r]
                solid_steep[sbase + w] = solid_steep[sbase + r]
                solid_obj_top[sbase + w] = solid_obj_top[sbase + r]
                solid_top_down[sbase + w] = solid_top_down[sbase + r]
                if extends_top:
                    # A finite-thickness shell's upper face is its cavity
                    # boundary too: a 0.5 mm crate floor ends on an up-facing
                    # face, yet the air above it is still the same crate's
                    # sealed interior.
                    solid_cavity_owner[sbase + w] = solid_obj_top[sbase + r]
                elif solid_top_down[sbase + r] == 1:
                    solid_cavity_owner[sbase + w] = solid_obj[sbase + r]
        else:
            w += 1
            solid_min[sbase + w] = solid_min[sbase + r]
            solid_max[sbase + w] = solid_max[sbase + r]
            solid_obj[sbase + w] = solid_obj[sbase + r]
            solid_obj_top[sbase + w] = solid_obj_top[sbase + r]
            solid_top_down[sbase + w] = solid_top_down[sbase + r]
            solid_cavity_owner[sbase + w] = solid_cavity_owner[sbase + r]
            solid_steep[sbase + w] = solid_steep[sbase + r]
        r += 1
    m = w + 1  # number of merged solid layers

    # Free spans = gaps ABOVE each solid layer (rest on a real support surface).
    fc = int(0)
    want = int(0)
    k = int(0)
    while k < m:
        floor = solid_max[sbase + k]
        ceil = max_z
        skip = int(0)
        if k + 1 < m:
            ceil = solid_min[sbase + k + 1]
            # Hollow-container skip (original space_detect_helper.py:387): a gap
            # bounded above and below by the SAME object is that object's own
            # interior, not placeable free space. The comparison is between the
            # down-facing contact face of the layer below and the BOTTOM face of the
            # layer above — the two surfaces that actually bound this gap. Without it the detector offers
            # the inside of a sealed box: on the shipped `simready/Cardbox`, a
            # query for a 5 cm object inside the closed carton returns 1 region
            # with this test removed and 0 with it in place.
            #
            # Object identity is the right granularity *because of how containers
            # are authored*. A rack, a cabinet, a shelving unit is a set of part
            # prims — uprights, panels, decks — which is why `--container` takes a
            # comma-separated component list at all; the air between two parts
            # carries two different ids and is kept. What the test excludes is the
            # cavity of a single closed object, which is exactly the thing you
            # cannot put anything into.
            #
            # `solid_obj_top` handles ordinary finite-thickness shell faces.  The
            # separate down-facing contact owner handles an exact floor/crate
            # contact, where the coplanar floor top must not overwrite the crate
            # underside that seals its cavity.
            same_top_owner = (solid_obj_top[sbase + k] >= 0
                              and solid_obj_top[sbase + k] == solid_obj[sbase + k + 1])
            same_contact_owner = (solid_cavity_owner[sbase + k] >= 0
                                  and solid_cavity_owner[sbase + k]
                                      == solid_obj[sbase + k + 1])
            if same_top_owner == 1 or same_contact_owner == 1:
                skip = int(1)
        # TERRAIN exclusion: a free span resting on a STEEP layer (a ramp surface)
        # is not a valid rest surface — drop it so nothing is placed on the
        # incline, while the layers above/below (real decks) still qualify.
        if exclude_steep == 1 and solid_steep[sbase + k] == 1:
            skip = int(1)
        if skip == 0 and (ceil - floor) > min_free:
            # `want` counts every usable gap this column HAS; `fc` counts the ones that
            # fit in the buffer. They differ on a column with more than MAX_FREE levels
            # — a rack with seventeen shelves — and the difference is upper levels that
            # simply vanish from the answer. Unlike the solid-layer overflow this one is
            # not fail-safe: it removes real regions rather than adding phantom solid,
            # so it has to be reported.
            want += 1
            if fc < MAX_FREE:
                span_min[fbase + fc] = floor
                span_max[fbase + fc] = ceil
                fc += 1
        k += 1
    span_count[cell] = fc
    span_wanted[cell] = want


def build_spans_raster(mesh, scope_min, scope_max, cell_size,
                       cell_height_threshold, merge_gap=DEFAULT_MERGE_GAP, min_free=None,
                       slope_mode="terrain", slope_threshold_deg=20.0,
                       vertical_cutoff_deg=80.0):
    """Rasterise the scene into a multi-layer solid heightfield, return free spans.

    Drop-in replacement for ESD's ``build_spans``: same output dict
    (``nx, ny, max_spans, origin, cell_size, min_z, max_z, span_min, span_max,
    span_count``) so ``smooth`` / ``extract`` consume it unchanged. Reuses the
    mesh's resident device geometry (points / indices / face->object).

    Free intervals thinner than ``min_free`` are not emitted; it defaults to
    ``cell_height_threshold`` (the object height), matching the original
    ESD's sampler, which dropped spans shorter than its step. That also stops a thin
    solid slab's own interior (the gap between its bottom and top faces) from
    consuming a free-span slot — those are always sub-object-height.

    Slope handling (``slope_mode`` ∈ ``{"ignore","flatten","terrain"}``, default
    ``"terrain"``): a face tilting between ``slope_threshold_deg`` and
    ``vertical_cutoff_deg`` from horizontal is a ramp. Without this, the
    whole-triangle-z-range slab writes a ramp into every covered cell as a solid
    from its lowest to its highest vertex — reporting the incline as a flat wall at
    its peak and destroying the free space beside it. See the ``SLOPE_*`` notes.
    Near-horizontal decks and near-vertical walls are unaffected by the mode.
    """
    if slope_mode not in _SLOPE_MODES:
        raise ValueError(
            f"slope_mode must be one of {sorted(_SLOPE_MODES)}, got {slope_mode!r}")
    smode = _SLOPE_MODES[slope_mode]
    # cos of the inclination thresholds (cos_tilt = |n_z|/|n|): a face is a ramp
    # when cos_wall < cos_tilt < cos_flat.
    cos_flat = float(np.cos(np.radians(float(slope_threshold_deg))))
    cos_wall = float(np.cos(np.radians(float(vertical_cutoff_deg))))
    exclude_steep = 1 if smode == _SLOPE_MODES["terrain"] else 0
    if min_free is None:
        min_free = float(cell_height_threshold)
    device = mesh.device
    ox, oy = float(scope_min[0]), float(scope_min[1])
    min_z, max_z = float(scope_min[2]), float(scope_max[2])
    # `ceil`, not `round`: the grid must COVER the scope. Rounding down drops the
    # remainder — a 1.04 m span at cell=0.1 became 10 cells covering 1.00 m, so the
    # last 4 cm was never rasterised and any surface there silently vanished. Real
    # assets are rarely an integer number of cells wide (the shipped pallet is
    # 1.213 m), and `support.py`'s own DoS estimate already used `ceil`, so the
    # guard and the allocation disagreed. The epsilon keeps an exact multiple from
    # gaining a phantom trailing cell.
    nx = max(1, int(math.ceil((scope_max[0] - scope_min[0]) / cell_size - 1e-9)))
    ny = max(1, int(math.ceil((scope_max[1] - scope_min[1]) / cell_size - 1e-9)))
    n_cells = nx * ny
    n_tris = int(mesh._face_object.shape[0])

    ms = int(MAX_SOLID)
    solid_min = wp.zeros(n_cells * ms, dtype=wp.float32, device=device)
    solid_max = wp.zeros(n_cells * ms, dtype=wp.float32, device=device)
    solid_obj = wp.full(n_cells * ms, -1, dtype=wp.int32, device=device)
    solid_obj_top = wp.full(n_cells * ms, -1, dtype=wp.int32, device=device)
    solid_top_down = wp.zeros(n_cells * ms, dtype=wp.int32, device=device)
    solid_cavity_owner = wp.full(n_cells * ms, -1, dtype=wp.int32, device=device)
    solid_steep = wp.zeros(n_cells * ms, dtype=wp.int32, device=device)
    solid_cnt = wp.zeros(n_cells, dtype=wp.int32, device=device)

    wp.launch(
        rasterize_solid_kernel,
        dim=n_tris,
        inputs=[
            mesh._points, mesh._indices, mesh._face_object,
            wp.float32(ox), wp.float32(oy), wp.float32(cell_size),
            wp.int32(nx), wp.int32(ny), wp.float32(min_z), wp.float32(max_z),
            wp.int32(smode), wp.float32(cos_flat), wp.float32(cos_wall),
        ],
        outputs=[solid_min, solid_max, solid_obj, solid_obj_top, solid_top_down,
                 solid_cavity_owner, solid_steep,
                 solid_cnt],
        device=device,
    )

    mf = int(MAX_FREE)
    span_min = wp.zeros(n_cells * mf, dtype=wp.float32, device=device)
    span_max = wp.zeros(n_cells * mf, dtype=wp.float32, device=device)
    span_count = wp.zeros(n_cells, dtype=wp.int32, device=device)
    span_wanted = wp.zeros(n_cells, dtype=wp.int32, device=device)

    wp.launch(
        invert_free_kernel,
        dim=n_cells,
        inputs=[
            solid_min, solid_max, solid_obj, solid_obj_top, solid_top_down,
            solid_cavity_owner, solid_steep,
            solid_cnt,
            # -eps so a span exactly == object height survives (matches extract's
            # _HEIGHT_EPS), while sub-object-height interior gaps are dropped.
            wp.float32(max_z), wp.float32(merge_gap), wp.float32(min_free - 1.0e-6),
            wp.int32(exclude_steep),
        ],
        outputs=[span_min, span_max, span_count, span_wanted],
        device=device,
    )

    overflow = int((solid_cnt.numpy() > ms).sum())
    free_overflow = int((span_wanted.numpy() > mf).sum())
    return {
        "nx": nx, "ny": ny, "max_spans": mf,
        "origin": (ox, oy), "cell_size": float(cell_size),
        "min_z": min_z, "max_z": max_z,
        "span_min": span_min.numpy().reshape(nx, ny, mf),
        "span_max": span_max.numpy().reshape(nx, ny, mf),
        "span_count": span_count.numpy().reshape(nx, ny),
        # Per-cell SOLID layer count. Without it `gapfill` cannot tell the two very
        # different reasons a cell has zero free spans apart: the column is entirely
        # solid (inside a block — nothing may be placed there), or entirely EMPTY (free,
        # but with no surface, so the "gaps above each solid layer" construction emits
        # nothing). Bridging the second is correct; bridging the first would report a
        # resting place inside geometry.
        "solid_count": solid_cnt.numpy().reshape(nx, ny),
        "solid_overflow_cells": overflow,
        # Cells whose usable free levels outnumbered `MAX_FREE`. The extra levels are
        # the HIGHEST ones and they are gone from the answer, so this is a loss the
        # caller has to hear about.
        "free_overflow_cells": free_overflow,
    }
