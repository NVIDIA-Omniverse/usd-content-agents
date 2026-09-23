"""Source-fidelity measurement v2. No submission code is imported.

Corresponding triangles give a conservative bound for EVERY surface point, not
just samples: barycentric interpolation cannot exceed the largest paired vertex
distance. Changed topology uses the original deterministic symmetric sample gate,
with all vertices retained, in recentered millimetre coordinates.
"""
import hashlib
import time
import numpy as np
import trimesh


class DistanceExceeded(Exception):
    def __init__(self, point, tolerance, candidate_count, candidate_distance=None):
        self.evidence={'query_point_recentered_mm':np.asarray(point).tolist(),
                       'distance_strictly_greater_than_m':tolerance/1000.,
                       'complete_threshold_aabb_candidates':candidate_count,
                       'minimum_candidate_distance_m':None if candidate_distance is None else candidate_distance/1000.}


def triangle_distances(triangles, points):
    """Float64 point/triangle distances via plane and three clamped segments.

    Cross products avoid the Gram-determinant cancellation for very long, thin
    CAD triangles. Degenerate triangles reduce to their segments/points.
    """
    t=np.asarray(triangles,float);p=np.asarray(points,float)
    a,b,c=t[:,0],t[:,1],t[:,2];ab=b-a;ac=c-a
    n=np.cross(ab,ac);n2=np.einsum('ij,ij->i',n,n)
    dot=np.einsum('ij,ij->i',p-a,n)
    factor=np.divide(dot,n2,out=np.zeros_like(dot),where=n2>0)
    projected=p-factor[:,None]*n
    inside=n2>0
    for start,end in [(a,b),(b,c),(c,a)]:
        inside &= np.einsum('ij,ij->i',np.cross(end-start,projected-start),n)>=0
    best=np.full(len(p),np.inf)
    best[inside]=(dot[inside]**2)/n2[inside]
    for start,end in [(a,b),(b,c),(c,a)]:
        edge=end-start;length2=np.einsum('ij,ij->i',edge,edge)
        along=np.divide(np.einsum('ij,ij->i',p-start,edge),length2,out=np.zeros(len(p)),where=length2>0)
        closest=start+np.clip(along,0,1)[:,None]*edge
        best=np.minimum(best,np.einsum('ij,ij->i',p-closest,p-closest))
    return np.sqrt(np.maximum(best,0))


def _arrays(vertices, faces):
    v = np.asarray(vertices, dtype=np.float64)
    f0 = np.asarray(faces)
    if v.ndim != 2 or v.shape[1:] != (3,) or not len(v) or not np.isfinite(v).all():
        raise ValueError('Mesh vertices must be a finite nonempty Nx3 array')
    if f0.ndim != 2 or f0.shape[1:] != (3,) or not len(f0):
        raise ValueError('Mesh faces must be a nonempty Mx3 triangle array')
    if not np.issubdtype(f0.dtype, np.integer):
        raise ValueError('Face indices must be integers')
    f = np.asarray(f0, dtype=np.int64)
    if f.min() < 0 or f.max() >= len(v):
        raise ValueError('Face index is outside the vertex array')
    return v, f


def _canonical_triangles(v, f):
    t = v[f].copy()
    # Sorting only proposes a correspondence. The actual unrounded distances
    # must still prove it; unstable sorting can only cause the slower fallback.
    order = np.lexsort((t[:, :, 2], t[:, :, 1], t[:, :, 0]), axis=1)
    t = np.take_along_axis(t, order[:, :, None], axis=1)
    rows = t.reshape(len(t), 9)
    return rows[np.lexsort(rows.T[::-1])].reshape(-1, 3, 3)


def _correspondence_bound(av, af, bv, bf, tolerance):
    # This bound includes every point, including unreferenced CAD tessellation
    # points. An unchanged loose point is preserved geometry, not a point which
    # must somehow move onto its own source triangle surface (the v1 defect).
    if av.shape == bv.shape and np.array_equal(af, bf):
        d = float(np.linalg.norm(av - bv, axis=1).max())
        if d <= tolerance:
            return d, 'indexed_triangle_correspondence'
    if af.shape == bf.shape:
        a, b = _canonical_triangles(av, af), _canonical_triangles(bv, bf)
        d = float(np.linalg.norm(a - b, axis=2).max())
        # A surface certificate alone cannot excuse added/removed loose points.
        if len(np.unique(af)) != len(av) or len(np.unique(bf)) != len(bv):
            if av.shape != bv.shape:
                return None
            ap = av[np.lexsort(av.T[::-1])]
            bp = bv[np.lexsort(bv.T[::-1])]
            d = max(d, float(np.linalg.norm(ap - bp, axis=1).max()))
        if d <= tolerance:
            return d, 'unrounded_canonical_triangle_correspondence'
    return None


def _samples(mesh, rng):
    count = min(12000, max(2000, len(mesh.faces)))
    index = rng.choice(len(mesh.faces), size=count, p=mesh.area_faces / mesh.area)
    uv = rng.random((count, 2))
    uv[uv.sum(1) > 1] = 1 - uv[uv.sum(1) > 1]
    tri = mesh.triangles[index]
    sampled = tri[:, 0] + uv[:, 0, None] * (tri[:, 1] - tri[:, 0]) + uv[:, 1, None] * (tri[:, 2] - tri[:, 0])
    # Loose points are checked as a separate symmetric point set. Requiring
    # them to lie on their own triangle surface caused the printer counterexample.
    return np.vstack((mesh.vertices[np.unique(mesh.faces)], sampled))


def _loose_points(v, f):
    used = np.zeros(len(v), dtype=bool);used[f.reshape(-1)] = True
    return v[~used]


def _loose_compare(av, af, bv, bf, tolerance):
    a, b = _loose_points(av, af), _loose_points(bv, bf)
    record={'source_loose_point_count':len(a),'final_loose_point_count':len(b)}
    if len(a) != len(b):return False, record
    if not len(a):return True, record
    from scipy.spatial import cKDTree
    ab=float(cKDTree(b).query(a, workers=1)[0].max())
    ba=float(cKDTree(a).query(b, workers=1)[0].max())
    record.update(source_to_final_loose_point_distance_m=ab,final_to_source_loose_point_distance_m=ba)
    return max(ab,ba)<=tolerance, record


def nearest_distance_mm(points, mesh, *, tolerance_mm=None, statistics=None, timeout_s=120.0):
    """All query points, spatially indexed triangles, bounded batch allocation.

    Caller must recenter/scale both points and mesh identically. No truncation,
    subsampling, bounding-box substitute, or rounded-coordinate acceptance. A
    point within tolerance of a triangle must be in that triangle's AABB expanded
    by tolerance. That complete candidate set decides the threshold exactly,
    avoiding the huge nearest-vertex search boxes in Trimesh's general query.
    """
    deadline = time.monotonic() + timeout_s
    maximum = 0.0
    from scipy.spatial import cKDTree
    # Surface landmarks retain an owning triangle. Vertices/edge midpoints are
    # essential for long thin CAD faces whose centroids are far from their ends.
    triangles=mesh.triangles
    landmarks=np.concatenate([triangles[:,0],triangles[:,1],triangles[:,2],
                              (triangles[:,0]+triangles[:,1])*.5,
                              (triangles[:,1]+triangles[:,2])*.5,
                              (triangles[:,2]+triangles[:,0])*.5,
                              mesh.triangles_center])
    candidate_tree = cKDTree(landmarks) if tolerance_mm is not None else None
    certified = fallback_count = 0
    # Building the immutable tree once avoids repeated index construction.
    _ = mesh.triangles_tree
    for start in range(0, len(points), 1024):
        if time.monotonic() > deadline:
            raise TimeoutError('Nearest-triangle measurement exceeded its per-direction budget')
        batch = points[start:start + 1024]
        if candidate_tree is not None:
            # A nearby candidate triangle is enough for a conservative distance
            # upper bound. It need not be the nearest one. Uncertified points
            # still receive the exact spatial-index query; none are skipped.
            k = min(4, len(mesh.faces))
            indices = candidate_tree.query(batch, k=k, workers=1)[1].reshape(-1, k)%len(mesh.faces)
            triangles = mesh.triangles[indices].reshape(-1, 3, 3)
            query = np.repeat(batch, k, axis=0)
            distance = triangle_distances(triangles,query).reshape(-1,k).min(1)
            # 1nm guard is far below the50um gate; boundary points use fallback.
            known = np.isfinite(distance) & (distance <= tolerance_mm - 1e-6)
            certified += int(known.sum())
        else:
            distance = np.full(len(batch), np.inf);known = np.zeros(len(batch), dtype=bool)
        pending = np.flatnonzero(~known);fallback_count += len(pending)
        for selected in pending:
            if time.monotonic()>deadline:raise TimeoutError('Complete triangle threshold query exceeded budget')
            q=batch[selected]
            if tolerance_mm is None:raise ValueError('A positive threshold is required')
            candidates=np.asarray(list(mesh.triangles_tree.intersection(np.r_[q-tolerance_mm,q+tolerance_mm])),dtype=np.int64)
            if not len(candidates):raise DistanceExceeded(q,tolerance_mm,0)
            found=float(triangle_distances(mesh.triangles[candidates],np.repeat(q[None,:],len(candidates),axis=0)).min())
            if found>tolerance_mm:raise DistanceExceeded(q,tolerance_mm,len(candidates),found)
            distance[selected]=found
        if not np.isfinite(distance).all():
            raise ArithmeticError('Nearest-triangle query returned nonfinite distances')
        maximum = max(maximum, float(distance.max()))
    if statistics is not None:
        statistics.update(query_points=len(points),candidate_triangle_certificates=certified,exact_fallback_points=fallback_count)
    return maximum


def surface_compare(source_v, source_f, final_v, final_f, tolerance):
    start = time.monotonic()
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError('A positive finite distance tolerance is required')
    av, af = _arrays(source_v, source_f)
    bv, bf = _arrays(final_v, final_f)
    a = trimesh.Trimesh(av, af, process=False)
    b = trimesh.Trimesh(bv, bf, process=False)
    if not np.isfinite([a.area, b.area]).all() or min(a.area, b.area) <= 0:
        raise ValueError('Source and submitted mesh must have positive finite surface area')
    area_ratio = float(b.area / a.area)
    bounds_error = float(np.max(np.abs(a.bounds - b.bounds)))
    detail = {'measurement_version': 2, 'area_ratio': area_ratio,
              'bounds_error_m': bounds_error, 'tolerance_m': float(tolerance),
              'source_vertices': len(av), 'final_vertices': len(bv),
              'source_triangles': len(af), 'final_triangles': len(bf)}
    loose_ok, loose_detail = _loose_compare(av, af, bv, bf, tolerance)
    detail.update(loose_detail)
    if not loose_ok:
        detail.update(method='loose_point_count_or_distance_rejection',elapsed_s=time.monotonic()-start)
        return False, detail
    # These are already concrete failing criteria. Early rejection is exact.
    if abs(area_ratio - 1) > 0.03 or bounds_error > tolerance:
        detail.update(method='area_or_bounds_rejection', elapsed_s=time.monotonic() - start)
        return False, detail
    certificate = _correspondence_bound(av, af, bv, bf, tolerance)
    if certificate:
        bound, method = certificate
        detail.update(method=method, full_surface_distance_upper_bound_m=bound,
                      correspondence_is_conservative=True, elapsed_s=time.monotonic() - start)
        return True, detail
    # Fixed scale preserves the independently qualified v1.1 metric. Recentring
    # removes irrelevant world-origin magnitude without changing distances.
    origin = (a.bounds[0] + a.bounds[1]) * 0.5
    aa = trimesh.Trimesh((av - origin) * 1000., af, process=False)
    bb = trimesh.Trimesh((bv - origin) * 1000., bf, process=False)
    rng = np.random.default_rng(101)
    ab_stats, ba_stats = {}, {}
    direction='source_to_final'
    try:
        ab = nearest_distance_mm(_samples(aa, rng), bb, tolerance_mm=tolerance*1000, statistics=ab_stats) / 1000.
        direction='final_to_source'
        ba = nearest_distance_mm(_samples(bb, rng), aa, tolerance_mm=tolerance*1000, statistics=ba_stats) / 1000.
    except DistanceExceeded as exc:
        detail.update(method='complete_triangle_threshold_rejection',direction=direction,violation=exc.evidence,elapsed_s=time.monotonic()-start)
        return False, detail
    detail.update(source_to_final_max_sample_distance_m=ab,
                  final_to_source_max_sample_distance_m=ba,
                  method='all_vertices_and_seed101_area_samples_symmetric_nearest_triangles_recentered_mm',
                  distance_semantics='conservative candidate-triangle upper bounds; otherwise minimum over every triangle whose AABB can meet the threshold',
                  source_query_statistics=ab_stats, final_query_statistics=ba_stats,
                  sample_count_per_direction=[len(np.unique(af)) + min(12000, max(2000, len(af))), len(np.unique(bf)) + min(12000, max(2000, len(bf)))],
                  elapsed_s=time.monotonic() - start)
    return bool(max(ab, ba) <= tolerance), detail
