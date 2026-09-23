import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from v2_geometry import surface_compare, nearest_distance_mm
from v2_policy import auxiliary_checks
from v2_process import run, inspect_with_deadline, InspectionDeadline
from v2_control import clamp_effort


@pytest.mark.parametrize('edge', [.0001, .0002, .0005, .001, .01])
@pytest.mark.parametrize('origin', [[0, 0, 0], [.1, .2, -.3], [1, -2, 3]])
@pytest.mark.parametrize('offset,passes', [(0, True), (.000025, True), (.000075, False)])
def test_analytic_plane_distance(edge, origin, offset, passes):
    # Source triangle, submitted four-triangle subdivision: fast path unavailable.
    # Distances are analytically exactly the normal translation, at every point.
    a = np.array([[0, 0, 0], [edge, 0, 0], [0, edge, 0]]) + origin
    b = np.vstack([a, (a[0] + a[1]) / 2, (a[1] + a[2]) / 2, (a[2] + a[0]) / 2])
    b[:, 2] += offset
    ok, detail = surface_compare(a, [[0, 1, 2]], b, [[0, 3, 5], [3, 1, 4], [5, 4, 2], [3, 4, 5]], .00005)
    assert ok is passes
    if passes:
        assert detail['method'].endswith('recentered_mm')
        # One-nanometre calibration accuracy is 50,000 times tighter than the
        # physical gate; the retained initial run used an unnecessary 1pm goal.
        assert detail['source_to_final_max_sample_distance_m'] == pytest.approx(offset, abs=1e-9)
        assert detail['final_to_source_max_sample_distance_m'] == pytest.approx(offset, abs=1e-9)


def test_surface_certificate_is_all_points_not_a_sample():
    a = trimesh.creation.icosphere(subdivisions=2, radius=.01)
    b = a.vertices.copy(); b[73] += [0, 0, .00004]
    ok, d = surface_compare(a.vertices, a.faces, b, a.faces, .00005)
    assert ok and d['correspondence_is_conservative']
    assert d['full_surface_distance_upper_bound_m'] == pytest.approx(.00004)
    b[73] += [0, 0, .002]
    assert not surface_compare(a.vertices, a.faces, b, a.faces, .00005)[0]


def test_reindexing_and_winding_are_not_geometry_loss():
    a = trimesh.creation.box(extents=[.01, .02, .03])
    order = np.random.default_rng(73).permutation(len(a.vertices))
    inverse = np.argsort(order)
    faces = inverse[a.faces][::-1, ::-1]
    ok, d = surface_compare(a.vertices, a.faces, a.vertices[order], faces, 5e-5)
    assert ok and d['full_surface_distance_upper_bound_m'] == 0


def test_duplicate_area_and_missing_thin_geometry_fail():
    a = trimesh.creation.box(extents=[.01, .02, .03])
    assert not surface_compare(a.vertices, a.faces, a.vertices, np.tile(a.faces, (4, 1)), 5e-5)[0]
    v = np.vstack([a.vertices, [0, 0, .025], [.0001, 0, .025], [0, .0001, .025]])
    f = np.vstack([a.faces, [8, 9, 10]])
    assert not surface_compare(v, f, a.vertices, a.faces, 5e-5)[0]


def test_unchanged_loose_source_point_is_preserved_but_cannot_be_deleted():
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.2, .2, 1]])
    ok, d = surface_compare(a, [[0, 1, 2]], a, [[0, 1, 2]], 5e-5)
    assert ok and d['full_surface_distance_upper_bound_m'] == 0
    assert not surface_compare(a, [[0, 1, 2]], a[:3], [[0, 1, 2]], 5e-5)[0]
    altered = a.copy(); altered[-1, 2] += .01
    assert not surface_compare(a, [[0, 1, 2]], altered, [[0, 1, 2]], 5e-5)[0]
    extra = np.vstack([a, [.8, .2, .4]]) # Same bounds, new off-surface point.
    assert not surface_compare(a, [[0, 1, 2]], extra, [[0, 1, 2]], 5e-5)[0]
    reordered = a[[3, 2, 0, 1]]
    ok, detail = surface_compare(a, [[0, 1, 2]], reordered, [[2, 3, 1]], 5e-5)
    assert ok and detail['full_surface_distance_upper_bound_m'] == 0


def test_same_bounds_area_but_shifted_interior_is_rejected():
    # Large unchanged square carries extrema; a tiny interior patch moves 1mm
    # vertically while keeping its own area and assembly bounds unchanged.
    base = trimesh.creation.box(extents=[.1, .1, .1])
    patch = trimesh.creation.box(extents=[.001, .001, .001])
    a = trimesh.util.concatenate([base, patch])
    b = a.vertices.copy(); b[len(base.vertices):, 0] += .01
    ok, d = surface_compare(a.vertices, a.faces, b, a.faces, 5e-5)
    assert not ok and d['bounds_error_m'] == 0 and d['area_ratio'] == pytest.approx(1)


def _auxiliary_fixture():
    bindings = {'bodies': [{'path': '/base', 'role': 'base', 'moving': False},
                           {'path': '/crank', 'role': 'crank', 'moving': True},
                           {'path': '/slot', 'role': 'slot', 'moving': True, 'auxiliary': 'passive_constraint'}]}
    bodies = {b['path']: b for b in bindings['bodies']}
    joints = {'a': {'body0': '/base', 'body1': '/slot', 'role': 'slot_slide'},
              'b': {'body0': '/slot', 'body1': '/crank', 'role': 'slot_pin'}}
    mapped = [{'body_path': '/crank'}]
    contract = {'required_body_roles': ['base', 'crank'], 'allow_passive_auxiliary_bodies': True, 'control': {'crank': {}}}
    return bindings, bodies, joints, mapped, contract


@pytest.mark.parametrize('negative', [None, 'undeclared', 'actuated', 'disconnected', 'required_role', 'other_case'])
def test_explicit_auxiliary_contract(negative):
    b, bodies, j, m, c = _auxiliary_fixture()
    if negative == 'undeclared': del b['bodies'][2]['auxiliary']
    if negative == 'actuated': j['a']['role'] = 'crank'
    if negative == 'disconnected': del j['a']
    if negative == 'required_role': bodies['/slot']['role'] = 'crank'
    if negative == 'other_case': c['allow_passive_auxiliary_bodies'] = False
    records = auxiliary_checks(b, bodies, j, m, c)
    assert all(r[1] for r in records) is (negative is None)


def test_timeout_returns_receipt_without_aborting_evaluator(tmp_path):
    with (tmp_path / 'worker.log').open('w') as out:
        result = run([sys.executable, '-c', 'import time; time.sleep(10)'], stdout=out, timeout=.05)
    receipt = json.loads((tmp_path / 'worker.log.process.json').read_text())
    assert result.timed_out and receipt['disposition'] == 'INCONCLUSIVE'
    assert receipt['elapsed_s'] < 3


def test_inspection_timeout_is_distinct_and_restores_alarm():
    with pytest.raises(InspectionDeadline):
        inspect_with_deadline(lambda: time.sleep(10), timeout_s=.02)
    assert inspect_with_deadline(lambda: 17, timeout_s=.05) == 17


def test_corrupt_geometry_does_not_pass():
    a = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    for v, f in [(a * np.nan, [[0, 1, 2]]), (a, [[0, 1, 99]]), (a, [[0., 1., 2.]]), (a, [])]:
        with pytest.raises(ValueError):
            surface_compare(a, [[0, 1, 2]], v, f, 5e-5)


@pytest.mark.parametrize('cap', [.001,.2,2.,30.])
def test_seeded_effort_never_exceeds_public_cap(cap):
    for seed in [11,23,47,83,131]:
        rng=np.random.default_rng(seed);jitter=rng.uniform(.98,1.02)
        for raw in np.linspace(-3*cap,3*cap,137):
            previous=float(np.clip(raw,-cap,cap))*jitter
            actual=clamp_effort(previous,cap)
            assert abs(actual)<=cap
            if abs(previous)<=cap:assert actual==previous
            else:assert actual==np.sign(previous)*cap


def test_candidate_index_miss_uses_exact_fallback():
    vertices=[[0.,0,0],[1000.,0,0],[0,1000.,0]];faces=[[0,1,2]]
    for z in [1.,2.,3.,4.]:
        i=len(vertices);vertices.extend([[100.01,100.01,z],[100.02,100.01,z],[100.01,100.02,z]]);faces.append([i,i+1,i+2])
    mesh=trimesh.Trimesh(vertices,faces,process=False);statistics={}
    d=nearest_distance_mm(np.array([[100.,100.,0]]),mesh,tolerance_mm=.05,statistics=statistics)
    assert d==pytest.approx(0,abs=1e-9)
    assert statistics['exact_fallback_points']==1


def test_retriangulation_preserves_original_loose_points():
    a=np.array([[0.,0,0],[1,0,0],[0,1,0],[.2,.2,1]])
    b=np.vstack([a,[.5,0,0],[.5,.5,0],[0,.5,0]])
    ok,d=surface_compare(a,[[0,1,2]],b,[[0,4,6],[4,1,5],[6,5,2],[4,5,6]],5e-5)
    assert ok and d['source_loose_point_count']==d['final_loose_point_count']==1
