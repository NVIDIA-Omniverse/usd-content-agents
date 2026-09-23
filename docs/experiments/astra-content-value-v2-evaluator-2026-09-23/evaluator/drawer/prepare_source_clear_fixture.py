"""Version an unscored initialization correction from original source geometry.

No authored submission is read. Frozen evaluator/solver bytes and all physical
acceptance thresholds are preserved; only protocol identity and payload Y change.
"""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip(poly, axis, bound, greater):
    if not poly:
        return []
    result = []
    previous = poly[-1]
    prev_inside = previous[axis] >= bound if greater else previous[axis] <= bound
    for point in poly:
        inside = point[axis] >= bound if greater else point[axis] <= bound
        if inside != prev_inside:
            fraction = (bound - previous[axis]) / (point[axis] - previous[axis])
            result.append(previous + fraction * (point - previous))
        if inside:
            result.append(point)
        previous, prev_inside = point, inside
    return result


def clipped(triangle, planes):
    poly = list(triangle)
    for axis, bound, greater in planes:
        poly = clip(poly, axis, bound, greater)
        if not poly:
            break
    return poly


def qualify_clipping():
    square = [(0, -0.1, True), (0, 0.1, False),
              (2, -0.1, True), (2, 0.1, False)]
    flat = np.array([[-1., 1., -1.], [2., 1., -1.], [-1., 1., 2.]])
    plane = clipped(flat, square)
    sloped = np.array([[0., 0., 0.], [2., 2., 0.], [0., 0., 2.]])
    cut = clipped(sloped, [(0, .5, True), (0, 1., False), (2, 0., True), (2, .5, False)])
    vertical = np.array([[0., -2., -2.], [0., 3., -2.], [0., -2., 3.]])
    box = square + [(1, -.1, True), (1, .1, False)]
    checks = {
        'flat_plane_height_preserved': bool(plane) and all(abs(p[1]-1.) < 1e-12 for p in plane),
        'sloped_clipped_extremum': bool(cut) and abs(max(p[1] for p in cut)-1.) < 1e-12,
        'separated_plane_rejected': not clipped(flat, box),
        'crossing_triangle_detected_without_inside_vertices': bool(clipped(vertical, box)),
        'empty_input_stays_empty': clip([], 0, 0., True) == [],
    }
    assert all(checks.values()), checks
    return {key: bool(value) for key, value in checks.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-evaluator', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.input_evaluator.resolve()
    output = args.output.resolve()
    assert not output.exists(), 'Use a new versioned fixture directory'
    algorithm_checks = qualify_clipping()
    freeze = json.loads((source / 'drawer_freeze.json').read_text())
    for name, expected in freeze['files'].items():
        assert sha(source / name) == expected, name
    sys.path.insert(0, str(source))
    import drawer_evaluate as original
    original_spec = copy.deepcopy(original.SPEC)
    meshes = original.gltf_reference()
    payload = original_spec['payload']
    assert payload['size_m'][0] == payload['size_m'][2]
    assert 0 <= payload['initial_yaw_jitter_deg'] <= 45
    angle = math.radians(payload['initial_yaw_jitter_deg'])
    half = payload['size_m'][0] / 2 * (math.cos(angle) + math.sin(angle))
    x_extent = half + payload['initial_x_jitter_m']
    z_extent = half + payload['initial_z_jitter_m']
    x, _, z = payload['initial_center_m']
    xz_planes = [(0, x - x_extent, True), (0, x + x_extent, False),
                 (2, z - z_extent, True), (2, z + z_extent, False)]
    highest = []
    for triangle in meshes['drawer_cabinet_drawer_01']:
        poly = clipped(triangle, xz_planes)
        if poly:
            highest.append(max(float(v[1]) for v in poly))
    assert highest
    floor_bound = max(highest)
    clearance = 0.010
    new_y = floor_bound + payload['size_m'][1] / 2 + clearance
    assert payload['retained_center_min_relative_to_drawer_translation_m'][1] < new_y
    assert new_y < payload['retained_center_max_relative_to_drawer_translation_m'][1]
    lower_y = new_y - payload['size_m'][1] / 2
    upper_y = new_y + payload['size_m'][1] / 2
    box_planes = xz_planes + [(1, lower_y, True), (1, upper_y, False)]
    intersections = {}
    for name, triangles in meshes.items():
        intersections[name] = sum(bool(clipped(triangle, box_planes)) for triangle in triangles)
    assert not any(intersections.values()), intersections
    spec = copy.deepcopy(original_spec)
    spec['protocol_id'] = 'astra-drawer-unscored-source-clear-v1'
    spec['payload']['initial_center_m'][1] = new_y
    restored = copy.deepcopy(spec)
    restored['protocol_id'] = original_spec['protocol_id']
    restored['payload']['initial_center_m'][1] = payload['initial_center_m'][1]
    assert restored == original_spec
    output.mkdir(parents=True)
    for name in freeze['files']:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (output / 'drawer_acceptance.json').write_text(json.dumps(spec, indent=2) + '\n')
    shutil.copyfile(source / 'drawer_freeze.json', output / 'original_drawer_freeze.json')
    shutil.copyfile(Path(__file__), output / 'prepare_source_clear_fixture.py')
    record = {
        'version': spec['protocol_id'],
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope': 'Unscored capstone initialization correction. No author artifact read or modified; not a scored-pilot correction or acceptance result.',
        'original_spec_sha256': freeze['files']['drawer_acceptance.json'],
        'original_payload_center_m': payload['initial_center_m'],
        'new_payload_center_m': spec['payload']['initial_center_m'],
        'source_surface_upper_bound_within_swept_payload_xz_m': floor_bound,
        'clearance_above_that_bound_m': clearance,
        'swept_xz_bounds_m': [[x-x_extent, z-z_extent], [x+x_extent, z+z_extent]],
        'method': 'Clip every original upper-drawer triangle to a conservative XZ bound covering all declared initial jitter and yaw. Place the lower payload face 10mm above the maximum clipped source height. Clip all five original source meshes against the entire swept initial payload box and require no surface intersection.',
        'original_source_triangle_intersections_with_new_swept_payload': intersections,
        'changed_spec_fields': ['protocol_id', 'payload.initial_center_m[1]'],
        'all_other_spec_fields_identical': restored == original_spec,
        'clipping_qualification': algorithm_checks,
        'evaluator_and_solver_bytes_unchanged': True,
        'native_cooked_geometry_preflight_still_required': True,
        'five_native_trials_still_required': True,
        'files_sha256': {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()},
    }
    for name, expected in freeze['files'].items():
        assert sha(source / name) == expected
        if name != 'drawer_acceptance.json':
            assert sha(output / name) == expected
    (output / 'source_clear_freeze.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'manifest_sha256': sha(output / 'source_clear_freeze.json'),
                      'new_payload_y_m': new_y, 'source_intersections': intersections}))


if __name__ == '__main__':
    main()
