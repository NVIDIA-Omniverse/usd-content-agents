"""Compare a measured deepest contact with moved original render surfaces.

Read-only distance diagnosis, not a volume-overlap or collider identity proof.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluator', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--trace-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    assert sha(a.trace) == a.trace_sha256
    inputs = [a.trace, a.evaluator / 'drawer_evaluate.py',
              a.evaluator / 'reference/drawer_cabinet_1k.gltf',
              a.evaluator / 'reference/drawer_cabinet.bin']
    digests = {str(x): sha(x) for x in inputs}
    sys.path.insert(0, str(a.evaluator.resolve()))
    import drawer_evaluate as evaluator
    rows = [json.loads(line) for line in a.trace.read_text().splitlines()]
    row, contact = min(((row, c) for row in rows for c in row['contacts']),
                       key=lambda item: item[1]['separation'])
    position = np.array(contact['p'], dtype=float)
    normal = np.array(contact['normal'], dtype=float)
    separation = float(contact['separation'])
    pose = row['drawer_pose']
    matrix = Rotation.from_quat(pose[3:7]).as_matrix()
    points = {'reported_contact_position': position,
              'position_plus_signed_separation_times_normal': position + separation * normal,
              'position_minus_signed_separation_times_normal': position - separation * normal}
    distances = {}
    for name, triangles in evaluator.gltf_reference().items():
        if name == 'drawer_cabinet_drawer_01':
            triangles = triangles @ matrix.T + np.array(pose[:3])
        # Millimetre preconditioning avoids scale-sensitive near-surface masks
        # in the library; reported coordinates and distances remain metres.
        scaled = triangles * 1000.
        result = {}
        for label, point in points.items():
            nearest = trimesh.triangles.closest_point(scaled, np.repeat((point * 1000.)[None, :], len(scaled), axis=0)) / 1000.
            ds = np.linalg.norm(nearest - point, axis=1)
            index = int(np.argmin(ds))
            result[label] = {'distance_m': float(ds[index]), 'closest_original_surface_point_m': nearest[index].tolist(), 'triangle_index': index, 'original_triangle_at_measured_pose_m': triangles[index].tolist()}
        distances[name] = result
    for path, digest in digests.items():
        assert sha(Path(path)) == digest
    record = {'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'scope': 'Read-only original surface distance at the deepest measured contact. Contact actor IDs are absent; no collision-pair or volume-intersection proof is inferred.',
              'input_sha256': digests, 'inputs_unchanged': True,
              'selection': 'Minimum recorded contact separation across the complete trial trace',
              'step': row['step'], 'time_s': row['time_s'], 'phase': row['phase'],
              'contact': contact, 'measured_drawer_pose_xyz_xyzw': pose,
              'original_drawer_body_frame': 'Identity initial world frame from verified native source/Joint and saved USD. Apply measured origin pose to the original upper drawer; cabinet/lower drawer surfaces stay static.',
              'point_construction_limit': 'The two normal-offset points are diagnostic hypotheses because the trace does not identify contact actor order or a second contact point.',
              'closest_original_surface_distances': distances,
              'new_physics_steps': 0, 'asset_mutations': 0}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'output': str(a.output), 'sha256': sha(a.output), 'time_s': row['time_s'], 'contact_separation_m': separation, 'closest_distances_at_reported_position_m': {name: value['reported_contact_position']['distance_m'] for name, value in distances.items()}}))


if __name__ == '__main__':
    main()
