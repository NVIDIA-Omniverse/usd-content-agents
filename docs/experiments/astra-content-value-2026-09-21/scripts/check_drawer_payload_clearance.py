"""Read-only source/proxy clearance diagnostic; never changes scored outcomes."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), 'Choose a new output file'
r = a.root.resolve()
sys.path.insert(0, str(r / 'evaluator'))
import drawer_evaluate as frozen
import numpy as np
import trimesh
from pxr import Usd, UsdGeom, UsdPhysics

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
run = r / 'runs/pilot-v1/01_drawer/plain_astra'
paths = [r / 'evaluator/drawer_evaluate.py', r / 'evaluator/drawer_acceptance.json',
         r / 'evaluator/reference/drawer_cabinet_1k.gltf', r / 'evaluator/reference/drawer_cabinet.bin',
         run / 'final.usd', run / 'bindings.json']
before = {str(path.relative_to(r)): sha(path) for path in paths}
triangles = frozen.gltf_reference()['drawer_cabinet_drawer_01']
mesh = trimesh.Trimesh(vertices=triangles.reshape(-1, 3),
                       faces=np.arange(triangles.size // 3).reshape(-1, 3), process=False)
bindings = json.loads((run / 'bindings.json').read_text())
stage = Usd.Stage.Open(str(run / 'final.usd'))
floor_path = bindings['drawer_body'] + '/Collisions/floor'
floor = stage.GetPrimAtPath(floor_path)
assert floor and floor.IsA(UsdGeom.Cube) and floor.HasAPI(UsdPhysics.CollisionAPI)
matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(floor))
assert np.max(np.abs(matrix[:3, :3] - np.diag(np.diag(matrix[:3, :3])))) < 1e-12
box = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.proxy]).ComputeWorldBound(floor).ComputeAlignedRange()
minimum, maximum = np.asarray(box.GetMin()), np.asarray(box.GetMax())
spec = frozen.SPEC
payload_bottom = spec['payload']['initial_center_m'][1] - spec['payload']['size_m'][1] / 2
samples = []
for x, z in [(0, 0), (-.05, -.035), (.05, .035)]:
    locations, _, _ = mesh.ray.intersects_location(np.array([[x, 2, z]]), np.array([[0, -1, 0]]))
    heights = sorted(set(locations[:, 1].tolist()))
    assert len(heights) >= 2 and minimum[0] < x < maximum[0] and minimum[2] < z < maximum[2]
    top = max(heights)
    samples.append({'x_m': x, 'z_m': z, 'source_intersection_y_m': heights,
                    'source_floor_top_y_m': top, 'source_floor_above_payload_bottom_m': top - payload_bottom,
                    'source_floor_above_proxy_top_m': top - float(maximum[1])})
assert before == {str(path.relative_to(r)): sha(path) for path in paths}
record = {'scope': 'Unscored source/proxy diagnostic at three declared positions, not a new acceptance rule or a full collider-fidelity certification.',
          'inputs_sha256': before, 'diagnostic_code_sha256': sha(Path(__file__)), 'inputs_unchanged': True,
          'frozen_evaluator_or_thresholds_modified': False, 'scored_artifacts_modified': False,
          'frozen_payload_bottom_y_m': payload_bottom, 'baseline_floor_collider_path': floor_path,
          'baseline_floor_bounds_min_m': minimum.tolist(), 'baseline_floor_bounds_max_m': maximum.tolist(),
          'baseline_initial_proxy_overlap_m': float(maximum[1]) - payload_bottom,
          'samples': samples,
          'interpretation': 'The frozen task measures contacts against authored collision approximations. Its payload starts about8mm below the original visible floor at these samples; the accepted baseline proxy floor is about4.50mm below that visible floor and initially overlaps the payload by3.50mm. The frozen5mm penetration gate therefore does not certify exact contact fidelity to the original rendered floor.'}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'output': str(a.output), 'sha256': sha(a.output), 'proxy_overlap_m': record['baseline_initial_proxy_overlap_m']}))
