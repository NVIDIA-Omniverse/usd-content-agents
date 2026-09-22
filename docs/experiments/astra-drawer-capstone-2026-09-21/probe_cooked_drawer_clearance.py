"""Read-only cooked-shape clearance; no USD edits, payload, force or time steps."""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from ovphysx import PhysX, TensorType, SceneQueryGeometryType, SceneQueryMode

parser = argparse.ArgumentParser()
parser.add_argument('--asset', type=Path, required=True)
parser.add_argument('--asset-sha256', required=True)
parser.add_argument('--spec', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(args.asset) == args.asset_sha256
assert not args.output.exists()
spec = json.loads(args.spec.read_text())
payload = spec['payload']
half = [v / 2 for v in payload['size_m']]
angle = math.radians(payload['initial_yaw_jitter_deg'])
# This one box contains every frozen seeded/jittered payload volume, including yaw.
extent = [half[0]*math.cos(angle)+half[2]*math.sin(angle)+payload['initial_x_jitter_m'],
          half[1],
          half[2]*math.cos(angle)+half[0]*math.sin(angle)+payload['initial_z_jitter_m']]
center = payload['initial_center_m']
record = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'asset': str(args.asset), 'asset_sha256': args.asset_sha256,
          'spec': str(args.spec), 'spec_sha256': sha(args.spec),
          'scope': 'Initial cooked-shape queries only; no time steps or physical-task acceptance.',
          'backend': 'ovphysx 0.4.13 CPU', 'simulation_steps': 0}
physx = PhysX(device='cpu')
binding = None
try:
    handle, operation = physx.add_usd(str(args.asset))
    physx.wait_op(operation)
    binding = physx.create_tensor_binding(pattern='/Asset/drawer_cabinet_drawer_01_1', tensor_type=TensorType.RIGID_BODY_POSE)
    pose = np.zeros(binding.shape, np.float32)
    binding.read(pose)
    record['initial_pose'] = pose.tolist()
    record['payload_envelope'] = {'center': center, 'half_extent': extent,
        'hits': physx.overlap(SceneQueryGeometryType.BOX, mode=SceneQueryMode.ALL,
                              half_extent=extent, position=center)}
    record['original_invalid_spawn_envelope'] = {
        'center': [center[0], 1.035, center[2]], 'half_extent': extent,
        'hits': physx.overlap(SceneQueryGeometryType.BOX, mode=SceneQueryMode.ALL,
                              half_extent=extent, position=[center[0], 1.035, center[2]])}
    interior = spec['interior_queries']
    record['frozen_interior_queries'] = []
    for x in interior['x_m']:
        for z in interior['z_m']:
            position = [x, interior['y_m'], z]
            hits = physx.overlap(SceneQueryGeometryType.SPHERE, mode=SceneQueryMode.ALL,
                                  radius=interior['sphere_radius_m'], position=position)
            record['frozen_interior_queries'].append({'position': position, 'hits': hits})
    record['floor_rays'] = []
    for i in range(5):
        for j in range(5):
            origin = [center[0]+extent[0]*(i/2-1), center[1], center[2]+extent[2]*(j/2-1)]
            hits = physx.raycast(origin=origin, direction=[0, -1, 0], distance=.1,
                                  mode=SceneQueryMode.ALL, both_sides=True)
            record['floor_rays'].append({'origin': origin, 'hits': hits})
finally:
    if binding is not None:
        binding.destroy()
    physx.release()
record['query_positive_control'] = bool(record['original_invalid_spawn_envelope']['hits']) and any(x['hits'] for x in record['floor_rays'])
record['asset_unchanged'] = sha(args.asset) == args.asset_sha256
record['initial_payload_envelope_clear'] = record['query_positive_control'] and not record['payload_envelope']['hits']
record['frozen_interior_clear'] = record['query_positive_control'] and all(not x['hits'] for x in record['frozen_interior_queries'])
record['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(record, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x))+'\n')
print(json.dumps({k:record[k] for k in ['asset_sha256', 'asset_unchanged', 'initial_payload_envelope_clear', 'frozen_interior_clear', 'simulation_steps']}))
