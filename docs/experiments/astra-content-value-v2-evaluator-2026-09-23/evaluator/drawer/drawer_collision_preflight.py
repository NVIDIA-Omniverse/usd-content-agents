"""Native initial cooked-shape queries, with an independent static witness."""
import hashlib
import json
from pathlib import Path
import sys
import traceback

import numpy as np
from ovphysx import PhysX, TensorType, SceneQueryGeometryType, SceneQueryMode


def run(request):
    p = None
    binding = None
    result = {'status': 'INCONCLUSIVE', 'pass': False, 'simulation_steps': 0}
    try:
        path = Path(request['scene_usd'])
        if hashlib.sha256(path.read_bytes()).hexdigest() != request['scene_sha256']:
            raise RuntimeError('Preflight scene bytes changed')
        p = PhysX(device='cpu')
        handle, op = p.add_usd(str(path))
        p.wait_op(op)
        binding = p.create_tensor_binding(pattern=request['drawer_body'], tensor_type=TensorType.RIGID_BODY_POSE)
        if binding.shape != (1, 7):
            raise RuntimeError('Expected exactly one drawer pose')
        pose = np.zeros(binding.shape, np.float32)
        binding.read(pose)  # Required initialization before any scene query.
        if not np.isfinite(pose).all():
            raise RuntimeError('Nonfinite initial drawer pose')
        witness = p.overlap(SceneQueryGeometryType.SPHERE, mode=SceneQueryMode.ALL,
                            radius=.05, position=request['witness_center'])
        result.update(initial_drawer_pose=pose.tolist(), positive_control_hits=witness)
        if not witness:
            raise RuntimeError('Initialized query failed the known static witness')
        payload = request['spec']['payload']
        half = np.array(payload['size_m']) / 2
        angle = np.deg2rad(payload['initial_yaw_jitter_deg'])
        if not (half[0] == half[2] and 0 <= angle <= np.pi/4):
            raise RuntimeError('Swept envelope qualification requires square XZ and yaw <=45 degrees')
        extent = [half[0]*np.cos(angle)+half[2]*np.sin(angle)+payload['initial_x_jitter_m'],
                  half[1], half[2]*np.cos(angle)+half[0]*np.sin(angle)+payload['initial_z_jitter_m']]
        hits = p.overlap(SceneQueryGeometryType.BOX, mode=SceneQueryMode.ALL,
                         half_extent=extent, position=payload['initial_center_m'])
        iq = request['spec']['interior_queries']
        interior = []
        for x in iq['x_m']:
            for z in iq['z_m']:
                position = [x, iq['y_m'], z]
                found = p.overlap(SceneQueryGeometryType.SPHERE, mode=SceneQueryMode.ALL,
                                  radius=iq['sphere_radius_m'], position=position)
                interior.append({'position': position, 'hits': found})
        checks = {'query_positive_control': bool(witness),
                  'initial_payload_envelope_clear': not hits,
                  'interior_clear': all(not x['hits'] for x in interior)}
        result.update(payload_envelope={'center': payload['initial_center_m'], 'half_extent': extent, 'hits': hits},
                      interior_queries=interior, checks=checks)
        result['pass'] = all(checks.values())
        result['status'] = 'PASS' if result['pass'] else 'FAIL'
    except Exception as exc:
        result.update(error=type(exc).__name__ + ': ' + str(exc), traceback=traceback.format_exc())
    finally:
        if binding is not None:
            binding.destroy()
        if p is not None:
            p.release()
    Path(request['output']).write_text(json.dumps(result, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if hasattr(x, 'tolist') else str(x)) + '\n')
    return result


if __name__ == '__main__':
    result = run(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps({'status': result['status']}))
    raise SystemExit(0 if result['pass'] else (1 if result['status'] == 'FAIL' else 2))
