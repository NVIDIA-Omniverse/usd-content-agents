"""Trusted evaluator math and verdicts. No benchmark author code is imported."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

SEEDS = [11, 23, 47, 83, 131]

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def check(checks, name, passed, evidence=None, insufficient=False):
    checks.append({'name': name, 'passed': bool(passed), 'evidence': evidence,
                   'failure_class': None if passed else ('evaluator_insufficient' if insufficient else 'rejected')})

def verdict(checks):
    failures = [x for x in checks if not x['passed']]
    concrete = [x for x in failures if x['failure_class'] != 'evaluator_insufficient']
    return {'accepted': not failures,
            'status': 'accepted' if not failures else ('not_accepted' if concrete else 'inconclusive'),
            'inconclusive_checks': [x['name'] for x in failures if x['failure_class'] == 'evaluator_insufficient'],
            'concrete_failures': [x['name'] for x in concrete],
            'checks': checks}

def rotation(q):
    """xyzw quaternion, as returned by ovphysx body-pose tensors."""
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    x, y, z, w = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])

def pose_matrix(p):
    result = np.eye(4)
    result[:3, :3] = rotation(p[3:7])
    result[:3, 3] = p[:3]
    return result

def rigid_matrix(m, tol=1e-5):
    m = np.asarray(m, float)
    return bool(m.shape == (4,4) and np.isfinite(m).all() and
                np.allclose(m[3], [0,0,0,1], atol=tol) and
                np.allclose(m[:3,:3].T @ m[:3,:3], np.eye(3), atol=tol) and
                abs(np.linalg.det(m[:3,:3])-1) <= tol)

def transform(vertices, matrix):
    return np.asarray(vertices) @ np.asarray(matrix)[:3,:3].T + np.asarray(matrix)[:3,3]

def frame(joint, poses, index):
    body = joint['body' + str(index)]
    base = np.eye(4) if body is None else pose_matrix(poses[body])
    return base @ np.asarray(joint['frame' + str(index)], float)

def joint_measure(joint, poses):
    a, b = frame(joint, poses, 0), frame(joint, poses, 1)
    axis = {'X': 0, 'Y': 1, 'Z': 2}[joint['axis']]
    delta = a[:3,:3].T @ (b[:3,3]-a[:3,3])
    r = a[:3,:3].T @ b[:3,:3]
    if joint['kind'] == 'prismatic':
        q = float(delta[axis])
        closure = float(np.linalg.norm(np.delete(delta, axis)))
        angular_error = math.acos(np.clip((np.trace(r)-1)/2, -1, 1))
    elif joint['kind'] == 'fixed':
        q = 0.0
        closure = float(np.linalg.norm(delta))
        angular_error = math.acos(np.clip((np.trace(r)-1)/2, -1, 1))
    else:
        i, k = [v for v in range(3) if v != axis]
        # X: atan2(Rzy,Ryy); Y: atan2(Rxz,Rzz); Z: atan2(Ryx,Rxx)
        q = math.atan2(r[2,1],r[1,1]) if axis == 0 else (math.atan2(r[0,2],r[2,2]) if axis == 1 else math.atan2(r[1,0],r[0,0]))
        closure = float(np.linalg.norm(delta))
        angular_error = math.acos(np.clip(np.dot(a[:3,axis],b[:3,axis]),-1,1))
    return {'q': q, 'closure_m': closure, 'axis_error_rad': angular_error, 'axis_world': a[:3,axis].tolist()}

def finite_positive(values):
    values = np.asarray(values, float)
    return bool(values.size and np.isfinite(values).all() and (values > 0).all())
