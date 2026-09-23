"""Pure numeric frame and contact-unit operations for drawer evaluation."""
import math
import numpy as np


def unit_quaternion(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all():
        raise ValueError('Invalid quaternion')
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        raise ValueError('Zero quaternion')
    return q / norm


def quaternion_angle_deg(a, b):
    dot = abs(float(np.dot(unit_quaternion(a), unit_quaternion(b))))
    return math.degrees(2 * math.acos(float(np.clip(dot, 0, 1))))


def rotation_matrix(q):
    x, y, z, w = unit_quaternion(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])


def payload_in_initial_world(payload_position, current_drawer, initial_drawer):
    current = np.asarray(current_drawer, dtype=float)
    initial = np.asarray(initial_drawer, dtype=float)
    position = np.asarray(payload_position, dtype=float)
    return initial[:3] + rotation_matrix(initial[3:7]) @ rotation_matrix(current[3:7]).T @ (position-current[:3])


def contact_force_from_impulse(impulse, dt):
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Positive finite dt required')
    return np.asarray(impulse, dtype=float) / dt
