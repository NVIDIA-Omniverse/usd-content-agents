"""Bound the actual controller command after its seeded multiplier."""
import math


def clamp_effort(value, cap):
    if not math.isfinite(value) or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Controller effort/cap must be finite and cap positive')
    return min(max(value, -cap), cap)
