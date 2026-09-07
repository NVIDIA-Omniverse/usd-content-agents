# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Argument parsing helpers for the hybrid syntax (cli-design.md §0).

- vectors are comma lists:           "0,1,0"  -> [0.0, 1.0, 0.0]
- signed scalars carry relative/abs:  "+3" -> ("relative", 3.0);  "3" -> ("absolute", 3.0)
- resolution:                         "1920x1080" -> (1920, 1080)
"""

from __future__ import annotations


def vec(s: str | None, dim: int | None = None,
        label: str | None = None) -> list[float] | None:
    """Parse 'a,b,c' into floats; errors NAME the argument and show the shape.

    A space-separated `--screen 512 512` fed the stray token into the next
    positional and produced "expected 3 components, got 1" with a rich
    traceback — the message must say which argument and that commas separate
    components."""
    if s is None:
        return None
    where = f"{label}: " if label else ""
    example = ",".join(str(i + 1) for i in range(dim or 3))
    try:
        parts = [float(x) for x in s.split(",")]
    except ValueError as exc:
        raise ValueError(f"{where}expected comma-separated numbers "
                         f"(e.g. {example}), got '{s}'") from exc
    if dim is not None and len(parts) != dim:
        raise ValueError(f"{where}expected {dim} comma-separated components "
                         f"(e.g. {example}), got {len(parts)} in '{s}' — "
                         "no spaces between components")
    return parts


def signed(s: str | None) -> dict | None:
    """Return {'mode': 'relative'|'absolute', 'value': float} for a transform delta.

    A leading '+' or '-' sign means *relative* (`--tx +3` adds 3); a bare number is
    *absolute* (`--tx 3` sets 3). See cli-design.md §0 principle 3.
    """
    if s is None:
        return None
    mode = "relative" if s[0] in "+-" else "absolute"
    try:
        return {"mode": mode, "value": float(s)}
    except ValueError as exc:
        raise ValueError(f"expected a number (optionally +/- prefixed), got '{s}'") from exc


def resolution(s: str | None) -> list[int] | None:
    if s is None:
        return None
    try:
        w, h = s.lower().split("x")
        return [int(w), int(h)]
    except ValueError as exc:
        raise ValueError(f"expected WxH (e.g. 1920x1080), got '{s}'") from exc


def _is_empty(v) -> bool:
    """True for the values drop_none strips: None, empty tuple/list, literal False.

    Identity/type-aware on purpose: `0 == False` and `0.0 == False` in Python, so a
    membership test like `v in (None, (), [], False)` silently swallowed zero-valued
    options (--elevation 0, --metallic 0, --roughness 0, --opacity 0 all vanished and
    the server defaults won). Numeric zeros must survive; only a *bool* False (an
    unset CLI flag) is dropped.
    """
    if v is None:
        return True
    if isinstance(v, (tuple, list)) and len(v) == 0:
        return True
    return isinstance(v, bool) and v is False


def drop_none(d: dict) -> dict:
    """Strip keys whose value is None/empty so payloads stay compact (zeros are kept)."""
    return {k: v for k, v in d.items() if not _is_empty(v)}
