#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify that a retained OpenCV build contains no FFmpeg surface."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Any

_FFMPEG_LIBRARIES = (
    "libavcodec",
    "libavformat",
    "libavutil",
    "libswresample",
    "libswscale",
)


def _load_cv2() -> Any:
    """Import OpenCV lazily so an absent install produces verifier evidence."""
    try:
        return importlib.import_module("cv2")
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"OpenCV is required for verification: {exc}") from None


def _linked_ffmpeg_libraries(linkage: str) -> list[str]:
    """Return FFmpeg-linked ``ldd`` rows, including versioned SONAMEs."""
    return [
        line.strip()
        for line in linkage.splitlines()
        if any(library in line for library in _FFMPEG_LIBRARIES)
    ]


def verify(*, ffmpeg_mode: str) -> dict[str, object]:
    """Return absence evidence or raise when the codec boundary is violated."""
    if ffmpeg_mode != "forbidden":
        raise ValueError(f"unsupported FFmpeg mode: {ffmpeg_mode!r}")
    cv2 = _load_cv2()
    package_dir = Path(cv2.__file__).resolve().parent
    native_candidates = sorted(package_dir.glob("cv2*.so"))
    if len(native_candidates) != 1:
        raise RuntimeError(
            f"expected one OpenCV extension in {package_dir}, got {native_candidates}"
        )
    native = native_candidates[0]
    site_packages = package_dir.parent
    bundled = sorted(
        str(path.relative_to(site_packages))
        for library in _FFMPEG_LIBRARIES
        for path in site_packages.rglob(f"{library}*.so*")
    )
    if bundled:
        raise RuntimeError(f"bundled FFmpeg libraries are forbidden: {bundled}")

    linkage = subprocess.run(
        ["ldd", str(native)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    build_info = cv2.getBuildInformation()
    ffmpeg_enabled = (
        re.search(r"^\s*FFMPEG:\s+YES\s*$", build_info, re.MULTILINE) is not None
    )
    linked_ffmpeg = _linked_ffmpeg_libraries(linkage)
    if ffmpeg_enabled or linked_ffmpeg:
        raise RuntimeError(
            "OpenCV included forbidden FFmpeg support: "
            f"enabled={ffmpeg_enabled}, linkage={linked_ffmpeg}"
        )

    bundled_executables = sorted(
        str(path.relative_to(site_packages))
        for path in site_packages.rglob("ffmpeg-*")
        if path.is_file()
    )
    if bundled_executables:
        raise RuntimeError(
            f"bundled FFmpeg executables are forbidden: {bundled_executables}"
        )

    if find_spec("imageio_ffmpeg") is not None:
        raise RuntimeError("the imageio-ffmpeg package is forbidden")

    return {
        "schema": "security.opencv-no-ffmpeg.v2",
        "ffmpeg_mode": ffmpeg_mode,
        "opencv_version": cv2.__version__,
        "opencv_extension": str(native),
        "ffmpeg_enabled": False,
        "linked_ffmpeg_libraries": [],
        "bundled_ffmpeg_artifacts": [],
        "imageio_ffmpeg_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ffmpeg", choices=("forbidden",), required=True)
    args = parser.parse_args()
    print(json.dumps(verify(ffmpeg_mode=args.ffmpeg), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
