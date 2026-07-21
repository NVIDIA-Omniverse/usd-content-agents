# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Joint Agent Service wheel metadata contract tests."""

from __future__ import annotations

import os
import subprocess
import zipfile
from email.parser import BytesParser
from email.policy import default
from pathlib import Path

from packaging.requirements import Requirement

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_built_wheel_requires_compatible_dependency_floors(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["UV_CACHE_DIR"] = str(tmp_path / "uv-cache")
    subprocess.run(
        [
            "uv",
            "build",
            "--wheel",
            "--out-dir",
            str(tmp_path),
            str(SERVICE_ROOT),
        ],
        check=True,
        cwd=SERVICE_ROOT.parents[1],
        env=env,
    )

    wheels = list(tmp_path.glob("joint_agent_service-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as wheel:
        metadata_paths = [
            name for name in wheel.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1
        metadata = BytesParser(policy=default).parsebytes(wheel.read(metadata_paths[0]))

    joint_agent_requirements = [
        requirement
        for value in metadata.get_all("Requires-Dist", [])
        if (requirement := Requirement(str(value))).name == "joint-agent"
    ]
    assert [str(requirement) for requirement in joint_agent_requirements] == [
        "joint-agent>=0.5.0"
    ]

    aioboto3_requirements = [
        requirement
        for value in metadata.get_all("Requires-Dist", [])
        if (requirement := Requirement(str(value))).name == "aioboto3"
    ]
    assert [str(requirement) for requirement in aioboto3_requirements] == [
        "aioboto3>=13.3.0"
    ]
