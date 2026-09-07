#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the OVRTX + official VoMP rigid-body mass blueprint."""

from __future__ import annotations

import argparse
from pathlib import Path

from physics_agent.integrations import VompRuntimeConfig, run_vomp_mass_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_usd", type=Path)
    parser.add_argument("output_usd", type=Path)
    parser.add_argument("--target-prim", required=True)
    parser.add_argument("--vomp-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("output/vomp_evidence"))
    args = parser.parse_args()

    vomp_root = args.vomp_root.expanduser().resolve()
    result = run_vomp_mass_pipeline(
        args.input_usd,
        args.output_usd,
        target_prim_path=args.target_prim,
        work_dir=args.work_dir,
        runtime_config=VompRuntimeConfig(
            runtime_root=vomp_root,
            python_executable=vomp_root / ".venv/bin/python",
            config_path=Path("weights/inference.json"),
        ),
    )
    properties = result.apply_result.mass_properties
    print(f"Mass: {properties.mass_kg:.9g} kg")
    print(f"Output USD: {result.apply_result.output_usd_path}")
    print(f"Provenance: {result.apply_result.provenance_path}")
    print(f"Evidence: {result.evidence.artifact_dir}")


if __name__ == "__main__":
    main()
