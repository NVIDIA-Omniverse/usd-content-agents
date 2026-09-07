# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run one rendered material refinement from a checked-in YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from material_agent.api import refine_material


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = refine_material(args.config, output_dir=args.output_dir)
    if not result.success:
        raise RuntimeError(result.error or "Material refinement failed")


if __name__ == "__main__":
    main()
