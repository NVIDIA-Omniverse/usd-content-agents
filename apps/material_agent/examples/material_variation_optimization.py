# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Create a rendered material variation set from a YAML configuration."""

from __future__ import annotations

import argparse
from pathlib import Path

from material_agent.api import create_material_variations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    result = create_material_variations(args.config, output_dir=args.output_dir)
    if not result.success:
        raise RuntimeError(result.error or "Material variation failed")


if __name__ == "__main__":
    main()
