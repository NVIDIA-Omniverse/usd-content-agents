# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal module entry point for Workflow 3 collection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from content_agent_workflows.common.artifacts import load_json

from .collector import CollectionRuntimeError, run_collection
from .contracts import CollectionRequest


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect asset-task results")
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        request = CollectionRequest.model_validate(load_json(args.request))
        result = run_collection(request)
    except (CollectionRuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
