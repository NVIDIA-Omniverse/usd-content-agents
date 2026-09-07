# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight confined artifact writer for managed child workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

CONTROLLED_ARTIFACT_ROOT_ENV = "CONTENT_WORKFLOW_CONTROLLED_ARTIFACT_ROOT"
CONTROLLED_JSON_MAX_CHARS = 8 * 1024 * 1024
CONTROLLED_JSON_REPLACE_KEYS = frozenset(
    {
        "raw/material_decision_patch.json",
        "raw/physics_decision_patch.json",
    }
)


def write_controlled_json_artifact(
    *,
    output: str,
    json_document: str | None,
    replace_existing: bool = False,
) -> None:
    """Publish model-authored JSON beneath the wrapper-pinned child root."""

    from world_understanding.utils.artifacts import (
        open_confined_directory,
        validated_artifact_relative_key,
        write_bytes_to_confined,
    )

    configured_root = os.environ.get(CONTROLLED_ARTIFACT_ROOT_ENV)
    if not configured_root:
        raise RuntimeError("controlled artifact root is not configured")
    run_dir = Path(configured_root).expanduser().resolve(strict=True)
    cwd = Path.cwd().resolve(strict=True)
    if os.path.normcase(os.fspath(cwd)) != os.path.normcase(os.fspath(run_dir)):
        raise RuntimeError("controlled JSON writes require the child run as cwd")
    output_key = validated_artifact_relative_key(output)
    if replace_existing and output_key not in CONTROLLED_JSON_REPLACE_KEYS:
        raise ValueError(
            "controlled JSON replacement is limited to child-owned decision patches"
        )
    raw_document = json_document
    if raw_document is None:
        raw_document = sys.stdin.read(CONTROLLED_JSON_MAX_CHARS + 1)
    if len(raw_document) > CONTROLLED_JSON_MAX_CHARS:
        raise ValueError("controlled JSON artifact exceeds the size limit")
    payload = json.loads(raw_document)
    if not isinstance(payload, dict | list):
        raise TypeError("controlled JSON artifact must be an object or array")
    encoded = (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    with open_confined_directory(run_dir) as root_descriptor:
        published = write_bytes_to_confined(
            root_descriptor,
            output_key,
            encoded,
            overwrite=replace_existing,
            file_mode=0o600,
        )
    if not published:
        raise FileExistsError(f"controlled JSON artifact already exists: {output_key}")
    print(
        json.dumps(
            {"output": output_key, "size_bytes": len(encoded)},
            sort_keys=True,
        )
    )


def main(argv: list[str]) -> int:
    """Run only the lightweight ``artifact write-json`` command surface."""

    parser = argparse.ArgumentParser(prog="content-workflow-cli artifact write-json")
    parser.add_argument(
        "--output",
        required=True,
        help="Canonical forward-slash path relative to the child run directory.",
    )
    parser.add_argument(
        "--json",
        dest="json_document",
        help="Complete JSON document. When omitted, the document is read from stdin.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an approved child-owned decision patch.",
    )
    try:
        args = parser.parse_args(argv)
        write_controlled_json_artifact(
            output=args.output,
            json_document=args.json_document,
            replace_existing=args.replace,
        )
    except SystemExit as exc:
        if exc.code is None:
            return 0
        return exc.code if isinstance(exc.code, int) else 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"content-workflow-cli: error: {exc}", file=sys.stderr)
        return 2
    return 0
