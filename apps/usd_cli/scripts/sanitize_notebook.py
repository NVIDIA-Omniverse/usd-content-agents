#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strip absolute host paths from an executed notebook's outputs.

The usd-cli CLI prints some absolute paths in its human-readable summaries (checkpoints,
physics recordings). When the demo notebook is executed on a GPU host, those leak the
host's home dir (e.g. /home/<user>/usd-cli/...). This rewrites output text so paths are
repo-relative and portable — no host or username in the committed notebook.

Usage: sanitize_notebook.py <notebook.ipynb> [repo_root] [home]
  repo_root defaults to $PWD, home to $HOME. The longer (repo_root) prefix is stripped
  first so repo paths become relative and any remaining home paths collapse to '~/'.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import nbformat


def _clean(s: str, root: str, home: str) -> str:
    s = s.replace(root.rstrip("/") + "/", "")
    return s.replace(home.rstrip("/") + "/", "~/")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    root = sys.argv[2] if len(sys.argv) > 2 else os.getcwd()
    home = sys.argv[3] if len(sys.argv) > 3 else os.path.expanduser("~")

    nb = nbformat.read(str(path), as_version=4)
    changed = 0
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream" and "text" in out:
                new = _clean(out["text"], root, home)
                if new != out["text"]:
                    out["text"] = new
                    changed += 1
            data = out.get("data") or {}
            tp = data.get("text/plain")
            if tp is not None:
                joined = "".join(tp) if isinstance(tp, list) else tp
                new = _clean(joined, root, home)
                if new != joined:
                    data["text/plain"] = new
                    changed += 1
    nbformat.write(nb, str(path))
    print(f"sanitized {path}: {changed} output block(s) rewritten (root={root}, home={home})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
