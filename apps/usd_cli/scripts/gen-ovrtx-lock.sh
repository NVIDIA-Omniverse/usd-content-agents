#!/usr/bin/env bash
# Regenerate the hash-pinned OVRTX runtime lock (PEP 751) that
# usd_core/render/ovrtx.py::_provision_venv installs from, then mark every
# artifact-digest line for secret scanners.
#
# The pragma step is NOT optional: the lock's sha256 values are package pins,
# not credentials, but they read as "Hex High Entropy String" to
# detect-secrets. A scan-side exclusion only protects invocations that pass
# the flag — the world-understanding repo's `usd-cli source gate` mirror
# workflow scans the vendored copy with a plain `detect-secrets scan
# --all-files`, so the allowlist must travel INSIDE the generated file.
# `uv pip compile` rewrites the lock wholesale (destroying any hand-added
# comments), which is why regeneration goes through this script instead of
# the bare compile command. tests/test_pr819_review_fixes.py fails if the
# pragmas are missing, so a hand-run compile that skips this script is
# caught locally before any gate sees it.
#
# Locally:  bash scripts/gen-ovrtx-lock.sh
# Requires: uv >= 0.7 (pylock.toml output format). The native profile must
# match OVRTX_PIN/OVSTAGE_PIN/WARP_PIN in src/usd_core/render/ovrtx.py.
set -euo pipefail
cd "$(dirname "$0")/.."

PROFILE=src/usd_core/render/ovrtx_runtime_profile.in
LOCK=src/usd_core/render/pylock.ovrtx-runtime.toml

uv pip compile --no-config "${PROFILE}" \
  --python-version 3.11 --universal \
  --extra-index-url https://pypi.nvidia.com \
  --format pylock.toml \
  --output-file "${LOCK}"

python3 - "${LOCK}" <<'PY'
import re
import sys

path = sys.argv[1]
lines = []
for line in open(path, encoding="utf-8").read().splitlines():
    if re.search(r'sha256 = "[0-9a-f]{64}"', line) and "allowlist secret" not in line:
        line += "  # pragma: allowlist secret"
    lines.append(line)
with open(path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")
print(f"pragma-marked digest lines in {path}")
PY
