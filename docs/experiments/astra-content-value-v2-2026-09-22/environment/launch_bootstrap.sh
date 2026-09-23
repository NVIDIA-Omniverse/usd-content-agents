#!/usr/bin/env bash
set -euo pipefail
RERUN_ROOT="$(cd "$1" && pwd)"
export TMPDIR="$RERUN_ROOT/tmp" UV_NO_CACHE=1 PIP_NO_CACHE_DIR=1
export UV_PYTHON_INSTALL_DIR="$RERUN_ROOT/tools/python"
mkdir -p "$RERUN_ROOT/tools" "$RERUN_ROOT/tmp" "$RERUN_ROOT/environment/evidence"
if [[ ! -x "$RERUN_ROOT/tools/uv/uv" ]]; then
 curl -fL --retry 3 --max-time 240 https://github.com/astral-sh/uv/releases/download/0.8.22/uv-x86_64-unknown-linux-gnu.tar.gz -o "$TMPDIR/uv.tar.gz"
 printf '%s  %s\n' 741ff1f5742c5a4a25d2f829e8395355e43f7a5ae2ebc6368e9ae2df0efb69cf "$TMPDIR/uv.tar.gz" | sha256sum --check
 mkdir -p "$RERUN_ROOT/tools/uv"
 tar -xzf "$TMPDIR/uv.tar.gz" --strip-components=1 -C "$RERUN_ROOT/tools/uv"
 rm "$TMPDIR/uv.tar.gz"
fi
export PATH="$RERUN_ROOT/tools/uv:$PATH"
uv python install 3.12.11
export UV_PYTHON=3.12.11
python3 "$RERUN_ROOT/environment/bootstrap.py" --root "$RERUN_ROOT" "${@:2}"
