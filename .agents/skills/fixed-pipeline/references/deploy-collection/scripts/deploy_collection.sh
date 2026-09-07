#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

is_wu_repo_root() {
  [[ -f "$1/deploy/collection/deploy.py" ]]
}

if [[ -n "${WU_REPO_ROOT:-}" ]]; then
  repo_root="$(cd "${WU_REPO_ROOT}" && pwd)"
elif repo_root="$(git -C "${script_dir}" rev-parse --show-toplevel 2>/dev/null)" \
  && is_wu_repo_root "${repo_root}"; then
  :
elif repo_root="$(git rev-parse --show-toplevel 2>/dev/null)" \
  && is_wu_repo_root "${repo_root}"; then
  :
else
  echo "ERROR: run this from the world-understanding checkout or set WU_REPO_ROOT." >&2
  exit 1
fi

if ! is_wu_repo_root "${repo_root}"; then
  echo "ERROR: deploy/collection/deploy.py not found under repo root: ${repo_root}" >&2
  exit 1
fi

if [[ -x "${repo_root}/.venv/bin/python" ]]; then
  python_bin="${repo_root}/.venv/bin/python"
else
  python_bin="python3"
fi

exec "${python_bin}" "${repo_root}/deploy/collection/deploy.py" "$@"
