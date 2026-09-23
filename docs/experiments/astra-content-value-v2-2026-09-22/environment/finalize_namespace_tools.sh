#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then
  printf '%s\n' 'Usage: finalize_namespace_tools.sh EXPERIMENT_ROOT EXACT_MODEL_CATALOG_JSON' >&2
  exit 2
fi
TASK_ROOT="$(realpath "$1")"
TASK_CATALOG="$(realpath "$2")"
TASK_PYTHON="$TASK_ROOT/repo/.venv/bin/python"
TASK_ENV="$TASK_ROOT/environment"
TASK_EXPORT="$TASK_ROOT/namespace_tools_final"
TASK_EVIDENCE="$TASK_ENV/evidence"
# Every output is create-only. This is the same command sequence for both hosts.
nice -n15 ionice -c3 "$TASK_PYTHON" "$TASK_ENV/export_namespace_tools.py" \
  --root "$TASK_ROOT" --output "$TASK_EXPORT" \
  --receipt "$TASK_EVIDENCE/final_export_initial.json"
nice -n15 "$TASK_PYTHON" "$TASK_ENV/supplement_namespace_tools.py" \
  --tools "$TASK_EXPORT" --parent-receipt "$TASK_EVIDENCE/final_export_initial.json" \
  --catalog "$TASK_CATALOG" --receipt "$TASK_EVIDENCE/final_export_catalog.json"
nice -n15 "$TASK_PYTHON" "$TASK_ENV/normalize_export_metadata.py" \
  --root "$TASK_ROOT" --tools "$TASK_EXPORT" \
  --parent-receipt "$TASK_EVIDENCE/final_export_catalog.json" \
  --receipt "$TASK_EVIDENCE/final_export_metadata.json"
nice -n15 ionice -c3 "$TASK_PYTHON" "$TASK_ENV/prepare_ovrtx_cache_targets.py" \
  --tools "$TASK_EXPORT" --parent-receipt "$TASK_EVIDENCE/final_export_metadata.json" \
  --receipt "$TASK_EVIDENCE/final_export.json"
sudo -n nice -n15 ionice -c3 "$TASK_PYTHON" "$TASK_ENV/prepare_native_git_ownership.py" \
  --tools "$TASK_EXPORT" --receipt "$TASK_EVIDENCE/final_native_git_ownership.json"
nice -n15 "$TASK_PYTHON" "$TASK_ENV/run_namespace_smoke.py" \
  --tools "$TASK_EXPORT" --output "$TASK_EVIDENCE/final_namespace_smoke"
nice -n15 "$TASK_PYTHON" "$TASK_ENV/run_namespace_native_step.py" \
  --qualified-smoke "$TASK_EVIDENCE/final_namespace_smoke" \
  --output "$TASK_EVIDENCE/final_namespace_native_step"
