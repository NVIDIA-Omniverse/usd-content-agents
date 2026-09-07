#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
destination="$repo_root/.data/examples/scene-material-pass"
revision=0ed0dfbc539c9de99289771bd6848effe3ef5779
base_url="https://media.githubusercontent.com/media/NVIDIA/simready-foundation/$revision"
raw_base_url="https://raw.githubusercontent.com/NVIDIA/simready-foundation/$revision"
manifest="$repo_root/agentic/examples/scene/material-pass/assets.sha256"
scene="$destination/mini_workcell.usda"

mkdir -p "$destination"
rm -f "$scene"
while read -r expected relative; do
  [[ -n "$expected" ]] || continue
  target="$destination/$relative"
  if [[ -f "$target" ]] &&
    printf '%s  %s\n' "$expected" "$target" | sha256sum --check --status; then
    continue
  fi
  source_path="${relative#simready-foundation/}"
  url="$base_url/$source_path"
  [[ "$relative" == *.mdl || "$relative" == *.txt ]] &&
    url="$raw_base_url/$source_path"
  mkdir -p "$(dirname -- "$target")"
  curl --fail --location --silent --show-error "$url" --output "$target.part"
  printf '%s  %s\n' "$expected" "$target.part" | sha256sum --check --status || {
    rm -f "$target.part"
    echo "SimReady asset checksum mismatch: $relative" >&2
    exit 1
  }
  mv "$target.part" "$target"
done < "$manifest"

cp "$repo_root/agentic/examples/scene/material-pass/mini_workcell.usda" \
  "$scene"
echo "$scene"
