#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git -C "$(dirname -- "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
destination="$repo_root/.data/examples/toycar/ToyCar.glb"
revision=0e3a605bda7c758293ab58432f1d51a2a355d47a
expected=01a60862de55cd4b9f3acfab0b0def86451800f9c42467fcd61052c16cb9838c
url="https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/$revision/Models/ToyCar/glTF-Binary/ToyCar.glb"

mkdir -p "$(dirname -- "$destination")"
curl --fail --location "$url" --output "$destination.part"
printf '%s  %s\n' "$expected" "$destination.part" | sha256sum --check --status || {
  rm -f "$destination.part"
  echo "ToyCar checksum mismatch" >&2
  exit 1
}
mv "$destination.part" "$destination"
echo "$destination"
