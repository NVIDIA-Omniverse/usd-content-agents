#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Link compatibility skill paths to the canonical agent skills tree.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./scripts/sync_agent_skills.sh [--check] [--force]

Links .claude/skills and .codex/skills to .agents/skills.

Options:
  --check   Verify compatibility skill paths are symlinks to .agents/skills.
  --force   Replace dirty legacy mirror directories. Use only after moving any
            mirror-only skill work into .agents/skills.
  -h, --help
            Show this help message.
EOF
}

mode="sync"
force=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --check)
            mode="check"
            shift
            ;;
        --force)
            force=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR/.." rev-parse --show-toplevel 2>/dev/null)"; then
    :
else
    REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi

CANONICAL="$REPO_ROOT/.agents/skills"
MIRRORS=(
    "$REPO_ROOT/.claude/skills"
    "$REPO_ROOT/.codex/skills"
)
LINK_TARGET="../.agents/skills"

if [[ ! -d "$CANONICAL" ]]; then
    echo "ERROR: canonical skills directory not found: $CANONICAL" >&2
    exit 1
fi

ensure_clean_mirror() {
    local mirror="$1"
    local mirror_rel="${mirror#"$REPO_ROOT"/}"
    local status

    if [[ "$force" == true ]]; then
        return
    fi
    if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        return
    fi

    status="$(git -C "$REPO_ROOT" status \
        --porcelain \
        --untracked-files=all \
        -- "$mirror_rel")"
    if [[ -n "$status" ]]; then
        if [[ -L "$mirror" ]] && [[ "$(realpath "$mirror")" == "$(realpath "$CANONICAL")" ]]; then
            return
        fi
        echo "ERROR: refusing to replace dirty skill mirror: $mirror_rel" >&2
        echo "$status" >&2
        echo >&2
        echo "Move mirror-only skill changes into .agents/skills/ first, then rerun." >&2
        echo "Use --force only when the mirror changes are safe to discard." >&2
        exit 1
    fi
}

if [[ "$mode" == "check" ]]; then
    ok=true
    for mirror in "${MIRRORS[@]}"; do
        if [[ ! -L "$mirror" ]]; then
            echo "ERROR: compatibility skills path is not a symlink: $mirror" >&2
            ok=false
            continue
        fi
        if [[ "$(realpath "$mirror")" != "$(realpath "$CANONICAL")" ]]; then
            echo "ERROR: compatibility skills path does not point to $CANONICAL: $mirror" >&2
            ok=false
        fi
    done
    if [[ "$ok" == true ]]; then
        exit 0
    fi
    exit 1
fi

for mirror in "${MIRRORS[@]}"; do
    ensure_clean_mirror "$mirror"
    rm -rf "$mirror"
    mkdir -p "$(dirname "$mirror")"
    ln -s "$LINK_TARGET" "$mirror"
done

echo "Linked .claude/skills and .codex/skills to .agents/skills."
