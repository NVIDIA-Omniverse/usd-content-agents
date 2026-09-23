"""Shared, dependency-free v2 harness primitives. No evaluator imports."""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path

MODEL = "gpt-6-astra"
EFFORT = "ultra"
ARMS = ("plain_astra", "content_agents")
SEEDS = [11, 23, 47, 83, 131]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def object_hash(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def identifier(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,95}", value), "Unsafe identifier")
    return value


def digest_string(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Expected SHA256")
    return value


def inventory(root, *, internal_symlinks=False, output_tree=False):
    """Hash a closed tree. Never follow symlinks or accept devices/sockets."""
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Expected a real directory")
    root = root.resolve()
    files = []
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in sorted(dirs + names):
            p = Path(parent) / name
            mode = p.lstat().st_mode
            rel = p.relative_to(root).as_posix()
            if stat.S_ISLNK(mode):
                target = os.readlink(p)
                if output_tree:
                    # Preserve link text only. Unrelated helper links must not
                    # invalidate otherwise closed submitted scene/bindings.
                    # Never read their target or admit them as final files.
                    files.append({"path": rel, "kind": "symlink", "target": target,
                                  "excluded_from_submission": True})
                    continue
                require(internal_symlinks and not os.path.isabs(target), f"Forbidden symlink: {rel}")
                require(p.resolve().is_relative_to(root), f"Escaping symlink: {rel}")
                require(p.exists(), f"Dangling symlink: {rel}")
                files.append({"path": rel, "kind": "symlink", "target": target})
            elif stat.S_ISREG(mode):
                require(output_tree or p.stat().st_nlink == 1, f"External hardlink not permitted: {rel}")
                files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p)})
            else:
                require(output_tree or stat.S_ISDIR(mode), f"Special file forbidden: {rel}")
                if not stat.S_ISDIR(mode):
                    files.append({"path": rel, "kind": "nonportable_special_file", "mode": stat.S_IFMT(mode), "excluded_from_submission": True})
    return {"files": sorted(files, key=lambda x: x["path"])}


def closed_file(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), "Expected relative file")
    path = Path(root) / relative
    require(".." not in Path(relative).parts, "Path traversal")
    current = Path(root)
    for piece in Path(relative).parts:
        current /= piece
        require(not current.is_symlink(), "Symlink in submitted path")
    require(path.is_file() and path.resolve().is_relative_to(Path(root).resolve()), "File outside run")
    require(path.stat().st_nlink == 1, "Hardlinked submission")
    return path


def validate_protocol(p):
    require(p.get("schema_version") == "isolated-lanes.v2", "Unknown protocol schema")
    require(p.get("frozen") is True, "Protocol is not frozen")
    require(p.get("model") == MODEL and p.get("reasoning_effort") == EFFORT, "Astra Ultra required")
    require(p.get("codex_version") == "0.154.0" and p.get("wire_reasoning_effort") == "xhigh", "Qualified Codex Ultra wire mapping required")
    digest_string(p["reasoning_mapping_sha256"])
    digest_string(p["model_catalog_sha256"])
    require(p.get("resource_rule") == "four_exclusive_gpu_lanes", "Wrong resource rule")
    require(p.get("seeds") == SEEDS, "Seed mismatch")
    require(type(p.get("wall_seconds")) is int and 1 <= p["wall_seconds"] <= 2400, "Invalid wall budget")
    require("repair_limit" in p and p["repair_limit"] is None, "V2 measures repairs; fixed wall time is the budget")
    require(p.get("repair_definition") == "measured_not_capped", "Repair convention must be explicit")
    require(p.get("network_policy") == "model_gateway_only", "Unqualified author network")
    require(p.get("fresh_context") is True, "Fresh contexts are mandatory")
    require(p.get("storage") == {"kind": "tmpfs", "workspace_bytes": 8 * 2**30,
            "home_bytes": 2 * 2**30, "charged_within_memory_limit": True}, "Matched bounded tmpfs storage required")
    lanes = p.get("lanes", [])
    require(len(lanes) == 4, "Exactly four GPU lanes required")
    for lane in lanes:
        identifier(lane["id"])
        identifier(lane["host"])
        require(re.fullmatch(r"GPU-[0-9a-fA-F-]+", lane["gpu_uuid"]), "GPU UUID required; no index fallback")
        require(type(lane["cpu_millicores"]) is int and 1 <= lane["cpu_millicores"] <= 4000, "CPU cap exceeds allocation")
        require(type(lane["memory_bytes"]) is int and 1 <= lane["memory_bytes"] <= 32 * 2**30, "RAM cap exceeds allocation")
        require(type(lane["pids_max"]) is int and 8 <= lane["pids_max"] <= 4096, "Bounded process/thread cap required")
        require(isinstance(lane.get('cpu_affinity'),list) and len(lane['cpu_affinity'])==4
                and all(type(x)is int and x>=0 for x in lane['cpu_affinity'])
                and lane['cpu_affinity']==sorted(set(lane['cpu_affinity'])), 'Four exact CPU IDs required')
        digest_string(lane["qualification_sha256"])
    require(len({x["id"] for x in lanes}) == 4, "Duplicate lane")
    require(len({x["gpu_uuid"] for x in lanes}) == 4, "A GPU cannot back two lanes")
    resources = {(x["cpu_millicores"], x["memory_bytes"], x["pids_max"]) for x in lanes}
    require(len(resources) == 1, "Lane resource budgets differ")
    require(all(a['host']!=b['host'] or not set(a['cpu_affinity'])&set(b['cpu_affinity'])
                for i,a in enumerate(lanes) for b in lanes[i+1:]), 'Same-host author cpusets must be disjoint')
    for key in ("common_rootfs_sha256", "common_tools_sha256", "common_environment_sha256", "workflow_sha256", "model_gateway_policy_sha256"):
        digest_string(p[key])
    return p


def validate_pair(a, b):
    require({a["arm"], b["arm"]} == set(ARMS), "Incomplete paired arms")
    for key in ("case_id", "task_sha256", "input_sha256", "source_sha256", "protocol_sha256", "lane_id"):
        require(a[key] == b[key], f"Paired {key} differs")
    return True
