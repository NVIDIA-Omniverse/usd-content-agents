#!/usr/bin/env python3
"""Trusted worker supervisor. No scoring, evaluator loading, or model fallback.

Input is a controller-produced JSON request on stdin. Config and evidence stay
outside all author mounts. Execute only after the controller issues a lane lease.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common import (ARMS, EFFORT, MODEL, identifier, inventory, object_hash, read,
                    require, sha256, validate_protocol, write_new)
from isolation import (REQUIRED_WITNESSES, RUNTIME_CACHE_TARGETS, bubblewrap_command, check_host_owned,
                       check_limits, kill_and_reap, populated, validate_mounts)
from ui_audit import Capture
from finalize import inspect as inspect_delivery
from network_health import verify as verify_network


def prompt(task, arm, wall):
    common = f"""Solve the frozen source asset task within {wall} seconds, including all tools and child-model work. Every model call must use {MODEL} with {EFFORT} reasoning. Start a fresh context; no prior attempts or solutions are available. Read original source at /source and common tools/documentation at /tools. Work under /work. Installed tools and original inputs are read-only. Native children share your assigned GPU lane and its fixed CPU, RAM and process/thread limits; do not try to escape that boundary.
Preserve original source geometry and assembly identity. Author genuine force-driven physics for the stated task. The independent evaluator runs after your immutable submission; you cannot access it or its results during authoring. Record initial candidate, repair outputs and repair chronology; repairs are measured, not capped. No human assistance occurs during the measured run.
All model subprocesses must use the run gateway and explicit --model gpt-6-astra --model-reasoning-effort ultra settings where supported. A secret-free provider configuration is available at /home/agent/.codex/native_provider.json. For native commands exposing --codex-config-file (including Physics apply and Validation run), pass that file; inspect the command's --help and do not assume Geometry's responses-url/api-key-env flags exist on other commands. Native subprocesses inherit the run-only ASTRA_GATEWAY_TOKEN; no upstream credentials are available.
The canonical installed repository is /tools/repo. Whenever a permitted command takes --repo-root, use /tools/repo so package and Git provenance resolve to the same path. /workflows is an additional read-only documentation alias; do not use that alias as --repo-root. This path guidance does not change the arm's workflow-use policy.
Task and public contract (identical across arms):
{json.dumps(task, sort_keys=True, indent=2)}
Write submission.json with claimed_accepted boolean, final_scene and bindings as relative paths or null, reason string, repairs_used nonnegative integer, and limitations list. Preserve partial output and report false when blocked. Do not substitute another model, change packages or source, or claim that rendering/schema checks prove physical acceptance.
"""
    treatment = ("Use the common low-level tools directly. The installed implementation and common documentation are identical across arms, but this control must not invoke Content Agents workflows, decision helpers or workflow skills (including Python import-and-call routes). Inspecting shared low-level APIs is allowed; record inspection separately from workflow invocation. Baseline workflow non-use is independently audited, not a claim that common implementation files are physically absent."
                 if arm == "plain_astra" else
                 "Use the frozen public Content Agents workflows at /workflows and their native preparation, authoring, review, application and validation entrypoints. Preserve failed/conditional outcomes and native receipts. All workflow children must explicitly select Astra Ultra; do not replace a failed native workflow with an ad hoc solution and label it native success.")
    return common + "\n" + treatment + "\n"


def provider_config(base_url):
    require(base_url.startswith("http://") and base_url.endswith("/v1") and '"' not in base_url and "\n" not in base_url,
            "Expected qualified private model gateway URL")
    return f'''model = "{MODEL}"
model_reasoning_effort = "{EFFORT}"
model_provider = "trial_gateway"
approval_policy = "never"
model_catalog_json = "/tools/astra_model_catalog.json"
[model_providers.trial_gateway]
name = "Run-scoped Astra gateway"
base_url = "{base_url}"
env_key = "ASTRA_GATEWAY_TOKEN"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
'''


def qualified_environment(tools, expected_sha256, lane, token):
    path = Path(tools) / "environment.json"
    require(sha256(path) == expected_sha256, "Qualified runtime environment changed")
    environment = read(path)
    require(isinstance(environment, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()),
            "Runtime environment must contain string fields")
    private = {"HOME": "/home/agent", "CODEX_HOME": "/home/agent/.codex", "LANG": "C.UTF-8",
               "TMPDIR": "/home/agent/.runtime_tmp",
               "CUDA_VISIBLE_DEVICES": lane["gpu_uuid"], "ASTRA_GATEWAY_TOKEN": token,
               "CONTENT_AGENT_CODEX_RESPONSES_URL": "http://127.0.0.1:18862/v1/responses",
               "CONTENT_AGENT_CODEX_API_KEY_ENV": "ASTRA_GATEWAY_TOKEN",
               "CONTENT_AGENT_MODEL": MODEL, "USD_CLI_SESSION": "own-run"}
    require(not set(environment) & {"ASTRA_GATEWAY_TOKEN", "OPENAI_API_KEY", "CODEX_HOME", "HOME"},
            "Common environment must not contain private run fields")
    return environment | private


def prepare_runtime_tmp(home, uid):
    """Retain supported native bridge staging inside the existing home budget."""
    path = Path(home) / '.runtime_tmp'
    path.mkdir(mode=0o700)
    os.chown(path, uid, uid)
    return {'path': '/home/agent/.runtime_tmp', 'mode': '0700', 'author_uid': uid,
            'storage': 'Existing 2GiB home tmpfs; no additional allocation',
            'capture': 'Existing trusted home session capture observes temporary native Codex homes'}


def write_dummy_auth(codex_home, token, uid):
    path=Path(codex_home)/'auth.json'
    with path.open('x')as f:json.dump({'auth_mode':'apikey','OPENAI_API_KEY':token},f)
    path.chmod(0o600);os.chown(path,uid,uid)
    return {'auth_mode':'apikey','credential_field':'OPENAI_API_KEY',
            'credential_semantics':'This run gateway token only; no upstream bearer/account authority',
            'mode':'0600','author_uid':uid}


def write_native_provider(codex_home, base_url, uid):
    require(base_url == 'http://127.0.0.1:18862/v1', 'Only the qualified private gateway is supported')
    value={'model_provider':'trial_gateway','model_providers':{'trial_gateway':{
        'name':'Run-scoped Astra gateway','base_url':base_url,'env_key':'ASTRA_GATEWAY_TOKEN',
        'wire_api':'responses','requires_openai_auth':False,'supports_websockets':False,
        'request_max_retries':0,'stream_max_retries':0}}}
    path=Path(codex_home)/'native_provider.json'
    write_new(path,value);path.chmod(0o600);os.chown(path,uid,uid)
    return {'path':'/home/agent/.codex/native_provider.json','sha256':sha256(path),
            'contains_secret_values':False,'scope':'Supported native --codex-config-file provider route; explicit model/Ultra flags still required'}


def mount_run_storage(out, uid, storage):
    """Controller-owned tmpfs mounts; author pages share its memory cgroup cap."""
    records = []
    for name, key in (("workspace", "workspace_bytes"), ("home", "home_bytes")):
        path = Path(out) / name
        require(path.is_dir() and not path.is_symlink() and not any(path.iterdir()), "Fresh empty storage target required")
        require(not os.path.ismount(path), "Storage target already mounted")
        size = storage[key]
        options = f"size={size},mode=0700,uid={uid},gid={uid},nosuid,nodev"
        subprocess.run(["/usr/bin/mount", "-t", "tmpfs", "-o", options, "astra-v2-" + name, str(path)], check=True)
        # Never unmount automatically on later failure: preserve partial data
        # until a controller has retained and verified this exact run.
        require(os.path.ismount(path), "Tmpfs storage mount was not established")
        vfs = os.statvfs(path)
        require(vfs.f_blocks * vfs.f_frsize == size, "Tmpfs capacity differs from protocol")
        rows = [line for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4] == str(path)]
        require(len(rows) == 1 and ' - tmpfs ' in rows[0], "Unexpected storage filesystem")
        records.append({"path": str(path), "kind": "tmpfs", "size_bytes": size, "mountinfo": rows[0],
                        "author_pages_charged_within_lane_memory_cap": True})
    return records


def prepare_runtime_caches(tools, home, uid):
    cache_root = Path(home) / ".runtime-caches"
    cache_root.mkdir(mode=0o700); os.chown(cache_root, uid, uid)
    result = {}
    for i, target in enumerate(RUNTIME_CACHE_TARGETS):
        original = Path(tools) / target.removeprefix('/tools/')
        require(original.is_dir() and not original.is_symlink(), "Missing qualified native cache target")
        path = cache_root / str(i)
        path.mkdir(mode=0o700); os.chown(path,uid,uid)
        if target.endswith('/bin/cache'):
            require({p.name for p in original.iterdir()} == {'shadercache'}, "Unexpected bundled cache layout")
            # The 826 MiB vendor shader seed remains a nested read-only bind.
            # Only generated sibling caches consume the author's home budget.
            (path/'shadercache').mkdir(); os.chown(path/'shadercache',uid,uid)
        else:
            require(not any(original.iterdir()), "Generated MDL extension target must start empty")
        result[target] = str(path)
    return result


def execute(config, request):
    require(sys.platform == "linux" and os.geteuid() == 0, "Qualified Linux root supervisor required")
    protocol = validate_protocol(read(config["protocol_file"]))
    require(sha256(config["protocol_file"]) == request["protocol_sha256"], "Frozen protocol changed")
    run_id = identifier(request["run_id"])
    lane = next((x for x in protocol["lanes"] if x["id"] == request["lane_id"]), None)
    require(lane is not None and lane["host"] == config["host_id"], "Wrong worker lane")
    require(request["arm"] in ARMS, "Unknown arm")
    require(isinstance(request.get("lease"), str) and len(request["lease"]) == 32, "Controller lease required")
    qualification_file = config["lanes"][lane["id"]]["qualification_file"]
    require(sha256(qualification_file) == lane["qualification_sha256"], "Qualification changed")
    q = read(qualification_file)
    require(q.get("status") == "PASS" and q.get("host_id") == config["host_id"] and q.get("lane_id") == lane["id"], "Wrong runtime qualification")
    require(all(q.get("checks", {}).get(k) is True for k in REQUIRED_WITNESSES), "Runtime boundary not fully qualified")
    for key in ("common_rootfs_sha256", "common_tools_sha256", "common_environment_sha256", "workflow_sha256", "model_gateway_policy_sha256"):
        require(q.get(key) == protocol[key], f"Unqualified {key}")
    base = Path(config["runs_root"])
    require(base.is_absolute() and base.is_dir(), "Missing trusted run root")
    out = base / run_id
    require(not out.exists(), "Run directory already exists; resume forbidden")
    task_file = Path(request["task_file"])
    input_dir = Path(request["input_dir"])
    require(task_file.parent.resolve() == input_dir.resolve() and task_file.name == "task.json", "Task must be inside its dedicated input directory")
    public_inputs = inventory(input_dir)
    require(object_hash(public_inputs) == request["input_sha256"], "Public input closure mismatch")
    require({x["path"] for x in public_inputs["files"]} == {"task.json", request["case_id"] + "_source_inventory.json"}, "Only this case task and source inventory may be mounted")
    require(sha256(task_file) == request["task_sha256"], "Task mismatch")
    task = read(task_file)
    require(task.get("frozen") is True and task.get("case_id") == request["case_id"], "Task must be frozen and matched")
    source = Path(request["source"])
    source_inventory = inventory(source)
    require(object_hash(source_inventory) == request["source_sha256"], "Source closure mismatch")
    require(sha256(Path(config["tools"]) / "astra_model_catalog.json") == protocol["model_catalog_sha256"], "Model catalog changed")
    environment = qualified_environment(config["tools"], protocol["common_environment_sha256"], lane, request["gateway_token"])
    lane_config = config["lanes"][lane["id"]]
    limits = check_limits(lane_config["cgroup"], lane)
    require(populated(lane_config["cgroup"]) == 0, "Lane already contains processes")
    network_health = verify_network(lane_config['network_namespace'], lane_config['bridge_receipt'])
    out.mkdir(mode=0o700)
    for name in ("workspace", "home", "private"):
        (out / name).mkdir(mode=0o700)
    # Dedicated UID sees only its own empty home and workspace. The real auth,
    # upstream routes, controller DB, launch/evidence are outside these mounts.
    uid = lane_config["uid"]
    storage_volumes = mount_run_storage(out, uid, protocol["storage"])
    write_new(out / "private" / "storage.json", {"volumes": storage_volumes, "policy": protocol["storage"],
              "cleanup": "Retain and verify full run before unmounting these exact two directories."})
    for name in ("workspace", "home"):
        os.chown(out / name, uid, uid)
    home = out / "home"
    runtime_tmp = prepare_runtime_tmp(home, uid)
    (home / ".codex").mkdir(mode=0o700)
    os.chown(home / ".codex", uid, uid)
    (home / ".codex" / "config.toml").write_text(provider_config(config["gateway_base_url"]))
    os.chown(home / ".codex" / "config.toml", uid, uid)
    dummy_auth=write_dummy_auth(home/'.codex',request['gateway_token'],uid)
    native_provider=write_native_provider(home/'.codex',config['gateway_base_url'],uid)
    lock_dir = home / ".runtime-locks"; lock_dir.mkdir(mode=0o700); os.chown(lock_dir, uid, uid)
    runtime_locks = {}
    for name in ("ovphysx-venv.provision.lock", "ovrtx-venv.provision.lock"):
        p = lock_dir / name; p.touch(mode=0o600); os.chown(p, uid, uid)
        runtime_locks[name] = str(p)
    paths = {"rootfs": config["rootfs"], "tools": config["tools"], "workflow": config["workflow"],
             "source": str(source), "task": str(task_file), "input": str(input_dir), "workspace": str(out / "workspace"), "home": str(home),
             "network_namespace": lane_config["network_namespace"], "cgroup": lane_config["cgroup"], "gpu_devices": lane_config["gpu_devices"]}
    paths["system_mounts"] = config.get("system_mounts", [])
    paths["runtime_locks"] = runtime_locks
    paths["runtime_caches"] = prepare_runtime_caches(config["tools"], home, uid)
    validate_mounts(paths, request["arm"])
    command = ["/tools/bin/codex", "exec", "--skip-git-repo-check", "-C", "/work", "-m", MODEL,
               "-c", f'model_reasoning_effort="{EFFORT}"', "-c", 'approval_policy="never"',
               "-s", "danger-full-access", "--json", "-o", "/work/last_message.txt", "-"]
    text = prompt(task, request["arm"], protocol["wall_seconds"])
    write_new(out / "private" / "launch.json", {k: request[k] for k in request if k != "gateway_token"} |
              {"prompt_sha256": object_hash(text), "limits": limits, "model": MODEL, "effort": EFFORT,
               "fresh_home": True, "source_inventory": source_inventory,
               "common_environment_sha256": protocol["common_environment_sha256"],
               "dummy_auth_state":dummy_auth,
               "native_temporary_directory":runtime_tmp,
               "native_provider_config":native_provider,
               "network_health":network_health,
               "effective_environment_without_token": {k: v for k, v in environment.items() if k != "ASTRA_GATEWAY_TOKEN"}})
    netfd = os.open(paths["network_namespace"], os.O_RDONLY)
    argv = bubblewrap_command(paths, request["arm"], lane, uid, netfd, command, environment)
    timed_out = False
    proc = None
    started = time.time()
    began = time.monotonic()
    capture = Capture([home, out / "workspace"], out / "private" / "session_captures", uid)
    try:
        def enter_cgroup():
            (Path(paths["cgroup"]) / "cgroup.procs").write_text(str(os.getpid()))
            os.sched_setaffinity(0,lane['cpu_affinity'])
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        with (out / "private" / "events.jsonl").open("x") as stdout, (out / "private" / "stderr.log").open("x") as stderr:
            proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
                                    env={"PATH": "/usr/bin:/bin"}, pass_fds=(netfd,), preexec_fn=enter_cgroup)
            # Start the observer after fork so preexec_fn never runs from a
            # multithreaded supervisor. No model prompt has been sent yet.
            capture.start()
            try:
                remaining = min(protocol["wall_seconds"], request["deadline_unix"] - time.time())
                require(remaining > 0, "Allocation expired during startup; no author work permitted")
                proc.communicate(text, timeout=remaining)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Kills every member, including setsid/double-fork descendants.
                kill_and_reap(paths["cgroup"])
                proc.wait(timeout=15)
    finally:
        os.close(netfd)
        kill_and_reap(paths["cgroup"])
        if proc is not None:
            proc.wait(timeout=15)
        ui_audit = capture.stop()
        write_new(out / "private" / "ui_effort_audit.json", ui_audit)
    elapsed = time.monotonic() - began
    output_error = None
    try:
        output_inventory = inventory(out / "workspace", internal_symlinks=True, output_tree=True)
    except (OSError, ValueError) as exc:
        # Malformed author artifacts cannot erase the independently established
        # reap proof or hold all subsequent cases hostage. Never score these as
        # a complete immutable submission without an explicit artifact audit.
        output_inventory = None
        output_error = type(exc).__name__
    source_unchanged = inventory(source) == source_inventory
    receipt = {"schema_version": "worker-reap.v2", "run_id": run_id, "lease": request["lease"], "lane_id": lane["id"],
               "started_unix": started, "elapsed_seconds": elapsed, "timed_out": timed_out, "returncode": proc.returncode,
               "namespace_init_exited": True, "cgroup_populated": populated(paths["cgroup"]), "source_unchanged": source_unchanged,
               "output_inventory_sha256": object_hash(output_inventory), "protocol_sha256": request["protocol_sha256"],
               "output_snapshot_complete": output_inventory is not None, "output_snapshot_error": output_error,
               "qualification_sha256": lane["qualification_sha256"], "scoring": "pending independent accounting, protocol audit and physics evaluator"}
    write_new(out / "private" / "output_inventory.json", output_inventory)
    write_new(out / "private" / "reap.json", receipt)
    write_new(out / "private" / "author_delivery.json", inspect_delivery(out))
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    check_host_owned(args.config)
    print(json.dumps(execute(read(args.config), json.load(sys.stdin)), sort_keys=True))
