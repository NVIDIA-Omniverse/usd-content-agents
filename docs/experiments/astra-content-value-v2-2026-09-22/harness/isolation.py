"""Linux worker boundary, invoked only by the trusted root supervisor.

The root filesystem is curated, not a bind of the worker host. A separate
setpriv step drops UID/groups/capabilities before ANY author code executes.
"""
from __future__ import annotations

import os
import re
import stat
import time
from pathlib import Path

from common import require

RUNTIME_CACHE_TARGETS = (
    "/tools/ovrtx-venv/lib/python3.12/site-packages/ovrtx/bin/cache",
    "/tools/ovrtx-venv/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts",
)

REQUIRED_WITNESSES = (
    "private_pid_namespace", "foreign_cmdline_unreadable", "foreign_files_unreadable",
    "source_readonly", "tools_readonly", "common_implementation_identical_both_arms",
    "workflow_activation_matches_treatment",
    "fresh_home_no_history", "nonroot_no_capabilities", "no_new_privileges",
    "mount_escape_refused", "only_assigned_gpu_accessible", "network_gateway_only",
    "foreign_gateway_token_refused", "model_effort_gate", "primary_child_requests_accounted",
    "cpu_limit_enforced", "memory_limit_enforced", "process_limit_enforced",
    "orphan_killed_and_reaped", "native_child_workflow_compatible",
    "assigned_ovrtx_frame", "other_gpu_compute_and_vulkan_refused",
)


def cgroup_values(path):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "Missing lane cgroup")
    return {name: (path / name).read_text().strip() for name in
            ("cpu.max", "memory.max", "memory.swap.max", "pids.max", "cgroup.events")}


def check_limits(path, lane):
    values = cgroup_values(path)
    quota, period = values["cpu.max"].split()
    require(quota != "max" and int(quota) * 1000 == lane["cpu_millicores"] * int(period), "CPU quota differs from frozen lane")
    require(values["memory.max"] == str(lane["memory_bytes"]), "RAM cap differs from frozen lane")
    require(values["memory.swap.max"] == "0", "Swap must be disabled per lane")
    require(values["pids.max"] == str(lane["pids_max"]), "Process/thread cap differs from frozen lane")
    require(set(lane['cpu_affinity']) <= os.sched_getaffinity(0), 'Initial CPU affinity unavailable')
    values['initial_cpu_affinity'] = lane['cpu_affinity']
    values['affinity_enforcement'] = 'Initial inherited setting; may be widened within the unchanged hard cpu.max budget'
    return values


def cpu_set(value):
    result=[]
    for part in value.split(','):
        bounds=part.split('-'); result.extend(range(int(bounds[0]), int(bounds[-1])+1))
    return sorted(set(result))


def populated(path):
    return int(dict(line.split() for line in (Path(path) / "cgroup.events").read_text().splitlines())["populated"])


def kill_and_reap(path, timeout=15):
    path = Path(path)
    if populated(path):
        (path / "cgroup.kill").write_text("1")
    end = time.monotonic() + timeout
    while populated(path) and time.monotonic() < end:
        time.sleep(0.05)
    require(populated(path) == 0, "Lane remains populated; never release its lease")


def validate_mounts(paths, arm):
    required = ("rootfs", "tools", "source", "workspace", "home", "input", "task", "network_namespace", "cgroup")
    for name in required:
        p = Path(paths[name])
        require(p.is_absolute() and p.exists() and not p.is_symlink(), f"Invalid {name} path")
    dirs = [Path(paths[k]).resolve() for k in ("rootfs", "tools", "source", "workspace", "home")]
    require(Path(paths["workflow"]).is_dir() and not Path(paths["workflow"]).is_symlink(), "Missing common workflow code")
    workflow = Path(paths["workflow"]).resolve()
    if not workflow.is_relative_to(Path(paths["tools"]).resolve()):
        dirs.append(workflow)
    require(all(a != b and not a.is_relative_to(b) and not b.is_relative_to(a) for i, a in enumerate(dirs) for b in dirs[i + 1:]),
            "Mount trees must be disjoint; never mount a parent containing other runs")
    rootfs = Path(paths["rootfs"])
    for unsafe in ("home/horde", "root/.codex", "root/.ssh", "var/run/docker.sock", "opt/astra-content-value-20260921", "evaluator"):
        require(not (rootfs / unsafe).exists(), f"Host/private content in rootfs: {unsafe}")


def bubblewrap_command(paths, arm, lane, uid, netns_fd, command, environment):
    require(type(uid) is int and uid >= 10000, "Dedicated unprivileged UID required")
    require(type(netns_fd) is int and netns_fd >= 3, "Open private network namespace FD required")
    require(environment.get("CUDA_VISIBLE_DEVICES") == lane["gpu_uuid"], "GPU binding mismatch")
    # Root bwrap does not use --uid without --unshare-user: setpriv works on
    # workers that permit root PID/mount namespaces but disallow nested userns.
    argv = ["/usr/bin/nsenter", "--net=/proc/self/fd/" + str(netns_fd), "--", "/usr/bin/bwrap",
            "--unshare-pid", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup", "--die-with-parent", "--new-session",
            "--ro-bind", paths["rootfs"], "/", "--proc", "/proc", "--dev", "/dev",
            "--perms", "1777", "--tmpfs", "/dev/shm", "--perms", "1777", "--tmpfs", "/tmp",
            "--bind", paths["workspace"], "/work", "--bind", paths["home"], "/home/agent",
            "--ro-bind", paths["source"], "/source", "--ro-bind", paths["tools"], "/tools",
            "--ro-bind", paths["input"], "/input"]
    for path in paths.get("system_mounts", []):
        require(path in ("/usr", "/lib", "/lib64", "/etc/ld.so.cache", "/etc/ssl/certs", "/etc/fonts", "/etc/vulkan", "/etc/alternatives"), "Unqualified system bind")
        argv += ["--ro-bind", path, path]
    for name, source in sorted(paths.get("runtime_locks", {}).items()):
        require(name in ("ovphysx-venv.provision.lock", "ovrtx-venv.provision.lock"), "Unqualified writable runtime file")
        p = Path(source)
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(Path(paths["home"]).resolve()), "Runtime lock must be an own-home file")
        argv += ["--bind", source, "/tools/" + name]
    for target, source in sorted(paths.get("runtime_caches", {}).items()):
        require(target in RUNTIME_CACHE_TARGETS, "Unqualified writable runtime directory")
        p = Path(source)
        require(p.is_dir() and not p.is_symlink() and p.resolve().is_relative_to(Path(paths["home"]).resolve()), "Runtime cache must be an own-home directory")
        argv += ["--bind", source, target]
        if target.endswith('/bin/cache'):
            seed = Path(paths['tools']) / target.removeprefix('/tools/') / 'shadercache'
            require(seed.is_dir() and not seed.is_symlink(), "Missing bundled shader seed")
            argv += ['--ro-bind', str(seed), target + '/shadercache']
    argv += ["--ro-bind", paths["workflow"], "/workflows"]
    devices = paths["gpu_devices"]
    require(isinstance(devices, list) and devices, "GPU device allowlist required")
    require(len([x for x in devices if re.fullmatch(r"/dev/nvidia[0-9]+", x)]) == 1, "Exactly one physical GPU device required")
    for device in devices:
        require(re.fullmatch(r"(?:/dev/nvidia(?:[0-9]+|ctl|-uvm|-uvm-tools|-modeset|-caps/nvidia-cap[0-9]+)|/dev/dri/(?:card[0-9]+|renderD[0-9]+))", device), "Unqualified device mount")
        argv += ["--dev-bind", device, device]
    argv += ["--clearenv", "--hostname", "author", "--chdir", "/work"]
    allowed_env = {"PATH", "HOME", "CODEX_HOME", "TMPDIR", "LANG", "LC_ALL", "CUDA_VISIBLE_DEVICES", "UV_NO_CACHE",
                   "WU_OVRTX_VENV_DIR", "WU_OVPHYSX_VENV_DIR", "WU_OVRTX_AUTO_PROVISION", "WU_OVPHYSX_AUTO_PROVISION",
                   "WU_SO_PACKAGE_DIR", "USD_CLI_UV_EXECUTABLE", "CONTENT_AGENT_MODEL",
                   "ASTRA_GATEWAY_TOKEN", "OPENAI_BASE_URL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                   "PYTHONPATH", "LD_LIBRARY_PATH", "USD_CLI_SESSION"}
    allowed_env |= {"WU_SO_PYTHON", "GEOMETRY_REPAIR_GEOGRAM_EXECUTABLE_SHA256", "GIT_CONFIG_COUNT", "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0", "PIP_NO_CACHE_DIR", "PYTHONDONTWRITEBYTECODE"}
    allowed_env |= {"CONTENT_AGENT_CODEX_RESPONSES_URL", "CONTENT_AGENT_CODEX_API_KEY_ENV"}
    require(set(environment) <= allowed_env, "Unexpected inherited environment")
    for key, value in sorted(environment.items()):
        argv += ["--setenv", key, str(value)]
    # Root-only setup ends here. No-new-privs prevents a setuid binary or file
    # capability in the tool image from recovering host privileges.
    argv += ["--", "/usr/bin/setpriv", "--reuid", str(uid), "--regid", str(uid), "--clear-groups",
             "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs", "--", *command]
    return argv


def check_host_owned(path):
    p = Path(path)
    require(p.stat().st_uid == 0 and not (p.stat().st_mode & (stat.S_IWGRP | stat.S_IWOTH)), "Trusted configuration must be root-owned and not group/world writable")
