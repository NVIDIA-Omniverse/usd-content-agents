#!/usr/bin/env python3
"""Create-only Linux namespace fixture; no model/native renderer/solver.

Uses explicit read-only system libraries plus an empty synthetic rootfs. This
qualifies kernel setup, not the later exported matched runtime or GPU access.
"""
import argparse
import json
import os
import secrets
import subprocess
from pathlib import Path

from common import require, sha256, write_new

INSIDE = r'''
import json,os,subprocess,sys
from pathlib import Path
c=json.loads(Path('/work/config.json').read_text())
s=dict(x.split(':',1) for x in Path('/proc/self/status').read_text().splitlines() if ':' in x)
foreign=False
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try: foreign|=c['canary'].encode() in p.read_bytes()
 except (OSError,ProcessLookupError): pass
try: Path(c['foreign_file']).read_text(); unreadable=False
except OSError: unreadable=True
write_refused=False
try: Path('/source/original.txt').write_text('forbidden mutation')
except OSError as e: write_refused=e.errno==30
mount=subprocess.run(['/usr/bin/mount','-t','tmpfs','none','/work/mount'],capture_output=True)
print(json.dumps({'pid_ns':os.readlink('/proc/self/ns/pid'),'net_ns':os.readlink('/proc/self/ns/net'),
 'uid':os.geteuid(),'caps':{k:s[k].strip() for k in ('CapEff','CapPrm','CapBnd','CapAmb','NoNewPrivs')},
 'foreign_cmdline_absent':not foreign,'foreign_file_unreadable':unreadable,'source_write_EROFS':write_refused,
 'source_readonly':bool(os.statvfs('/source').f_flag & os.ST_RDONLY),
 'mount_refused':mount.returncode!=0,'visible_proc_count':len(list(Path('/proc').glob('[0-9]*')))}))
'''


def run(output):
    require(os.geteuid() == 0, "Root required for qualification")
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    root = out / "rootfs"; root.mkdir()
    for name in ("usr", "lib", "lib64", "bin", "etc", "proc", "dev", "tmp", "source", "work", "home", "home/agent"):
        (root / name).mkdir(exist_ok=True)
    for name in ("work", "source", "foreign"):
        (out / name).mkdir()
    (out / "work/mount").mkdir()
    uid = 20000
    os.chown(out / "work", uid, uid)
    os.chmod(out / "work", 0o700)
    original = out / "source/original.txt"; original.write_text("synthetic source\n"); original.chmod(0o666)
    before = sha256(original)
    canary = "foreign-command-" + secrets.token_hex(12)
    foreign_file = out / "foreign/private.txt"; foreign_file.write_text("synthetic foreign bytes\n")
    config = {"canary": canary, "foreign_file": str(foreign_file)}
    write_new(out / "work/config.json", config)
    (out / "work/probe.py").write_text(INSIDE)
    original_ns = {k: os.readlink("/proc/self/ns/" + k) for k in ("pid", "net")}
    foreign = subprocess.Popen(["/usr/bin/python3", "-c", "import time; time.sleep(60)", canary])
    argv = ["/usr/bin/bwrap", "--unshare-pid", "--unshare-net", "--unshare-ipc", "--unshare-uts", "--unshare-cgroup", "--die-with-parent", "--new-session",
            "--ro-bind", str(root), "/", "--ro-bind", "/usr", "/usr", "--ro-bind", "/lib", "/lib", "--ro-bind", "/lib64", "/lib64",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", str(out / "work"), "/work", "--ro-bind", str(out / "source"), "/source",
            "--clearenv", "--setenv", "PATH", "/usr/bin", "--setenv", "HOME", "/home/agent", "--chdir", "/work", "--",
            "/usr/bin/setpriv", "--reuid", str(uid), "--regid", str(uid), "--clear-groups", "--bounding-set=-all", "--inh-caps=-all", "--ambient-caps=-all", "--no-new-privs", "--",
            "/usr/bin/python3", "-B", "/work/probe.py"]
    try:
        completed = subprocess.run(argv, text=True, capture_output=True, timeout=30)
    finally:
        foreign.terminate(); foreign.wait(timeout=5)
    result = {"schema_version": "kernel-smoke.partial.v2", "argv": argv, "returncode": completed.returncode,
              "stdout": completed.stdout, "stderr": completed.stderr, "host_namespace": original_ns,
              "script_sha256": sha256(__file__), "source_unchanged": sha256(original) == before,
              "scope": "Synthetic namespace only; no model, GPU, resource-pressure or matched runtime qualification"}
    if completed.returncode == 0:
        actual = json.loads(completed.stdout)
        checks = {"private_pid_namespace": actual["pid_ns"] != original_ns["pid"], "private_net_namespace": actual["net_ns"] != original_ns["net"],
                  "nonroot_no_capabilities": actual["uid"] == uid and all(int(actual["caps"][x], 16) == 0 for x in ("CapEff", "CapPrm", "CapBnd", "CapAmb")),
                  "no_new_privileges": actual["caps"]["NoNewPrivs"] == "1", "foreign_cmdline_unreadable": actual["foreign_cmdline_absent"],
                  "foreign_files_unreadable": actual["foreign_file_unreadable"], "source_readonly": actual["source_readonly"] and actual["source_write_EROFS"],
                  "mount_escape_refused": actual["mount_refused"]}
        result["checks"] = checks; result["partial_pass"] = all(checks.values()) and result["source_unchanged"]
    else:
        result["partial_pass"] = False
    write_new(out / "receipt.json", result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); a = p.parse_args()
    print(json.dumps(run(a.output)))
