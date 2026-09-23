#!/usr/bin/env python3
"""Trusted, bounded GPU fixture supervisor. One lane; no author/model task."""
import argparse
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

from common import read, require, sha256, write_new
from isolation import bubblewrap_command, kill_and_reap, populated
from run_arm import prepare_runtime_caches


def run(tools, gpu_index, cgroup_receipt, output, sandbox_only=False):
    require(os.geteuid() == 0, "Root required")
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    tools = Path(tools)
    gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id", "--format=csv,noheader"], text=True)
    entries = [dict(zip(("index", "uuid", "pci"), (part.strip() for part in line.split(",")))) for line in gpu_rows.splitlines()]
    entry = next(x for x in entries if int(x["index"]) == gpu_index)
    minor = None
    for path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
        fields = dict((a.strip(),b.strip()) for a,b in (line.split(":",1) for line in path.read_text().splitlines() if ":" in line))
        if fields.get("GPU UUID") == entry["uuid"]: minor = int(fields["Device Minor"])
    require(minor is not None, "Could not bind UUID to actual device minor")
    devices = [f"/dev/nvidia{minor}"]
    for path in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset"):
        if Path(path).exists(): devices.append(path)
    # No DRM nodes are assumed. Include only those whose PCI device is this GPU.
    pci = entry["pci"].lower().replace("00000000:", "0000:")
    for path in Path("/sys/class/drm").glob("*"):
        if (path/"device").exists() and (path/"device").resolve().name.lower() == pci and Path("/dev/dri", path.name).exists():
            devices.append("/dev/dri/" + path.name)
    rootfs = out / "rootfs"; rootfs.mkdir()
    for name in ("usr", "lib", "lib64", "bin", "etc", "etc/ssl", "etc/ssl/certs", "etc/vulkan", "etc/alternatives", "etc/fonts", "proc", "dev", "tmp", "source", "work", "tools", "workflows", "input", "home", "home/agent"):
        (rootfs/name).mkdir(exist_ok=True)
    (rootfs/"etc/ld.so.cache").touch()
    if Path('/etc/os-release').exists():shutil.copyfile('/etc/os-release',rootfs/'etc/os-release')
    (rootfs/'etc/passwd').write_text('agent:x:20000:20000:Author:/home/agent:/bin/sh\n')
    (rootfs/'etc/group').write_text('agent:x:20000:\n')
    (rootfs/"bin/sh").symlink_to("/usr/bin/sh")
    for name in ("workspace", "source", "home", "input"):
        (out/name).mkdir(); os.chown(out/name,20000,20000)
    shutil.copyfile(Path(__file__).with_name("gpu_probe.py"),out/"workspace/gpu_probe.py")
    (out/"input/task.json").write_text('{"synthetic":true}\n')
    locks={}
    for name in ("ovphysx-venv.provision.lock","ovrtx-venv.provision.lock"):
        path=out/"home"/name;path.touch();os.chown(path,20000,20000);locks[name]=str(path)
    group=Path(read(cgroup_receipt)["lanes"][gpu_index]["path"])
    require(populated(group)==0,"Qualification lane is occupied")
    available=sorted(os.sched_getaffinity(0))
    require(len(available)>=8,'Eight observed allowed CPU IDs required')
    cpus=available[gpu_index*4:gpu_index*4+4]
    namespace=f"/run/netns/astra-v2-lane{gpu_index}"
    paths={"rootfs":str(rootfs),"tools":str(tools),"workflow":str(tools/"repo"),"source":str(out/"source"),"task":str(out/"input/task.json"),"input":str(out/"input"),
           "workspace":str(out/"workspace"),"home":str(out/"home"),"gpu_devices":devices,"runtime_locks":locks,
           "system_mounts":[p for p in ("/usr","/lib","/lib64","/etc/ld.so.cache","/etc/ssl/certs","/etc/vulkan","/etc/alternatives","/etc/fonts") if Path(p).exists()]}
    paths['runtime_caches']=prepare_runtime_caches(tools,out/'home',20000)
    env=read(tools/"environment.json")
    env.update(HOME="/home/agent",CODEX_HOME="/home/agent/.codex",CUDA_VISIBLE_DEVICES=entry["uuid"],USD_CLI_SESSION="gpu-qualification")
    netfd=os.open(namespace,os.O_RDONLY)
    argv=bubblewrap_command(paths,"plain_astra",{"gpu_uuid":entry["uuid"]},20000,netfd,["/tools/main-venv/bin/python","-B","/work/gpu_probe.py","--assigned",entry["uuid"],"--sandbox-only" if sandbox_only else "--render"],env)
    write_new(out/"launch.json",{"argv":argv,"gpu_mapping":entry|{"minor":minor,"device_nodes":devices},"environment_sha256":sha256(tools/"environment.json"),"probe_sha256":sha256(out/"workspace/gpu_probe.py"),"cgroup":str(group),"cpu_affinity":cpus,"affinity_enforcement":"Initial inherited runtime setting; hard CPU budget is cpu.max, not cpuset containment","namespace":namespace,"scored":False,"model_calls":0})
    started=time.monotonic();p=None
    try:
        def enter():
            (group/"cgroup.procs").write_text(str(os.getpid()))
            os.sched_setaffinity(0,cpus)
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        with (out/"stdout.log").open("x") as stdout,(out/"stderr.log").open("x") as stderr:
            p=subprocess.Popen(argv,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL,pass_fds=(netfd,),preexec_fn=enter,env={"PATH":"/usr/bin:/bin"})
            p.wait(timeout=700)
    finally:
        os.close(netfd);kill_and_reap(group)
        if p:p.wait(timeout=15)
    result={"schema_version":"gpu-qualification-supervisor.v2","returncode":p.returncode,"elapsed_seconds":time.monotonic()-started,"cgroup_populated":populated(group),"launch_sha256":sha256(out/"launch.json"),"model_calls":0,"scored":False}
    receipt=out/"workspace/gpu_receipt.json"
    if receipt.exists():result.update(probe_sha256=sha256(receipt),passed=read(receipt)["passed"])
    else:result["passed"]=False
    write_new(out/"receipt.json",result);return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--tools",required=True);p.add_argument("--gpu-index",type=int,choices=[0,1],required=True);p.add_argument("--cgroup-receipt",required=True);p.add_argument("--output",required=True);p.add_argument('--sandbox-only',action='store_true');a=p.parse_args()
    print(json.dumps(run(a.tools,a.gpu_index,a.cgroup_receipt,a.output,a.sandbox_only)))
