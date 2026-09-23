#!/usr/bin/env python3
"""Actual isolated Astra/native-child preflight on one synthetic cube only.

Controller invokes explicitly. No scored protocol, source asset, prior solution,
or evaluator is accepted by this interface. A fresh home is never resumed.
"""
import argparse
import json
import os
import resource
import subprocess
import time
from pathlib import Path

from common import MODEL, EFFORT, read, require, sha256, write_new
from isolation import bubblewrap_command, kill_and_reap, populated
from run_arm import mount_run_storage, prepare_runtime_caches, prepare_runtime_tmp, provider_config, qualified_environment, write_dummy_auth, write_native_provider
from ui_audit import Capture
from network_health import verify as verify_network

SOURCE = '''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "World" {
    def Cube "Cube" {
        double size = 1
        color3f[] primvars:displayColor = [(0.1, 0.55, 0.85)]
    }
}
'''


def run(request, tools, boundary_launch, rootfs, output):
    require(os.geteuid() == 0, 'Trusted root supervisor required')
    require(request.get('scope') == 'unscored_synthetic_namespace', 'Qualification scope required')
    require(type(request.get('wall_seconds')) is int and 1 <= request['wall_seconds'] <= 1200, 'Bounded synthetic time required')
    require(isinstance(request.get('prompt'), str) and request['prompt'], 'Controller synthetic prompt required')
    boundary = read(boundary_launch)
    require(boundary.get('scored') is False and boundary.get('model_calls') == 0, 'Use the native GPU fixture boundary')
    require(sha256(Path(tools)/'environment.json') == boundary['environment_sha256'], 'Qualified common environment changed')
    group = Path(boundary['cgroup']); require(populated(group)==0, 'Lane occupied')
    network_health = verify_network(boundary['namespace'], request['bridge_receipt'])
    out=Path(output); out.mkdir(parents=True,exist_ok=False,mode=0o700)
    uid=20000
    for name in ('workspace','home','private','source','input'):(out/name).mkdir(mode=0o700)
    volumes=mount_run_storage(out,uid,{'workspace_bytes':8*2**30,'home_bytes':2*2**30})
    write_new(out/'private/storage.json',{'volumes':volumes,'synthetic':True})
    (out/'source/cube.usda').write_text(SOURCE)
    (out/'source').chmod(0o755);(out/'source/cube.usda').chmod(0o444)
    (out/'input/task.json').write_text('{"synthetic":true,"source":"/source/cube.usda"}\n')
    (out/'input').chmod(0o755);(out/'input/task.json').chmod(0o444)
    home=out/'home';ch=home/'.codex';ch.mkdir(mode=0o700);os.chown(ch,uid,uid)
    runtime_tmp=prepare_runtime_tmp(home,uid)
    (ch/'config.toml').write_text(provider_config('http://127.0.0.1:18862/v1'));os.chown(ch/'config.toml',uid,uid)
    dummy_auth=write_dummy_auth(ch,request['token'],uid)
    native_provider=write_native_provider(ch,'http://127.0.0.1:18862/v1',uid)
    locks={}
    for name in ('ovphysx-venv.provision.lock','ovrtx-venv.provision.lock'):
        p=home/name;p.touch(mode=0o600);os.chown(p,uid,uid);locks[name]=str(p)
    mapping=boundary['gpu_mapping']
    paths={'rootfs':str(rootfs),'tools':str(tools),'workflow':str(Path(tools)/'repo'),
           'workspace':str(out/'workspace'),'home':str(home),'source':str(out/'source'),'input':str(out/'input'),
           'gpu_devices':mapping['device_nodes'],'runtime_locks':locks,
           'runtime_caches':prepare_runtime_caches(tools,home,uid),
           'system_mounts':[p for p in ('/usr','/lib','/lib64','/etc/ld.so.cache','/etc/ssl/certs','/etc/vulkan','/etc/alternatives','/etc/fonts') if Path(p).exists()]}
    env=qualified_environment(tools,boundary['environment_sha256'],{'gpu_uuid':mapping['uuid']},request['token'])
    command=['/tools/bin/codex','exec','--skip-git-repo-check','-C','/work','-m',MODEL,
             '-c',f'model_reasoning_effort="{EFFORT}"','-s','danger-full-access','--json','-o','/work/answer.txt','-']
    fd=os.open(boundary['namespace'],os.O_RDONLY)
    argv=bubblewrap_command(paths,'content_agents',{'gpu_uuid':mapping['uuid']},uid,fd,command,env)
    capture=Capture([home,out/'workspace'],out/'private/session_captures',uid)
    # No token is retained in this launch receipt or argv.
    write_new(out/'private/launch.json',{'scored':False,'scope':request['scope'],'lane_id':request.get('lane_id','qualification'),
              'wall_seconds':request['wall_seconds'],'boundary_launch_sha256':sha256(boundary_launch),
              'environment_sha256':boundary['environment_sha256'],'source_sha256':sha256(out/'source/cube.usda'),
              'prompt_sha256':__import__('hashlib').sha256(request['prompt'].encode()).hexdigest(),'model':MODEL,'ui_effort':EFFORT,
              'dummy_auth_state':dummy_auth,
              'native_temporary_directory':runtime_tmp,
              'native_provider_config':native_provider,
              'network_health':network_health,
              'harness_files':{p.name:sha256(p)for p in Path(__file__).parent.glob('*.py')}})
    began=time.time();p=None;timeout=False
    try:
        def enter():
            (group/'cgroup.procs').write_text(str(os.getpid()));os.sched_setaffinity(0,boundary['cpu_affinity'])
            resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        with (out/'private/events.jsonl').open('x')as log,(out/'private/stderr.log').open('x')as err:
            p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=log,stderr=err,text=True,pass_fds=(fd,),preexec_fn=enter,env={'PATH':'/usr/bin:/bin'})
            capture.start()
            try:p.communicate(request['prompt'],timeout=request['wall_seconds'])
            except subprocess.TimeoutExpired:timeout=True;kill_and_reap(group);p.wait(timeout=15)
    finally:
        os.close(fd);kill_and_reap(group)
        if p:p.wait(timeout=15)
        write_new(out/'private/ui_effort_audit.json',capture.stop())
    result={'schema_version':'synthetic-namespace-qualification.v2','scored':False,'source_assets_used':False,
            'returncode':p.returncode,'timed_out':timeout,'elapsed_seconds':time.time()-began,'cgroup_populated':populated(group),
            'source_unchanged':(out/'source/cube.usda').read_text()==SOURCE,
            'launch_sha256':sha256(out/'private/launch.json'),'model_accounting':'Requires independent gateway ledger reconciliation',
            'native_workflow_compatibility':'Requires actual retained native receipts; not inferred from model claim or return code'}
    write_new(out/'private/reap.json',result);return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--tools',required=True)
    p.add_argument('--boundary-launch',required=True);p.add_argument('--rootfs',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(run(read(a.request),a.tools,a.boundary_launch,a.rootfs,a.output)))
