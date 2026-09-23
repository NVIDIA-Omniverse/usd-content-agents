"""Run the real native source/package verifier without a model, solver or asset."""
import argparse
import dataclasses
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

def inside():
    from content_agent_workflows.common import usd_cli
    assert os.getuid()==20000
    result={'schema_version':'native-source-verifier.v2','uid':os.getuid(),'model_calls':0,
      'solver_calls':0,'render_calls':0,'verifier_sha256':hashlib.sha256(Path(usd_cli.__file__).read_bytes()).hexdigest(),'aliases':[]}
    for name in ('/tools/repo','/workflows'):
        root=Path(name)
        env=usd_cli.sanitized_git_verification_env(trusted_repository=root)
        git=subprocess.run(usd_cli.git_verification_command('rev-parse','--show-toplevel'),cwd=root,env=env,text=True,capture_output=True)
        row={'repository':name,'root_owner':root.stat().st_uid,'git_owner':(root/'.git').stat().st_uid,
          'git_returncode':git.returncode,'git_stdout':git.stdout.strip(),'git_stderr':git.stderr.strip(),
          'ambient_git_safe_config_removed':not any(k.startswith('GIT_CONFIG_KEY_') for k in env),
          'global_git_config_disabled':env.get('GIT_CONFIG_GLOBAL')==os.devnull,
          'system_git_config_disabled':env.get('GIT_CONFIG_NOSYSTEM')=='1'}
        try:
            with (root/'.qualification-must-not-write').open('x') as f:f.write('unreachable')
            row['owner_cannot_write_readonly_mount']=False
        except OSError as exc:
            row['owner_cannot_write_readonly_mount']=exc.errno==errno.EROFS
        start=time.monotonic()
        try:
            route=usd_cli.resolve_package_owned_usd_cli_route(root)
            row.update(status='PASS',route={k:str(v) for k,v in dataclasses.asdict(route).items()})
            from content_workflow_cli.runner import _probe_usd_cli_telemetry_wrapper
            row['telemetry_version_probe_error']=_probe_usd_cli_telemetry_wrapper(route.wrapper,route.target)
            if row['telemetry_version_probe_error'] is not None:row['status']='FAIL'
        except Exception as exc:
            row.update(status='FAIL',error=type(exc).__name__+': '+str(exc))
            if exc.__cause__ is not None:
                row['cause']=type(exc.__cause__).__name__+': '+str(exc.__cause__)
                stderr=getattr(exc.__cause__,'stderr',None)
                if stderr:row['cause_stderr']=stderr
        row['elapsed_seconds']=time.monotonic()-start;result['aliases'].append(row)
    Path('/evidence/receipt.json').write_text(json.dumps(result,indent=2)+'\n')

def run(a):
    assert os.geteuid()==0
    sys.path.insert(0,a.candidate_controller)
    import evaluate_run as candidate
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    group=Path(json.loads(Path(a.cgroups).read_text())['lanes'][1]['path'])
    assert candidate.populated(group)==0,'Lane1 occupied'
    paths={}
    for name in ('workspace','source','original','evaluator','evidence'):
        paths[name]=out/name;paths[name].mkdir(mode=0o755)
    os.chown(paths['evidence'],20000,20000)
    shutil.copyfile(__file__,paths['evaluator']/'probe.py')
    paths['tools']=Path(a.tools)
    cfg={'tools':a.tools,'rootfs':a.rootfs,'system_mounts':[p for p in (
      '/usr','/lib','/lib64','/etc/ld.so.cache','/etc/ssl/certs','/etc/fonts','/etc/vulkan','/etc/alternatives') if Path(p).exists()]}
    argv=candidate.namespace_command(cfg,paths,['/tools/main-venv/bin/python','-B','/evaluator/probe.py','--inside'],'/evaluator',20000)
    i=argv.index('--')
    argv[i:i]=['--ro-bind',str(Path(a.tools)/'repo'),'/workflows',
      '--setenv','GIT_CONFIG_COUNT','1','--setenv','GIT_CONFIG_KEY_0','safe.directory',
      '--setenv','GIT_CONFIG_VALUE_0','/tools/repo']
    cpus=sorted(os.sched_getaffinity(0))[4:8];assert len(cpus)==4
    def enter():
        (group/'cgroup.procs').write_text(str(os.getpid()));os.sched_setaffinity(0,cpus)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with (out/'stdout.log').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
        proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,preexec_fn=enter,env={'PATH':'/usr/bin:/bin'})
        try:proc.wait(timeout=300)
        finally:candidate.kill_and_reap(group);proc.wait(timeout=10)
    summary={'returncode':proc.returncode,'cgroup_populated':candidate.populated(group),
      'model_calls':0,'solver_calls':0,'render_calls':0,'candidate_sha256':candidate.sha256(candidate.__file__),
      'probe_sha256':candidate.sha256(__file__)}
    (out/'supervisor.json').write_text(json.dumps(summary,indent=2)+'\n')
    print((out/'evidence/receipt.json').read_text() if (out/'evidence/receipt.json').exists() else json.dumps(summary))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true')
    for key in ('tools','rootfs','cgroups','output','candidate-controller'):p.add_argument('--'+key)
    a=p.parse_args();inside() if a.inside else run(a)
