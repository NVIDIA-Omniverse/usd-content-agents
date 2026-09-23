"""Trusted pre-score qualification limiter; NOT scored-author isolation."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import resource
import pwd
import subprocess
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--receipt',type=Path,required=True);p.add_argument('--cgroup',type=Path);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    cmd=a.command[1:] if a.command[:1]==['--'] else a.command
    if not cmd:raise ValueError('A trusted qualification argument list is required')
    affinity=sorted(os.sched_getaffinity(0))[:4]
    if a.cgroup:
        expected={'cpu.max':'400000 100000','memory.max':str(32*1024**3),'memory.swap.max':'0','pids.max':'512'}
        if a.cgroup.name!='astra-v2-lane0':raise ValueError('Only the root-delegated free astra-v2-lane0 is permitted')
        for name,value in expected.items():
            if (a.cgroup/name).read_text().strip()!=value:raise ValueError('Unexpected cgroup bound: '+name)
        if (a.cgroup/'cgroup.procs').read_text().strip():raise ValueError('Evaluator lane is already occupied')
        account=pwd.getpwnam('horde')
    def constrain():
        os.sched_setaffinity(0,affinity)
        if a.cgroup:
            (a.cgroup/'cgroup.procs').write_text(str(os.getpid()))
            os.setgroups([]);os.setgid(account.pw_gid);os.setuid(account.pw_uid)
        else:
            resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
        os.nice(15)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    tick=time.monotonic();start=datetime.now(timezone.utc).isoformat()
    result=subprocess.run(cmd,env=env,preexec_fn=constrain)
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    record={'schema_version':1,'scope':'Trusted synthetic qualification only; no scored author isolation claim',
            'started_at':start,'completed_at':datetime.now(timezone.utc).isoformat(),'elapsed_s':time.monotonic()-tick,
            'returncode':result.returncode,'cpu_affinity':affinity,'threads_per_numerical_library':1,'nice':15,
            'rlimit_as_per_process_bytes':None if a.cgroup else 8*1024**3,'aggregate_cgroup_memory_enforced':bool(a.cgroup),
            'cgroup_path':str(a.cgroup) if a.cgroup else None,
            'child_max_rss_kib':usage.ru_maxrss,'child_user_cpu_s':usage.ru_utime,'child_system_cpu_s':usage.ru_stime,
            'model_calls':0}
    a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(record,indent=2)+'\n')
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
