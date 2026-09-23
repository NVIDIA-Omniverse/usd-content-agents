#!/usr/bin/env python3
"""Add hard four-CPU cpusets only to our two empty author lane children."""
import argparse
import os
from pathlib import Path
from common import read, require, write_new
from isolation import populated


def configure(receipt, output):
    require(os.geteuid()==0,'Root required')
    lanes=[Path(x['path']) for x in read(receipt)['lanes']]
    require(len(lanes)==2 and lanes[0].parent==lanes[1].parent,'Two own-container lanes required')
    parent=lanes[0].parent
    current=Path('/sys/fs/cgroup')/next(x[3:]for x in Path('/proc/self/cgroup').read_text().splitlines()if x.startswith('0::')).lstrip('/')
    require(current==parent/'astra-v2-supervisor' and parent.name.startswith('cri-containerd-'),'Refuse host/sibling cgroup')
    require(all(p.name==f'astra-v2-lane{i}' and populated(p)==0 for i,p in enumerate(lanes)),'Expected empty author lanes')
    require('cpuset' in (parent/'cgroup.controllers').read_text().split(),'cpuset delegation unavailable')
    available=sorted(os.sched_getaffinity(0));require(len(available)>=8,'Eight observed allowed CPU IDs required')
    inherited={key:(parent/key).read_text().strip() for key in ('cpuset.cpus','cpuset.mems','cpuset.cpus.effective','cpuset.mems.effective')}
    # Some kernels reject delegation from an implicit empty parent cpuset.
    # Materialize exactly its current inherited sets, never an ancestor/sibling
    # or additional CPU/NUMA allocation.
    for key in ('cpuset.mems','cpuset.cpus'):
        if not inherited[key]:(parent/key).write_text(inherited[key+'.effective'])
    (parent/'cgroup.subtree_control').write_text('+cpuset')
    result=[]
    for i,p in enumerate(lanes):
        chosen=available[4*i:4*i+4]
        (p/'cpuset.cpus').write_text(','.join(map(str,chosen)))
        result.append({'path':str(p),'cpu_affinity':chosen,'cpuset_cpus_effective':(p/'cpuset.cpus.effective').read_text().strip(),
                       'cpuset_mems_effective':(p/'cpuset.mems.effective').read_text().strip(),
                       'cpu_max_unchanged':(p/'cpu.max').read_text().strip()})
    value={'schema_version':'author-lane-affinity.v2','observed_parent_affinity':available,'parent_cpuset_before':inherited,'lanes':result,
           'source_assets_touched':False,'native_jobs_launched':False,'cpu_memory_process_quotas_increased':False}
    write_new(output,value);return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cgroup-receipt',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(__import__('json').dumps(configure(a.cgroup_receipt,a.output)))
