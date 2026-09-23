"""Sequential trusted native qualification in one root-delegated cgroup."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--cgroup',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--solver-python',required=True);p.add_argument('--case',action='append');p.add_argument('--seed',action='append',type=int);p.add_argument('--negatives',action='store_true');a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
    cases=a.case or ['02_conveyor','03_hinge','04_gripper','05_vise','06_engine','07_robot_arm','08_excavator','09_printer','10_complex']
    rows=[]
    for case in cases:
        jobs=[('positive',s) for s in (a.seed or [11,23,47,83,131])]
        if a.negatives:
            variants={'02_conveyor':['blocked_motion','no_contact'],'04_gripper':['blocked_motion','one_contact','no_friction'],'09_printer':['blocked_motion','no_contact']}.get(case,['blocked_motion'])
            jobs.extend((v,11) for v in variants)
        for variant,seed in jobs:
            folder=a.output/f'{case}_{variant}_{seed}'
            command=['sudo','-n',sys.executable,str(root/'qualification/limited_run.py'),'--cgroup',str(a.cgroup),'--receipt',str(folder)+'.limits.json','--',sys.executable,str(root/'qualification/native_qualify.py'),'--case',case,'--variant',variant,'--output',str(folder),'--run-solver','--solver-python',a.solver_python,'--seed',str(seed)]
            with Path(str(folder)+'.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            record=json.loads((folder/'qualification.json').read_text()) if (folder/'qualification.json').is_file() else {}
            row={'case_id':case,'variant':variant,'seed':seed,'returncode':result.returncode,'qualified':record.get('qualified',False),'native_run':record.get('native_run',False),'native_accepted':record.get('native_accepted'), 'concrete_failures':record.get('concrete_failures'),'disposition':record.get('disposition'),'path':str(folder)}
            rows.append(row);(a.output/'summary.json').write_text(json.dumps({'complete':False,'rows':rows},indent=2)+'\n');print(json.dumps(row),flush=True)
    (a.output/'summary.json').write_text(json.dumps({'complete':True,'all_qualified':all(r['qualified'] for r in rows),'rows':rows},indent=2)+'\n')
    raise SystemExit(0 if all(r['qualified'] for r in rows) else 1)


if __name__=='__main__':main()
