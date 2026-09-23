"""Remaining synthetic production-cam controls, sequential hard-limited jobs."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cgroup',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent;rows=[]
for variant,seed in [('positive',s) for s in [23,47,83,131]]+[('broken_linkage',11)]:
 folder=a.output/(variant+'_'+str(seed));command=['sudo','-n',sys.executable,str(root/'limited_run.py'),'--cgroup',a.cgroup,'--receipt',str(folder)+'.limits.json','--',sys.executable,str(root/'cam_linkage.py'),'--output',str(folder),'--variant',variant,'--seed',str(seed),'--run-solver','--solver-python',a.solver_python]
 with Path(str(folder)+'.log').open('w') as log:proc=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 row=json.loads((folder/'qualification.json').read_text()) if (folder/'qualification.json').is_file() else {'qualified':False};rows.append(dict(row,returncode=proc.returncode,path=str(folder)));print(json.dumps(rows[-1]),flush=True)
(a.output/'summary.json').write_text(json.dumps({'qualified':all(r['qualified'] for r in rows),'rows':rows},indent=2)+'\n')
