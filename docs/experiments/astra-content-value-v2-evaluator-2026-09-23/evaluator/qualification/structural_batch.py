"""Final synthetic USD preparation and rejection controls, no native solver."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
for case in sorted(x.name for x in (ROOT/'cases').iterdir() if x.is_dir()):
 variants=['positive']+([] if case in ['02_conveyor','04_gripper'] else ['kinematic','missing_joint','changed_geometry','gravity_disabled'])
 for variant in variants:
  folder=a.output/(case+'_'+variant)
  with Path(str(folder)+'.log').open('w') as log:proc=subprocess.run([sys.executable,str(ROOT/'qualification/native_qualify.py'),'--case',case,'--variant',variant,'--output',str(folder)],stdout=log,stderr=subprocess.STDOUT,timeout=120)
  record=json.loads((folder/'qualification.json').read_text());qualified=record.get('preparation_passed') if variant=='positive' else record['qualified'];rows.append({'case_id':case,'variant':variant,'qualified':bool(qualified) and proc.returncode==0,'native_run':record['native_run'],'path':str(folder)})
(a.output/'summary.json').write_text(json.dumps({'qualified':all(r['qualified'] for r in rows),'rows':rows},indent=2)+'\n');print(json.dumps({'qualified':all(r['qualified'] for r in rows),'count':len(rows)}));sys.exit(0 if all(r['qualified'] for r in rows) else 1)
