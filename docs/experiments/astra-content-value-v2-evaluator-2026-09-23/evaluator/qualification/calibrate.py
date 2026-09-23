"""Sequential independent analytic controls inside the delegated cgroup."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python',required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
for f in (ROOT/'calibration_sources').iterdir():shutil.copy2(f,a.output/f.name)
rows=[]
for name,python,result,key in [('case04_torque_units_probe.py',sys.executable,'case04_torque_units_evidence.json','pass'),('case04_contact_units_probe.py',sys.executable,'case04_contact_units_evidence.json','matrix_is_impulse'),('drawer_force_units_probe.py',a.solver_python,'force_units_evidence.json','pass'),('gravity_witness_v2.py',a.solver_python,'gravity_v2/summary.json','qualified')]:
 with (a.output/(name+'.log')).open('w') as log:proc=subprocess.run([python,str(a.output/name)],stdout=log,stderr=subprocess.STDOUT,timeout=600)
 report=json.loads((a.output/result).read_text()) if (a.output/result).is_file() else {}
 rows.append({'script':name,'returncode':proc.returncode,'qualified':proc.returncode==0 and report.get(key) is True,'evidence':result})
(a.output/'summary.json').write_text(json.dumps({'qualified':all(r['qualified'] for r in rows),'rows':rows,'model_calls':0},indent=2)+'\n');print(json.dumps(rows));sys.exit(0 if all(r['qualified'] for r in rows) else 1)
