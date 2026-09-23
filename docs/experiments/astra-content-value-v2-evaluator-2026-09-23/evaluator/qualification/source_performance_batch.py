"""Full original09 source checks under actual hard limits; no native solver."""
import argparse,json,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parent;rows=[]
jobs=[('usd_build_read',[sys.executable,str(root/'source_stage.py'),'--output',str(a.output/'usd_build_read')]),('usd_fresh_read',[sys.executable,str(root/'source_stage.py'),'--existing-source-only-scene',str(a.output/'usd_build_read/original_source_visual_only.usdc'),'--output',str(a.output/'usd_fresh_read')]),('retessellated_all_parts',[sys.executable,str(root/'source_roundtrip.py'),'--case','09_printer','--mode','subdivide','--output',str(a.output/'retessellated_all_parts')])]
for name,command in jobs:
 t=time.monotonic()
 with (a.output/(name+'.log')).open('w') as log:proc=subprocess.run(['/usr/bin/timeout','--kill-after=5s','900s']+command,stdout=log,stderr=subprocess.STDOUT)
 path=a.output/name/'result.json';result=json.loads(path.read_text()) if path.is_file() else {}
 row={'name':name,'returncode':proc.returncode,'elapsed_s':time.monotonic()-t,'wall_cap_s':900,'qualified':proc.returncode==0 and result.get('passed') is True,'result':str(path),'disposition':'completed_source_measurement' if proc.returncode==0 else 'INCONCLUSIVE_SOURCE_QUALIFICATION'};rows.append(row);print(json.dumps(row),flush=True)
 (a.output/'summary.json').write_text(json.dumps({'complete':False,'rows':rows},indent=2)+'\n')
(a.output/'summary.json').write_text(json.dumps({'complete':True,'qualified':all(r['qualified'] for r in rows),'rows':rows,'native_solver_calls':0,'model_calls':0},indent=2)+'\n');sys.exit(0 if all(r['qualified'] for r in rows) else 1)
