import datetime,json,os,subprocess
from pathlib import Path
cap=Path('/opt/astra-content-value-20260921/capstone');r=cap/'runs/drawer_packaging_04';publication=cap/'runs/articulation_preparation_03';py=cap/'repo/.venv/bin/python';cli=cap/'repo/.venv/bin/content-workflow-cli'
assert not publication.exists();records=[]
commands=[
 [str(cli),'articulation','publish-preparation','--readback',str(r/'articulation_preparation_readback.json'),'--retained-root',str(r),'--output-dir',str(publication)],
 [str(cli),'articulation','validate-preparation','--publication',str(publication/'articulation_preparation_publication.json')]]
for i,c in enumerate(commands):
 x={'command':c,'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};proc=subprocess.run(c,capture_output=True,text=True,cwd=cap/'repo');(cap/'evidence'/f'joint03_preparation_retry01_{i}.json').write_text(proc.stdout);(cap/'evidence'/f'joint03_preparation_retry01_{i}.stderr').write_text(proc.stderr);x.update(returncode=proc.returncode,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());records.append(x)
 print(json.dumps(x),flush=True)
 if proc.returncode:break
(cap/'evidence/joint03_preparation_retry01_execution.json').write_text(json.dumps(records,indent=2)+'\n');raise SystemExit(records[-1]['returncode'])
