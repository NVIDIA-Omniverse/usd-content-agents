"""Use supported native usd-cli USDZ save to package Geometry's texture closure."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
from content_agent_workflows.geometry.lossless_gltf import verify_preserved_render

root = Path('/opt/astra-content-value-20260921')
cap = root/'capstone'
run = cap/'runs/drawer_packaging_04'
run.mkdir(exist_ok=False)
source = cap/'runs/drawer_geometry_04/source_preserved.geometry.usdc'
config = run/'.usd-cli/config.toml'
config.parent.mkdir()
config.write_text('[server]\nallowed_roots = '+json.dumps([str(cap)])+'\nallowed_write_roots = '+json.dumps([str(run)])+'\n[render]\nrenderer = "ovrtx"\novrtx_auto_install = false\n')
env = dict(os.environ, PATH=str(cap/'repo/.venv/bin')+':/home/horde/.local/bin:/usr/local/bin:/usr/bin:/bin', CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'], TMPDIR=str(root/'tmp'), WU_OVRTX_VENV_DIR=str(root/'ovrtx-venv'), WU_OVPHYSX_VENV_DIR=str(root/'ovphysx-venv'), WU_OVPHYSX_AUTO_PROVISION='0', USD_CLI_TEL_BACKENDS='file', USD_CLI_TEL_FILE=str(run/'packaging_operations.jsonl'))
cli = cap/'repo/.venv/bin/usd-cli-tel'
records=[]
try:
 for name, command in [
  ('open_geometry.json',['open',str(source)]),
  ('save_package.json',['save',str(run/'source.usdz'),'--flatten']),
  ('open_package.json',['open',str(run/'source.usdz')]),
  ('inspection_snapshot.json',['snapshot','--bounds','--properties']),
  ('packaged_render.json',['render','--focus','/Asset','--res','1024x1024','--mode','fast','--output',str(run/'packaged_preview.png')]),
 ]:
  start=datetime.datetime.now(datetime.timezone.utc).isoformat()
  proc=subprocess.run([str(cli),'--json',*command],cwd=run,env=env,capture_output=True,text=True)
  (run/name).write_text(proc.stdout);(run/(name+'.stderr')).write_text(proc.stderr)
  records.append({'command':command,'returncode':proc.returncode,'start_utc':start,'end_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
  if proc.returncode:raise RuntimeError(f'Native packaging failed: {name}: {proc.stderr[-1500:]}')
 fidelity=verify_preserved_render(source,run/'source.usdz')
 (run/'packaging_fidelity.json').write_text(json.dumps({'source':str(source),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'package_sha256':hashlib.sha256((run/'source.usdz').read_bytes()).hexdigest(),'source_render_fidelity':fidelity,'operation':'native usd-cli save USDZ --flatten; no geometry edits'},indent=2)+'\n')
finally:
 proc=subprocess.run([str(cli),'--json','server','stop'],cwd=run,env=env,capture_output=True,text=True)
 (run/'server_stop.json').write_text(proc.stdout)
 records.append({'command':['server','stop'],'returncode':proc.returncode})
 (run/'operations.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
