"""Run inside the actual /tools read-only export, without models or GPU jobs."""
import hashlib, importlib, importlib.metadata as metadata, json, os, subprocess, sys, time
from pathlib import Path

def main():
 out=Path('/work'); checks={};errors=[];started=time.time();details={}
 def check(name,fn):
  try:details[name]=fn();checks[name]=True
  except Exception as e:checks[name]=False;errors.append({'check':name,'error':repr(e)})
 check('nonroot',lambda: None if os.getuid()!=0 else (_ for _ in ()).throw(ValueError('Root UID')))
 def read_only():
  path=Path('/tools/.write_probe')
  try:path.write_text('should not be writable')
  except OSError:return True
  raise ValueError('/tools is writable')
 check('readonly_tools',read_only)
 def imports():
  names=['pxr.Usd','OCP','vtk','rtree','content_agent_workflows.geometry','content_agent_workflows.articulation','content_agent_workflows.physics','content_workflow_cli.cli','world_understanding.agentic.validation_scaffold','world_understanding.functions.physics.native_behavior_validation']
  results={}
  for name in names:
   module=importlib.import_module(name);path=getattr(module,'__file__',None)
   if path and not path.startswith('/tools/'):raise ValueError(name+' imported outside /tools: '+path)
   results[name]=path
  return results
 check('relocated_imports',imports)
 for command in [['codex','--version'],['content-workflow-cli','--help'],['usd-cli','--help'],['usd-convert-cad','--help']]:
  def invoke(command=command):
   r=subprocess.run(command,capture_output=True,text=True,timeout=45);(out/(command[0]+'.stdout')).write_text(r.stdout);(out/(command[0]+'.stderr')).write_text(r.stderr)
   if r.returncode:raise ValueError('exit '+str(r.returncode)+' '+r.stderr[-800:])
   if command[0]=='codex' and r.stdout.strip()!='codex-cli 0.154.0':raise ValueError('Codex version mismatch')
   return {'returncode':r.returncode,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest()}
  check('cli_'+command[0],invoke)
 def native():
  from usd_core.physics_runtime import _ovphysx_python
  from usd_core.render.ovrtx import _venv_matches_pin
  probe=subprocess.run(['/tools/ovphysx-venv/bin/python','-c',"from ovphysx import PhysX; p=PhysX(device='cpu'); p.release()"],capture_output=True,text=True,timeout=60)
  (out/'native_probe.stdout').write_text(probe.stdout);(out/'native_probe.stderr').write_text(probe.stderr)
  if probe.returncode:raise ValueError('Native constructor exit '+str(probe.returncode)+': '+probe.stderr[-1000:])
  python=_ovphysx_python(Path('/tools/ovphysx-venv'))
  if python!='/tools/ovphysx-venv/bin/python':raise ValueError('wrong native interpreter')
  if not _venv_matches_pin(Path('/tools/ovrtx-venv')):raise ValueError('OVRTX lock readiness failed')
  return {'ovphysx_constructor_ready':True,'ovrtx_lock_ready':True,'auto_provision':{k:os.environ.get(k) for k in ['WU_OVPHYSX_AUTO_PROVISION','WU_OVRTX_AUTO_PROVISION']},'render_or_step_executed':False}
 check('native_readonly_resolution',native)
 def git():
  r=subprocess.run(['git','-C','/tools/repo','status','--porcelain','--untracked-files=no'],capture_output=True,text=True,check=True)
  if r.stdout.strip():raise ValueError('source mutation')
  return subprocess.check_output(['git','-C','/tools/repo','rev-parse','HEAD'],text=True).strip()
 check('source_git_readable_clean',git)
 check('backend_functional',lambda:subprocess.run([sys.executable,'/work/smoke_backends.py','--execute','--output','/work/backends'],check=True,timeout=180).returncode)
 result={'schema_version':'readonly-namespace-environment-smoke.v2','passed':all(checks.values()),'checks':checks,'details':details,'errors':errors,'uid':os.getuid(),'elapsed_seconds':time.time()-started,'model_calls':False,'gpu_jobs':False,'native_simulation_steps':False,'full_native_workflow_chain_qualified':False,'limitations':['The pinned USD provider omits pxr.UsdValidation; custom NVIDIA rules are available, extension-dependent rules are skipped.','Native resolution initializes the CPU runtime but does not simulate. OVRTX readiness is lock/import readiness, not an image.']}
 (out/'receipt.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result));return 0 if result['passed'] else 2
if __name__=='__main__':raise SystemExit(main())
