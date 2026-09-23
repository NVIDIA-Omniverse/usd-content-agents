"""Actual isolated CPU guard/CLI probe; preserves silent probe stderr separately."""
import argparse,hashlib,json,os,resource,shutil,subprocess,sys,time
from pathlib import Path

def inside():
 from usd_core import physics_runtime as p
 from pxr import Usd,UsdGeom,UsdPhysics,Gf
 out=Path('/evidence');(out/'home').mkdir(exist_ok=True);os.environ['HOME']=str(out/'home');os.environ['USD_CLI_SESSION']='ovphysx-guard-qualification'
 py=p._ovphysx_venv_python_path(Path('/tools/ovphysx-venv'));lock=p._ovphysx_runtime_lock(Path('/tools/ovphysx-venv'));marker=Path('/tools/ovphysx-venv')/p._OVPHYSX_READY_MARKER
 result={'schema_version':'ovphysx-cli-guard-diagnostic.v1','guard_sha256':hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest(),'python':str(py),'python_exists':py.is_file(),'runtime_lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),'marker':json.loads(marker.read_text()),'probes':[],'model_calls':0,'gpu_jobs':0}
 for name,drop in [('inherited',False),('daemon_sanitized',True)]:
  env=dict(os.environ);env.pop('PYTHONPATH',None)
  if drop:
   for key in tuple(env):
    if key.startswith('PYTHON')or key.startswith('DYLD_')or key in {'LD_AUDIT','LD_LIBRARY_PATH','LD_PRELOAD','PXR_PLUGINPATH_NAME','USD_PLUGIN_PATH'}:env.pop(key,None)
  start=time.monotonic();r=subprocess.run([str(py),'-c',p._OVPHYSX_RUNTIME_PROBE],capture_output=True,text=True,env=env,timeout=70)
  (out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr)
  result['probes'].append({'name':name,'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest(),'stderr_tail':r.stderr[-2500:]})
 result['direct_ready_python']=p._ovphysx_ready_python(Path('/tools/ovphysx-venv'))
 source=out/'gravity.usda';stage=Usd.Stage.CreateNew(str(source));root=UsdGeom.Xform.Define(stage,'/Fixture');stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1.0);UsdGeom.SetStageUpAxis(stage,'Z');scene=UsdPhysics.Scene.Define(stage,'/Fixture/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81);cube=UsdGeom.Cube.Define(stage,'/Fixture/Body');cube.CreateSizeAttr(.2);cube.AddTranslateOp().Set(Gf.Vec3d(0,0,1));UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1.0);stage.GetRootLayer().Save();before=hashlib.sha256(source.read_bytes()).hexdigest()
 config=out/'.usd-cli';config.mkdir();(config/'config.toml').write_text('[server]\nallowed_roots=["/evidence","/tools"]\nallowed_write_roots=["/evidence"]\n')
 cmd=['/tools/bin/usd-cli','--json','--timeout','90','physics','simulate','--scene',str(source),'--body','/Fixture/Body','--rest-position','0,0,1','--world-up','0,0,1','--engine','ovphysx','--duration','0.2','--dt',str(1/240),'--fps','30','--output','/evidence/simulation']
 opened=subprocess.run(['/tools/bin/usd-cli','--json','--timeout','90','open',str(source)],capture_output=True,text=True,cwd=out,timeout=120);(out/'open.stdout').write_text(opened.stdout);(out/'open.stderr').write_text(opened.stderr);result['open_returncode']=opened.returncode
 r=subprocess.run(cmd,capture_output=True,text=True,cwd=out,timeout=120);(out/'cli.stdout').write_text(r.stdout);(out/'cli.stderr').write_text(r.stderr)
 traces=list((out/'simulation').rglob('trajectory.jsonl')) if(out/'simulation').exists()else[]
 rows=[json.loads(x)for x in traces[0].read_text().splitlines()]if traces else[]
 result['cli']={'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'trace_count':len(rows),'final_z':rows[-1]['pose'][2]if rows else None,'source_unchanged':hashlib.sha256(source.read_bytes()).hexdigest()==before}
 result['passed']=all(x['returncode']==0 for x in result['probes'])and r.returncode==0 and bool(rows)and rows[-1]['pose'][2]<.9 and result['cli']['source_unchanged']
 (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

def run(a):
 assert os.geteuid()==0
 sys.path.insert(0,a.candidate_controller);import evaluate_run as c
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False);group=Path(json.loads(Path(a.cgroups).read_text())['lanes'][0]['path']);assert c.populated(group)==0
 paths={}
 for name in ('workspace','source','original','evaluator','evidence'):
  paths[name]=out/name;paths[name].mkdir(mode=0o755)
 os.chown(paths['evidence'],20000,20000);shutil.copyfile(__file__,paths['evaluator']/'probe.py');paths['tools']=Path(a.tools)
 cfg={'tools':a.tools,'rootfs':a.rootfs,'system_mounts':[p for p in('/usr','/lib','/lib64','/etc/ld.so.cache','/etc/ssl/certs','/etc/fonts','/etc/vulkan','/etc/alternatives')if Path(p).exists()]}
 argv=c.namespace_command(cfg,paths,['/tools/main-venv/bin/python','-B','/evaluator/probe.py','--inside'],'/evidence',20000)
 i=argv.index('--');extra=['--setenv','PATH','/tools/bin:/tools/main-venv/bin:/usr/bin:/bin','--setenv','HOME','/evidence/home']
 for name in('ovphysx-venv.provision.lock','ovrtx-venv.provision.lock'):
  p=paths['evidence']/name;p.touch();p.chmod(0o600);os.chown(p,20000,20000);extra+=['--bind',str(p),'/tools/'+name]
 if a.gpu_minor is not None:
  for dev in ['/dev/nvidiactl','/dev/nvidia-uvm','/dev/nvidia-uvm-tools','/dev/nvidia'+str(a.gpu_minor)]:
   if Path(dev).exists():extra+=['--dev-bind',dev,dev]
 if a.loader_cache:extra+=['--ro-bind',a.loader_cache,'/etc/ld.so.cache']
 argv[i:i]=extra;cpus=sorted(os.sched_getaffinity(0))[:4]
 def enter():(group/'cgroup.procs').write_text(str(os.getpid()));os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 start=time.monotonic()
 with(out/'stdout.log').open('x')as so,(out/'stderr.log').open('x')as se:
  proc=subprocess.Popen(argv,stdout=so,stderr=se,preexec_fn=enter,env={'PATH':'/usr/bin:/bin'})
  try:proc.wait(timeout=300)
  finally:c.kill_and_reap(group);proc.wait(timeout=10)
 (out/'supervisor.json').write_text(json.dumps({'returncode':proc.returncode,'cgroup_populated':c.populated(group),'elapsed_seconds':time.monotonic()-start,'command':argv,'probe_sha256':c.sha256(__file__)},indent=2)+'\n')
 print((out/'evidence/receipt.json').read_text()if(out/'evidence/receipt.json').exists()else(out/'stderr.log').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true')
 for key in('tools','rootfs','cgroups','output','candidate-controller','loader-cache'):p.add_argument('--'+key)
 p.add_argument('--gpu-minor',type=int);a=p.parse_args();inside()if a.inside else run(a)
