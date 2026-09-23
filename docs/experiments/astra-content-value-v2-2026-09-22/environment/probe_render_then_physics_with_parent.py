"""Exact CLI sequence in qualified author boundary; no models or benchmark data."""
import argparse,hashlib,json,os,resource,shutil,subprocess,sys,time,threading
from pathlib import Path

def inside(parent_threads=0):
 assert 0 <= parent_threads <= 64
 hold=threading.Event();threads=[threading.Thread(target=hold.wait,daemon=True)for _ in range(parent_threads)]
 for thread in threads:thread.start()
 from pxr import Usd,UsdGeom,UsdPhysics,Gf
 from usd_core import physics_runtime as physics
 out=Path('/work');phase=out/'phase.json'
 def step(name):phase.write_text(json.dumps({'phase':name,'time':time.time()}))
 def invoke(name,cmd,timeout):
  step(name);start=time.monotonic();r=subprocess.run(cmd,cwd=out,capture_output=True,text=True,timeout=timeout);(out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr);return {'returncode':r.returncode,'elapsed_seconds':time.monotonic()-start,'stdout_sha256':hashlib.sha256(r.stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(r.stderr.encode()).hexdigest(),'command':cmd}
 source=out/'gravity.usda';stage=Usd.Stage.CreateNew(str(source));root=UsdGeom.Xform.Define(stage,'/Fixture');stage.SetDefaultPrim(root.GetPrim());UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z');scene=UsdPhysics.Scene.Define(stage,'/Fixture/PhysicsScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81);cube=UsdGeom.Cube.Define(stage,'/Fixture/Body');cube.CreateSizeAttr(.2);cube.AddTranslateOp().Set(Gf.Vec3d(0,0,1));UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim());UsdPhysics.CollisionAPI.Apply(cube.GetPrim());UsdPhysics.MassAPI.Apply(cube.GetPrim()).CreateMassAttr(1);stage.GetRootLayer().Save();before=hashlib.sha256(source.read_bytes()).hexdigest()
 config=out/'.usd-cli';config.mkdir();(config/'config.toml').write_text('[server]\nallowed_roots=["/work","/tools"]\nallowed_write_roots=["/work"]\n[render]\nrenderer="ovrtx"\novrtx_auto_install=false\n')
 base=['/tools/bin/usd-cli','--json','--timeout','600'];result={'scope':'Synthetic actual same-session render readiness followed by explicit CPU Physics; no model/benchmark acceptance.','synthetic_parent_idle_threads':parent_threads,'observed_python_thread_count':threading.active_count()}
 result['open']=invoke('open',base+['open',str(source)],90)
 result['render_probe']=invoke('render_probe',base+['render-probe','--require-engine','ovrtx','--output-dir','/work/readiness'],650)
 # Mirror the readiness guard exactly, except preserve its otherwise suppressed output.
 env=dict(os.environ)
 for key in tuple(env):
  if key.startswith('PYTHON')or key.startswith('DYLD_')or key in {'LD_AUDIT','LD_LIBRARY_PATH','LD_PRELOAD','PXR_PLUGINPATH_NAME','USD_PLUGIN_PATH'}:env.pop(key,None)
 step('constructor_after_render');r=subprocess.run(['/tools/ovphysx-venv/bin/python','-c',physics._OVPHYSX_RUNTIME_PROBE],env=env,capture_output=True,text=True,timeout=70);(out/'constructor_after_render.stdout').write_text(r.stdout);(out/'constructor_after_render.stderr').write_text(r.stderr);result['constructor_after_render']={'returncode':r.returncode,'stderr':r.stderr,'stdout':r.stdout}
 result['physics']=invoke('physics',base+['physics','simulate','--scene',str(source),'--body','/Fixture/Body','--rest-position','0,0,1','--world-up','0,0,1','--engine','ovphysx','--duration','.2','--dt',str(1/240),'--fps','30','--output','/work/simulation'],150)
 trace=out/'simulation/trajectory.jsonl';rows=[json.loads(x)for x in trace.read_text().splitlines()]if trace.exists()else[]
 result['final_z']=rows[-1]['pose'][2]if rows else None;result['source_unchanged']=hashlib.sha256(source.read_bytes()).hexdigest()==before;result['passed']=result['render_probe']['returncode']==0 and result['physics']['returncode']==0 and bool(rows)and rows[-1]['pose'][2]<.9 and result['source_unchanged'];step('finished');(out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

def run(a):
 assert os.geteuid()==0
 sys.path.insert(0,a.harness);from isolation import kill_and_reap,populated
 from run_arm import prepare_runtime_caches
 prior=Path(a.qualified);launch=json.loads((prior/'launch.json').read_text());group=Path(launch['cgroup']);assert populated(group)==0
 out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
 for name in('workspace','home','source','input'):
  p=out/name;p.mkdir(mode=0o700 if name in('workspace','home')else 0o755);os.chown(p,20000,20000)
 shutil.copyfile(__file__,out/'workspace/probe.py')
 for name in('ovphysx-venv.provision.lock','ovrtx-venv.provision.lock'):
  p=out/'home'/name;p.touch();p.chmod(0o600);os.chown(p,20000,20000)
 original=list(launch['argv']);tools=Path(original[original.index('/tools')-1]);prepare_runtime_caches(tools,out/'home',20000)
 start=original.index('/usr/bin/bwrap');argv=original[start:];argv.insert(1,'--unshare-net')
 argv=[str(out)+v[len(str(prior)):]if v.startswith(str(prior))and v!=str(prior/'rootfs')else v for v in argv]
 end=max(i for i,x in enumerate(argv)if x=='--');argv=argv[:end+1]+['/tools/main-venv/bin/python','-B','/work/probe.py','--inside','--parent-threads',str(a.parent_threads)]
 # Common correction selected by root, within unchanged own-home cap.
 temp=out/'home/.runtime_tmp';temp.mkdir(mode=0o700);os.chown(temp,20000,20000)
 for i,x in enumerate(argv):
  if x=='--setenv'and argv[i+1]=='TMPDIR':argv[i+2]='/home/agent/.runtime_tmp'
  if x=='--setenv'and argv[i+1]=='USD_CLI_SESSION':argv[i+2]='render-then-physics-qualification'
 cpus=launch['cpu_affinity']
 def enter():(group/'cgroup.procs').write_text(str(os.getpid()));os.sched_setaffinity(0,cpus);resource.setrlimit(resource.RLIMIT_CORE,(0,0))
 before=(group/'pids.events').read_text();metrics=[];begin=time.monotonic()
 with(out/'stdout.log').open('x')as so,(out/'stderr.log').open('x')as se:
  proc=subprocess.Popen(argv,stdout=so,stderr=se,preexec_fn=enter,env={'PATH':'/usr/bin:/bin'})
  try:
   while proc.poll()is None and time.monotonic()-begin<900:
    ph=json.loads((out/'workspace/phase.json').read_text())if(out/'workspace/phase.json').exists()else{}
    metrics.append({'elapsed_seconds':time.monotonic()-begin,'phase':ph.get('phase'),'pids_current':int((group/'pids.current').read_text()),'memory_current':int((group/'memory.current').read_text()),'pids_events':(group/'pids.events').read_text()});time.sleep(.2)
  finally:kill_and_reap(group);proc.wait(timeout=15)
 result={'returncode':proc.returncode,'cgroup_populated':populated(group),'pids_max':(group/'pids.max').read_text().strip(),'pids_events_before':before,'pids_events_after':(group/'pids.events').read_text(),'peak_pids':max([x['pids_current']for x in metrics],default=0),'elapsed_seconds':time.monotonic()-begin,'argv':argv,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'metrics':metrics}
 (out/'supervisor.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items()if k not in('argv','metrics')}));print((out/'workspace/receipt.json').read_text()if(out/'workspace/receipt.json').exists()else(out/'stderr.log').read_text())
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--inside',action='store_true')
 for k in('qualified','output','harness'):p.add_argument('--'+k)
 p.add_argument('--parent-threads',type=int,default=48);a=p.parse_args();inside(a.parent_threads)if a.inside else run(a)
