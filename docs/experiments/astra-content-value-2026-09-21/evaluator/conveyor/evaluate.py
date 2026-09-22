"""Source-grounded rotary contact-transfer acceptance, independent of authors."""
import argparse,json,subprocess,sys,traceback
from pathlib import Path
import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
from common import check,dump,frame,pose_matrix,sha,verdict
from structural import inspect
HERE=Path(__file__).resolve().parent
def evaluate(a):
 a.output.mkdir(parents=True,exist_ok=False);checks=[]
 spec=json.loads((HERE/'contract.json').read_text());inventory=json.loads(a.inventory.read_text());bindings=json.loads(a.bindings.read_text())
 if (HERE/'freeze.json').exists():
  frozen=json.loads((HERE/'freeze.json').read_text())
  for name,h in frozen['files'].items():check(checks,'frozen_evaluator:'+name,sha(HERE/name)==h,insufficient=True)
  check(checks,'frozen_source_inventory',sha(a.inventory)==frozen['source_inventory_sha256'],insufficient=True)
 contract={'required_body_roles':['base','rail','carrier','payload'],'required_joint_roles':['rail_rotation'],'joint_kinds':{'rail_rotation':'revolute'},'common':{'max_mesh_surface_error_m':.00005,'max_mesh_surface_error_fraction':.001}}
 try:
  import jsonschema
  jsonschema.validate(bindings,json.loads((HERE/'bindings.schema.json').read_text()))
  check(checks,'case_identity',bindings['case_id']=='02_conveyor')
  check(checks,'original_initial_world_frame',np.allclose(bindings['assembly_from_source'],np.eye(4),atol=1e-6))
  structural,config=inspect(a.usd,bindings,contract,inventory,Path(a.source_root),a.inventory.parent,a.output);checks+=structural
  if config:
   stage=Usd.Stage.Open(str(config['scenario']));roles=config['body_roles'];mapped={m['source_id']:m for m in bindings['source_map']}
   expected={'rail':'rail','box':'carrier','support':'base','part':'payload'};points={}
   for part in inventory['parts']:
    role=expected[part['source_id'].split(':')[-1]];mapping=mapped.get(part['source_id'],{});path=mapping.get('body_path')
    check(checks,'source_role:'+role,path==roles.get(role))
    if role in ['carrier','payload'] and path in config['bodies']:
     center=np.asarray(part['bounds_m']).mean(0);points[role]=(np.linalg.inv(pose_matrix(config['bodies'][path]['pose']))@np.r_[center,1])[:3].tolist()
    if path:
     for meshpath in mapping.get('mesh_paths',[]):
      p=stage.GetPrimAtPath(meshpath);ancestor=None
      while p and p.GetPath()!='/':
       if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get():ancestor=str(p.GetPath());break
       p=p.GetParent()
      check(checks,'source_inherits_correct_body:'+meshpath,ancestor==(None if role=='base' else path))
   check(checks,'exact_mechanism_bodies',set(roles)==set(expected.values()))
   check(checks,'free_carrier_and_payload',len(config['joints'])==1)
   joint=config['joints'][config['joint_roles']['rail_rotation']]
   check(checks,'rail_joint_endpoints',joint['body1']==roles['rail'] and joint['body0'] in [None,roles['base']])
   initial={p:b['pose'] for p,b in config['bodies'].items()};f=frame(joint,initial,0);axis={'X':0,'Y':1,'Z':2}[joint['axis']]
   check(checks,'rail_axis_and_center',np.linalg.norm(f[:3,3]-spec['center_world_m'])<=.001 and np.dot(f[:3,axis],[0,0,1])>=.999)
   for role in ['rail','carrier','payload']:
    p=stage.GetPrimAtPath(roles[role]);check(checks,'dynamic_role:'+role,config['bodies'][roles[role]]['moving'])
    m=UsdPhysics.MassAPI(p).GetMassAttr().Get();lo,hi=spec['mass_ranges_kg'][role];check(checks,'declared_mass_range:'+role,m is not None and lo<=m<=hi)
    p.CreateAttribute('physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(False);UsdPhysics.RigidBodyAPI(p).CreateVelocityAttr(Gf.Vec3f(0));UsdPhysics.RigidBodyAPI(p).CreateAngularVelocityAttr(Gf.Vec3f(0))
    p.AddAppliedSchema('PhysxRigidBodyAPI');p.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(16);p.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(8)
   for p in stage.Traverse():
    check(checks,'no_articulation:'+str(p.GetPath()),not p.HasAPI(UsdPhysics.ArticulationRootAPI))
    check(checks,'no_authored_force:'+str(p.GetPath()),not any('ForceAPI' in s for s in p.GetAppliedSchemas()))
    for attr in p.GetAttributes():
     if attr.GetName().startswith(('physics:','physx','xformOp:','points','faceVertex')):check(checks,'no_time_samples:'+str(attr.GetPath()),not attr.GetNumTimeSamples())
    for schema in list(p.GetAppliedSchemas()):
     if schema.startswith('PhysicsDriveAPI:'):p.RemoveAPI(UsdPhysics.DriveAPI,schema.split(':',1)[1])
   stage.GetRootLayer().Save();config['tracked_local_points']=points
   dump(a.output/'runtime_config.json',config)
   if all(c['passed'] for c in checks):
    for seed in spec['seeds']:
     trial=a.output/f'seed_{seed}';trial.mkdir();req={'scenario':config['scenario'],'scene_sha256':sha(config['scenario']),'config':config,'spec':spec,'seed':seed,'output':str(trial)};dump(trial/'request.json',req)
     with (trial/'solver.log').open('w') as log:subprocess.run([a.solver_python,str(HERE/'solver.py'),str(trial/'request.json')],stdout=log,stderr=subprocess.STDOUT,timeout=300)
     if not (trial/'report.json').exists():check(checks,f'seed{seed}:solver',False,'Missing native result',True);continue
     result=json.loads((trial/'report.json').read_text())
     if result['status']=='INCONCLUSIVE':check(checks,f'seed{seed}:solver',False,result.get('error'),True)
     else:
      for name,passed in result['checks'].items():check(checks,f'seed{seed}:'+name,passed)
 except Exception as exc:check(checks,'evaluator_completed',False,{'error':repr(exc),'traceback':traceback.format_exc()},True)
 report=verdict(checks);report.update(case_id='02_conveyor',source_inventory_sha256=sha(a.inventory),scene_sha256=sha(a.usd),bindings_sha256=sha(a.bindings),scope=spec['scope'])
 dump(a.output/'acceptance.json',report);print(json.dumps({k:report[k] for k in ['case_id','accepted','status','concrete_failures','inconclusive_checks']}));return report
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--usd',type=Path,required=True);p.add_argument('--bindings',type=Path,required=True);p.add_argument('--inventory',type=Path,required=True);p.add_argument('--source-root',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python',default='/opt/astra-content-value-20260921/ovphysx-venv/bin/python');evaluate(p.parse_args())
