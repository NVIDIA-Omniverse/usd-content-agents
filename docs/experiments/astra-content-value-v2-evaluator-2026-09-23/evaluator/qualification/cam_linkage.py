"""Synthetic two-slider closed linkage tests production cam torque/contact path.

No original CAD geometry or author output is used. The
payload/contact fixture is the existing evaluator-owned analytical test block.
"""
import argparse,importlib.util,json,shutil,subprocess,sys
from pathlib import Path
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--solver-python');p.add_argument('--run-solver',action='store_true');p.add_argument('--variant',choices=['positive','broken_linkage'],default='positive');p.add_argument('--seed',type=int,default=11);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 impl=a.output/'implementation';impl.mkdir()
 for f in (ROOT/'cases/04_gripper').iterdir():
  if f.is_file() and f.suffix in ('.py','.json'):shutil.copy2(f,impl/f.name)
 shutil.copy2(ROOT/'qualification/fixture_builders/04_gripper.py',impl/'fixture.py');sys.path.insert(0,str(impl))
 from common import dump,sha,verdict
 from v2_process import run
 from case04_evaluate import observe
 spec=importlib.util.spec_from_file_location('fixture',impl/'fixture.py');fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);fixture.ROOT=a.output/'generated';folder,config=fixture.fixture('positive')
 config.pop('_synthetic_fixture');config.pop('_synthetic_variant');s=Usd.Stage.Open(config['scenario']);center=np.array(config['case04']['payload']['center_source_m']);cam_center=center+np.array([0,-.065,-.03])
 bodies=config['bodies'];roles=config['body_roles'];joints=config['joints'];jroles=config['joint_roles'];config['actuator_sign']=1.
 w=s.GetPrimAtPath('/Witness');UsdGeom.Xformable(w).GetOrderedXformOps()[0].Set(Gf.Vec3d(1,1,1))
 for prim in Usd.PrimRange(w):prim.RemoveAPI(UsdPhysics.CollisionAPI)
 def body(role,pos,mass=.005):
  path='/'+role;x=UsdGeom.Xform.Define(s,path);x.AddTranslateOp().Set(Gf.Vec3d(*pos));prim=x.GetPrim();UsdPhysics.RigidBodyAPI.Apply(prim);m=UsdPhysics.MassAPI.Apply(prim);m.CreateMassAttr(mass);m.CreateCenterOfMassAttr(Gf.Vec3f(0));m.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-6));shape=UsdGeom.Cube.Define(s,path+'/Shape');shape.CreateSizeAttr(.002);UsdPhysics.CollisionAPI.Apply(shape.GetPrim());bodies[path]={'moving':True,'pose':list(pos)+[0,0,0,1],'colliders':[path+'/Shape']};roles[role]=path;return path
 def joint(role,b0,b1,pivot,limits=(-3.14,3.14)):
  path='/'+role;j=UsdPhysics.RevoluteJoint.Define(s,path);j.CreateAxisAttr('Z');j.CreateBody1Rel().SetTargets([b1]);j.CreateCollisionEnabledAttr(False)
  if b0:j.CreateBody0Rel().SetTargets([b0])
  f0=np.eye(4);f1=np.eye(4);f0[:3,3]=pivot-(np.array(bodies[b0]['pose'][:3]) if b0 else 0);f1[:3,3]=pivot-np.array(bodies[b1]['pose'][:3]);j.CreateLocalPos0Attr(Gf.Vec3f(*f0[:3,3]));j.CreateLocalPos1Attr(Gf.Vec3f(*f1[:3,3]));j.CreateLowerLimitAttr(float(np.degrees(limits[0])));j.CreateUpperLimitAttr(float(np.degrees(limits[1])));joints[path]={'path':path,'kind':'revolute','role':role,'axis':'Z','body0':b0,'body1':b1,'frame0':f0.tolist(),'frame1':f1.tolist(),'lower':limits[0],'upper':limits[1]};jroles[role]=path
 cam=body('turntable',cam_center,.2)
 # Independently specified physical flywheel: r32mm,h4mm,m200g. Its exact
 # cylindrical inertia keeps the fixed public torque PD in a sampled stable regime.
 s.RemovePrim(cam+'/Shape');rotor=UsdGeom.Cylinder.Define(s,cam+'/Shape');rotor.CreateRadiusAttr(.032);rotor.CreateHeightAttr(.004);rotor.CreateAxisAttr('Z');UsdPhysics.CollisionAPI.Apply(rotor.GetPrim());mass=UsdPhysics.MassAPI(s.GetPrimAtPath(cam));mass.CreateDiagonalInertiaAttr(Gf.Vec3f(.2*(3*.032**2+.004**2)/12,.2*(3*.032**2+.004**2)/12,.2*.032**2/2))
 joint('actuator',None,cam,cam_center,(-.6,.6))
 for side,sign in [('right',1),('left',-1)]:
  pin=cam_center+np.array([0,sign*.015,0]);outer=cam_center+np.array([sign*.02,sign*.03,0]);arm=body(side+'_arm',(pin+outer)/2)
  joint(side+'_inner',cam,arm,pin);joint(side+'_outer',arm,roles[side+'_finger'],outer)
 if a.variant=='broken_linkage':
  path=jroles['right_outer'];s.RemovePrim(path)
  # Keep declared observation edge: the measured closure must expose the break.
 s.GetRootLayer().Save();dump(folder/'config.json',config)
 receipt={'synthetic_only':True,'original_geometry_or_author_output_used':False,'production_torque_controller':True,'variant':a.variant,'seed':a.seed,'scene_sha256':sha(config['scenario']),'solver_sha256':sha(impl/'case04_solver.py'),'observer_sha256':sha(impl/'case04_evaluate.py'),'fixture_script_sha256':sha(Path(__file__)),'fixture_motor':'Independent200g,r32mm,h4mm cylinder with analytic inertia; no source mass claim','model_calls':0,'native_run':False}
 if a.run_solver:
  result=folder/'result.json'
  with (a.output/'solver.log').open('w') as log:run([a.solver_python,str(impl/'case04_solver.py'),'--config',str(folder/'config.json'),'--seed',str(a.seed),'--output',str(result),'--device','cpu'],stdout=log,stderr=subprocess.STDOUT,timeout=900)
  raw=json.loads(result.read_text());report=verdict(observe(raw,config));dump(a.output/'observed.json',report);receipt.update(native_run=True,accepted=report['accepted'],concrete_failures=report['concrete_failures'],qualified=report['accepted'] if a.variant=='positive' else bool(report['concrete_failures']),result_sha256=sha(result))
 dump(a.output/'qualification.json',receipt);print(json.dumps(receipt));sys.exit(0 if receipt.get('qualified',not a.run_solver) else 1)
if __name__=='__main__':main()
