"""Synthetic evaluator qualification only; never an online-asset solution."""
import json,subprocess,sys
from pathlib import Path
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
from common import dump,sha
from structural import pose
HERE=Path(__file__).resolve().parent
def make(directory,mutation='positive'):
 directory.mkdir(parents=True,exist_ok=True);path=directory/'synthetic.usda';stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
 scene=UsdPhysics.Scene.Define(stage,'/World/Physics');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
 material=UsdPhysics.MaterialAPI.Apply(stage.DefinePrim('/World/Material','Material'));material.CreateStaticFrictionAttr(.8);material.CreateDynamicFrictionAttr(.6);material.CreateRestitutionAttr(0)
 def body(path,position,mass,inertia):
  p=UsdGeom.Xform.Define(stage,path).GetPrim();UsdGeom.Xformable(p).AddTranslateOp().Set(Gf.Vec3d(*position));UsdPhysics.RigidBodyAPI.Apply(p);m=UsdPhysics.MassAPI.Apply(p);m.CreateMassAttr(mass);m.CreateDiagonalInertiaAttr(Gf.Vec3f(*inertia));m.CreateCenterOfMassAttr(Gf.Vec3f(0));p.AddAppliedSchema('PhysxContactReportAPI');p.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0);return p
 def collision(g):
  p=g.GetPrim();UsdPhysics.CollisionAPI.Apply(p);p.CreateRelationship('material:binding:physics').SetTargets(['/World/Material']);return p
 rail=body('/World/Rail',[.075,0,0],.35,[.0007,.0007,.0014]);c=UsdGeom.Cylinder.Define(stage,'/World/Rail/Disk');c.CreateRadiusAttr(.087);c.CreateHeightAttr(.007);c.CreateAxisAttr('Z');c.AddTranslateOp().Set(Gf.Vec3d(0,0,-.0035));collision(c)
 carrier=body('/World/Carrier',[.00476,-.00308,.005],.04,[3e-6,2e-6,4e-6]);c=UsdGeom.Cube.Define(stage,'/World/Carrier/Box');c.CreateSizeAttr(1);c.AddScaleOp().Set(Gf.Vec3f(.019,.024,.01));collision(c)
 payload=body('/World/Payload',[.00476,-.00308,.019],.01,[5e-7,4e-7,3e-7]);c=UsdGeom.Cube.Define(stage,'/World/Payload/Box');c.CreateSizeAttr(1);c.AddScaleOp().Set(Gf.Vec3f(.01,.014,.018));collision(c)
 base=UsdGeom.Xform.Define(stage,'/World/Base').GetPrim();c=UsdGeom.Cube.Define(stage,'/World/Base/Block');c.CreateSizeAttr(.01);c.AddTranslateOp().Set(Gf.Vec3d(.075,0,-.1));collision(c)
 j=UsdPhysics.RevoluteJoint.Define(stage,'/World/Joint');j.CreateAxisAttr('Z');j.CreateBody1Rel().SetTargets([rail.GetPath()]);j.CreateLocalPos0Attr(Gf.Vec3f(.075,0,0));j.CreateLocalPos1Attr(Gf.Vec3f(0));j.CreateLowerLimitAttr(-10);j.CreateUpperLimitAttr(120)
 if mutation=='locked':j.CreateLowerLimitAttr(0);j.CreateUpperLimitAttr(0)
 if mutation=='no_contact':UsdPhysics.CollisionAPI(stage.GetPrimAtPath('/World/Rail/Disk')).CreateCollisionEnabledAttr(False)
 witness=body('/__EvaluatorGravityWitness',[1000,1000,1000],.1,[1e-6]*3);c=UsdGeom.Cube.Define(stage,'/__EvaluatorGravityWitness/Cube');c.CreateSizeAttr(.01);collision(c)
 stage.GetRootLayer().Save()
 roles={'rail':'/World/Rail','carrier':'/World/Carrier','payload':'/World/Payload','base':'/World/Base'}
 bodies={p:{'role':r,'moving':r!='base','pose':pose(stage.GetPrimAtPath(p))} for r,p in roles.items()}
 f0=[[1,0,0,.075],[0,1,0,0],[0,0,1,0],[0,0,0,1]];ident=[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
 config={'bodies':bodies,'body_roles':roles,'joints':{'/World/Joint':{'body0':None,'body1':roles['rail'],'frame0':f0,'frame1':ident,'axis':'Z','kind':'revolute'}},'joint_roles':{'rail_rotation':'/World/Joint'},'tracked_local_points':{'carrier':[0,0,0],'payload':[0,0,0]},'gravity_witness':'/__EvaluatorGravityWitness'}
 request={'scenario':str(path),'scene_sha256':sha(path),'config':config,'spec':json.loads((HERE/'contract.json').read_text()),'seed':11,'output':str(directory)};dump(directory/'request.json',request)
 return directory/'request.json'
if __name__=='__main__':
 root=Path(sys.argv[1]);results=[]
 for mutation in ['positive','locked','no_contact']:
  request=make(root/mutation,mutation)
  with (root/mutation/'solver.log').open('w') as log:subprocess.run(['/opt/astra-content-value-20260921/ovphysx-venv/bin/python',str(HERE/'solver.py'),str(request)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=300)
  report=json.loads((root/mutation/'report.json').read_text());expected='PASS' if mutation=='positive' else 'FAIL';results.append({'fixture':mutation,'expected':expected,'actual':report['status'],'test_pass':report['status']==expected,'checks':report.get('checks'), 'error':report.get('error')})
 dump(root/'test_report.json',{'synthetic_only':True,'tests':results,'all_pass':all(r['test_pass'] for r in results)});print(json.dumps(results,indent=2))
