"""Nonscored analytic torque/inertia integration witness."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'case04_torque_units';OUT.mkdir(exist_ok=True)

def native(scene,inertia,torque,out):
    from ovphysx import PhysX,TensorType
    sdk=PhysX(device='cpu');_,op=sdk.add_usd(scene);sdk.wait_op(op)
    bindings={name:sdk.create_tensor_binding(pattern='/Rotor',tensor_type=getattr(TensorType,'RIGID_BODY_'+name)) for name in ['WRENCH','POSE','VELOCITY']}
    w=np.zeros(bindings['WRENCH'].shape,np.float32);w.reshape(-1)[5]=torque;dt=1/240;n=60
    for i in range(n):bindings['WRENCH'].write(w);sdk.step_sync(dt,i*dt)
    pose=np.zeros(bindings['POSE'].shape,np.float32);vel=np.zeros(bindings['VELOCITY'].shape,np.float32);bindings['POSE'].read(pose);bindings['VELOCITY'].read(vel);q=pose.reshape(-1)[3:];angle=2*float(np.arctan2(q[2],q[3]));expected_velocity=torque/inertia*n*dt;expected_angle=torque/inertia*dt*dt*n*(n+1)/2
    result={'inertia_kg_m2':inertia,'torque_Nm':torque,'dt_s':dt,'steps':n,'measured_angular_velocity_rad_s':float(vel.reshape(-1)[5]),'expected_angular_velocity_rad_s':expected_velocity,'measured_angle_rad':angle,'expected_semiimplicit_angle_rad':expected_angle};result['pass']=abs(result['measured_angular_velocity_rad_s']-expected_velocity)<.0001 and abs(angle-expected_angle)<.0001
    Path(out).write_text(json.dumps(result,indent=2));[b.destroy() for b in bindings.values()];sdk.release()

def main():
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
    results=[]
    for k,(inertia,torque) in enumerate([(.01,.01),(.02,.01),(.01,-.01)]):
        scene=OUT/f'rotor{k}.usda';s=Usd.Stage.CreateNew(str(scene));UsdGeom.SetStageUpAxis(s,'Z');UsdGeom.SetStageMetersPerUnit(s,1);sc=UsdPhysics.Scene.Define(s,'/Physics');sc.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));sc.CreateGravityMagnitudeAttr(9.81)
        g=UsdGeom.Cube.Define(s,'/Rotor');g.CreateSizeAttr(.2);g.AddTranslateOp().Set(Gf.Vec3d(0,0,1));p=g.GetPrim();UsdPhysics.RigidBodyAPI.Apply(p);UsdPhysics.CollisionAPI.Apply(p);m=UsdPhysics.MassAPI.Apply(p);m.CreateMassAttr(1);m.CreateDiagonalInertiaAttr(Gf.Vec3f(inertia));m.CreateCenterOfMassAttr(Gf.Vec3f(0));p.AddAppliedSchema('PhysxRigidBodyAPI');p.CreateAttribute('physxRigidBody:angularDamping',Sdf.ValueTypeNames.Float).Set(0.);p.CreateAttribute('physxRigidBody:linearDamping',Sdf.ValueTypeNames.Float).Set(0.);s.GetRootLayer().Save()
        out=OUT/f'rotor{k}.json'
        with (OUT/f'rotor{k}.log').open('w') as log:subprocess.run(['/opt/astra-content-value-20260922-rerun/ovphysx-venv/bin/python',str(Path(__file__).resolve()),'--native','--scene',str(scene),'--inertia',str(inertia),'--torque',str(torque),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        results.append(json.loads(out.read_text()))
    result={'scope':'Analytic nonscored free rotors, not source CAD mechanism','trials':results,'pass':all(x['pass'] for x in results)};(HERE/'case04_torque_units_evidence.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--scene');p.add_argument('--inertia',type=float);p.add_argument('--torque',type=float);p.add_argument('--output');a=p.parse_args()
    if a.native:native(a.scene,a.inertia,a.torque,a.output)
    else:main()
