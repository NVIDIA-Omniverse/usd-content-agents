"""Independent contact-unit calibration against analytically known m*g."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent;OUT=HERE/'case04_contact_units';OUT.mkdir(exist_ok=True)

def native(scene,dt,mass,out):
    from ovphysx import PhysX,TensorType
    sdk=PhysX(device='cpu');_,op=sdk.add_usd(str(scene));sdk.wait_op(op)
    cb=sdk.create_contact_binding(sensor_patterns=['/Block'],filter_patterns=['/Floor'],filters_per_sensor=1,max_contact_data_count=64)
    a=np.zeros((1,1,3),np.float32);v=sdk.create_tensor_binding(pattern='/Block',tensor_type=TensorType.RIGID_BODY_VELOCITY);vel=np.zeros(v.shape,np.float32)
    force=sdk.create_tensor_binding(pattern='/Block',tensor_type=TensorType.RIGID_BODY_FORCE);f=np.zeros(force.shape,np.float32);f.reshape(-1)[2]=-1.
    values=[];impulses=[]
    for i in range(round(1/dt)):
        force.write(f);sdk.step_sync(dt,i*dt);cb.read_force_matrix(a)
        if i*dt>.5:
            values.append(float(np.linalg.norm(a)));rep=sdk.get_contact_report();impulses.append(sum(abs(float(rep['points'][j].impulse[2])) for j in range(rep['num_points'])))
    v.read(vel);r={'mass_kg':mass,'dt_s':dt,'additional_downward_force_N':1.,'contact_matrix_mean':float(np.mean(values)),'raw_impulse_mean_Ns':float(np.mean(impulses)),'expected_force_N':mass*9.81+1.,'velocity':vel.tolist()};Path(out).write_text(json.dumps(r,indent=2));cb.destroy();v.destroy();force.destroy();sdk.release()

def main():
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
    results=[]
    for dt,mass in [(1/240,1),(1/120,1),(1/240,2)]:
        name=f'm{mass}_hz{round(1/dt)}';scene=OUT/(name+'.usda');s=Usd.Stage.CreateNew(str(scene));UsdGeom.SetStageMetersPerUnit(s,1);UsdGeom.SetStageUpAxis(s,'Z');p=UsdPhysics.Scene.Define(s,'/Physics');p.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));p.CreateGravityMagnitudeAttr(9.81)
        for path,pos,size,kinematic in [('/Block',[0,0,.05],[.1,.1,.1],False),('/Floor',[0,0,-.05],[1,1,.1],True)]:
            x=UsdGeom.Xform.Define(s,path);x.AddTranslateOp().Set(Gf.Vec3d(*pos));prim=x.GetPrim();rb=UsdPhysics.RigidBodyAPI.Apply(prim);rb.CreateKinematicEnabledAttr(kinematic);ma=UsdPhysics.MassAPI.Apply(prim);ma.CreateMassAttr(mass);ma.CreateDiagonalInertiaAttr(Gf.Vec3f(.0016667*mass));g=UsdGeom.Cube.Define(s,path+'/Shape');g.CreateSizeAttr(1);g.AddScaleOp().Set(Gf.Vec3f(*size));UsdPhysics.CollisionAPI.Apply(g.GetPrim());prim.AddAppliedSchema('PhysxContactReportAPI');prim.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
        s.GetRootLayer().Save();out=OUT/(name+'.json')
        with (OUT/(name+'.log')).open('w') as log:subprocess.run(['/opt/astra-content-value-20260922-rerun/ovphysx-venv/bin/python',str(Path(__file__).resolve()),'--native','--scene',str(scene),'--dt',str(dt),'--mass',str(mass),'--output',str(out)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
        r=json.loads(out.read_text());r['matrix_over_expected_force']=r['contact_matrix_mean']/r['expected_force_N'];r['matrix_over_expected_impulse']=r['contact_matrix_mean']/(r['expected_force_N']*dt);results.append(r)
    record={'trials':results,'matrix_is_impulse':all(abs(r['matrix_over_expected_impulse']-1)<.02 for r in results),'matrix_is_force':all(abs(r['matrix_over_expected_force']-1)<.02 for r in results),'scope':'Analytic independent native-contact calibration, nonscored'};(HERE/'case04_contact_units_evidence.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--scene');p.add_argument('--dt',type=float);p.add_argument('--mass',type=float);p.add_argument('--output');a=p.parse_args()
    if a.native:native(a.scene,a.dt,a.mass,a.output)
    else:main()
