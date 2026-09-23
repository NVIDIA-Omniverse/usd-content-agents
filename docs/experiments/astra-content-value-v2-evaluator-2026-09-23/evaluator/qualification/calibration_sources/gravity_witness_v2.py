"""Fresh near-origin collision-free witness calibration; one SDK per process."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from ovphysx import PhysX,TensorType
ROOT=Path(__file__).resolve().parent/'gravity_v2';ROOT.mkdir(exist_ok=True)
TRIALS=[('default',None,9.81),('constrained128',128,9.81),('wrong_gravity_negative',128,0)]
p=argparse.ArgumentParser();p.add_argument('--trial',type=int);a=p.parse_args()
if a.trial is None:
 for i in range(len(TRIALS)):
  with (ROOT/(TRIALS[i][0]+'.log')).open('w') as log:subprocess.run([sys.executable,__file__,'--trial',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=90)
 rows=[json.loads((ROOT/(name+'.json')).read_text()) for name,_,_ in TRIALS]
 (ROOT/'summary.json').write_text(json.dumps({'qualified':all(r['qualified'] for r in rows),'rows':rows,'scope':'Synthetic witness only; no author asset or solver task acceptance'},indent=2)+'\n');sys.exit(0 if all(r['qualified'] for r in rows) else 1)
name,iterations,gravity=TRIALS[a.trial]
scene=ROOT/(name+'.usda')
content=f'''#usda 1.0
(metersPerUnit=1
 upAxis="Z")
def PhysicsScene "Scene" (prepend apiSchemas=["PhysxSceneAPI"]) {{
 vector3f physics:gravityDirection=(0,0,-1)
 float physics:gravityMagnitude={gravity}
 uniform token physxScene:solverType="TGS"
}}
def Cube "Witness" (prepend apiSchemas=["PhysicsRigidBodyAPI","PhysicsMassAPI"]) {{
 double size=.01
 double3 xformOp:translate=(1,1,1)
 uniform token[] xformOpOrder=["xformOp:translate"]
 float physics:mass=.1
 float3 physics:diagonalInertia=(.000001666667,.000001666667,.000001666667)
 token visibility="invisible"
}}
'''
if iterations:
 content+='''def Cube "Companion" (prepend apiSchemas=["PhysicsRigidBodyAPI","PhysicsCollisionAPI","PhysicsMassAPI","PhysxRigidBodyAPI"]) {
 double size=.01
 double3 xformOp:translate=(0,0,1)
 uniform token[] xformOpOrder=["xformOp:translate"]
 float physics:mass=.1
 float3 physics:diagonalInertia=(.000001666667,.000001666667,.000001666667)
 int physxRigidBody:solverPositionIterationCount=128
 int physxRigidBody:solverVelocityIterationCount=32
}
def PhysicsFixedJoint "CompanionFixed" {
 rel physics:body1=</Companion>
 point3f physics:localPos0=(0,0,1)
 point3f physics:localPos1=(0,0,0)
}
'''
scene.write_text(content);sdk=PhysX(device='cpu');_,op=sdk.add_usd(str(scene));sdk.wait_op(op)
pb=sdk.create_tensor_binding(pattern='/Witness',tensor_type=TensorType.RIGID_BODY_POSE);vb=sdk.create_tensor_binding(pattern='/Witness',tensor_type=TensorType.RIGID_BODY_VELOCITY)
pose=np.zeros(pb.shape,np.float32);vel=np.zeros(vb.shape,np.float32);pb.read(pose);initial=pose.reshape(-1).tolist();trace=[]
for i in range(24):
 sdk.step_sync(1/240,i/240);pb.read(pose);vb.read(vel);trace.append({'t':(i+1)/240,'pose':pose.reshape(-1).tolist(),'velocity':vel.reshape(-1).tolist()})
dz=trace[-1]['pose'][2]-initial[2];vz=trace[-1]['velocity'][2];accepted=abs(dz+.5*9.81*.1**2)<.004 and abs(vz+.981)<.03
record={'name':name,'collision_free_witness':True,'companion_position_iterations':iterations,'gravity_m_s2':gravity,'native_initial_pose':initial,'delta_z_m':dz,'velocity_z_m_s':vz,'acceptance_thresholds':{'displacement_error_m':.004,'velocity_error_m_s':.03},'witness_accepted':accepted,'expected_acceptance':gravity==9.81,'qualified':accepted==(gravity==9.81),'trace':trace}
(ROOT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');pb.destroy();vb.destroy();sdk.release()
