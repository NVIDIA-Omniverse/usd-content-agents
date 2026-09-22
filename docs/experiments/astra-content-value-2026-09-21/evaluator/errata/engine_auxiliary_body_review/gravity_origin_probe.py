"""Unscored native freefall fixture; no benchmark scene or acceptance code modified."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from ovphysx import PhysX,TensorType

OUT=Path(__file__).resolve().parent/'gravity_origin_probe';OUT.mkdir(parents=True,exist_ok=False)
def cube(name,xyz,it=None):
 attrs='' if it is None else f'int physxRigidBody:solverPositionIterationCount = {it}\n int physxRigidBody:solverVelocityIterationCount = {max(1,it//4)}'
 return f'''def Cube "{name}" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]) {{
 double size = 0.01
 double3 xformOp:translate = ({xyz[0]},{xyz[1]},{xyz[2]})
 uniform token[] xformOpOrder = ["xformOp:translate"]
 float physics:mass = 0.1
 float3 physics:diagonalInertia = (0.000001666667,0.000001666667,0.000001666667)
 {attrs}
}}'''

rows=[]
trials=[('near_default',[1,1,1],None,None),('far_default',[1000,1000,1000],None,None),('near_128',[1,1,1],128,None),('far_128',[1000,1000,1000],128,None),('far_companion128',[1000,1000,1000],None,128),('near_companion128',[1,1,1],None,128)]
for name,pos,it,companion in trials:
 scene=OUT/(name+'.usda')
 content='''#usda 1.0
(metersPerUnit = 1; upAxis = "Z")
def PhysicsScene "Scene" (prepend apiSchemas=["PhysxSceneAPI"]) {
 vector3f physics:gravityDirection=(0,0,-1)
 float physics:gravityMagnitude=9.81
 uniform token physxScene:solverType="TGS"
}
'''.replace('; upAxis','\n upAxis')+cube('Witness',pos,it)
 if companion:content+='\n'+cube('Companion',[0,0,1],companion)
 scene.write_text(content)
 sdk=PhysX(device='cpu');handle,op=sdk.add_usd(str(scene));sdk.wait_op(op)
 pb=sdk.create_tensor_binding(prim_paths=['/Witness'],tensor_type=TensorType.RIGID_BODY_POSE,raise_if_empty=True)
 vb=sdk.create_tensor_binding(prim_paths=['/Witness'],tensor_type=TensorType.RIGID_BODY_VELOCITY,raise_if_empty=True)
 pose=np.zeros(pb.shape,np.float32);vel=np.zeros(vb.shape,np.float32);pb.read(pose);initial=pose.reshape(-1).astype(float).tolist();trace=[]
 for step in range(24):
  sdk.step_sync(1/240,step/240);pb.read(pose);vb.read(vel)
  trace.append({'t':(step+1)/240,'pose':pose.reshape(-1).astype(float).tolist(),'velocity':vel.reshape(-1).astype(float).tolist()})
 dz=trace[-1]['pose'][2]-initial[2];vz=trace[-1]['velocity'][2]
 row={'name':name,'initial_pose':initial,'position_iterations':it,'companion_position_iterations':companion,'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),'delta_z_m':dz,'vz_m_s':vz,'expected_continuous_delta_z_m':-.5*9.81*.1**2,'expected_vz_m_s':-.981,'position_error_m':abs(dz+.5*9.81*.1**2),'velocity_error_m_s':abs(vz+.981),'frozen_witness_threshold_pass':abs(vz+.981)<.03 and abs(dz+.5*9.81*.1**2)<.004,'trace':trace}
 rows.append(row);print(json.dumps({k:v for k,v in row.items() if k!='trace'}),flush=True)
 pb.destroy();vb.destroy();sdk.remove_usd(handle);sdk.release()
result={'scope':'Synthetic unscored evaluator-witness origin/iteration diagnostic; does not adjudicate or repair an author asset','runtime':'ovphysx0.4.13 native CPU','dt_s':1/240,'steps':24,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'trials':rows}
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
