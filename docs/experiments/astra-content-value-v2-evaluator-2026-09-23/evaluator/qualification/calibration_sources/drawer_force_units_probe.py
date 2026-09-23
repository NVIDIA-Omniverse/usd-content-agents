"""Native force and gravity witness on three tiny synthetic free cubes."""
import pathlib,json,numpy as np
from ovphysx import PhysX,TensorType
root=pathlib.Path(__file__).resolve().parent
scene=root/'force_units_fixture.usda'
text='#usda 1.0\n(\n upAxis="Y"\n metersPerUnit=1\n)\ndef Xform "Witness" {\n def PhysicsScene "Physics" {\n vector3f physics:gravityDirection=(0,-1,0)\n float physics:gravityMagnitude=9.81\n }\n'
for name,mass,x in [('One',1,0),('Two',2,2),('Freefall',1,4)]:
 text+=f''' def Cube "{name}" (prepend apiSchemas=["PhysicsRigidBodyAPI", "PhysicsCollisionAPI", "PhysicsMassAPI"]) {{
 double size=0.01
 double3 xformOp:translate=({x},3,0)
 uniform token[] xformOpOrder=["xformOp:translate"]
 float physics:mass={mass}
 float3 physics:diagonalInertia=(0.0001,0.0001,0.0001)
 }}\n'''
text+='}\n';scene.write_text(text)
p=PhysX(device='cpu');handle,op=p.add_usd(str(scene));p.wait_op(op);bindings=[]
def b(name,typ):
 a=p.create_tensor_binding(pattern='/Witness/'+name,tensor_type=typ);bindings.append(a);return a
poses={n:b(n,TensorType.RIGID_BODY_POSE) for n in ['One','Two','Freefall']};forces={n:b(n,TensorType.RIGID_BODY_FORCE) for n in ['One','Two']}
for i in range(120):
 for f in forces.values():f.write(np.array([[1,0,0]],np.float32))
 p.step_sync(1/240,i/240)
res={}
for n,x in [('One',0),('Two',2),('Freefall',4)]:
 a=np.zeros((1,7),np.float32);poses[n].read(a);res[n]={'pose':a[0].tolist(),'dx_m':float(a[0,0]-x),'fall_m':float(3-a[0,1])}
res['expected_analytic']={'one_kg_dx_m':.125,'two_kg_dx_m':.0625,'freefall_drop_m':1.22625,'integration_tolerance_m':.012}
res['pass']=abs(res['One']['dx_m']-.125)<.002 and abs(res['Two']['dx_m']-.0625)<.002 and abs(res['Freefall']['fall_m']-1.22625)<.012 and abs(res['One']['dx_m']/res['Two']['dx_m']-2)<.01
(root/'force_units_evidence.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res))
for a in reversed(bindings):a.destroy()
p.release()
