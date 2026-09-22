"""Mutations of synthetic source meshes, not scored online assets."""
from pathlib import Path
import json,sys
import numpy as np
import trimesh
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from common import dump,sha,pose_matrix
from fixtures import make
from evaluate import evaluate
from types import SimpleNamespace
root=Path(sys.argv[1]).resolve();root.mkdir(parents=True,exist_ok=True)
request=make(root/'source_fixture');data=json.loads(request.read_text());stage=Usd.Stage.Open(data['scenario']);stage.RemovePrim('/__EvaluatorGravityWitness')
parts=[];source_map=[];ident=np.eye(4).tolist();roles=data['config']['body_roles'];names={'rail':'rail','carrier':'box','base':'support','payload':'part'}
for i,(role,path) in enumerate(roles.items()):
 body=stage.GetPrimAtPath(path);child=next(p for p in Usd.PrimRange(body) if p.IsA(UsdGeom.Gprim));world=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(child)).T
 if child.IsA(UsdGeom.Cylinder):mesh=trimesh.creation.cylinder(radius=.087,height=.007,sections=64)
 else:mesh=trimesh.creation.box(extents=[.01]*3 if role=='base' else [1,1,1])
 vertices=mesh.vertices@world[:3,:3].T+world[:3,3];body_world=np.array(UsdGeom.XformCache().GetLocalToWorldTransform(body)).T;local=(np.c_[vertices,np.ones(len(vertices))]@np.linalg.inv(body_world).T)[:,:3]
 child_path=child.GetPath();stage.RemovePrim(child_path);usd=UsdGeom.Mesh.Define(stage,child_path);usd.CreatePointsAttr(local.tolist());usd.CreateFaceVertexCountsAttr([3]*len(mesh.faces));usd.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist());UsdPhysics.CollisionAPI.Apply(usd.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(usd.GetPrim()).CreateApproximationAttr('convexHull')
 usd.GetPrim().CreateRelationship('material:binding:physics').SetTargets(['/World/Material'])
 filename=f'geometry_{i}.npz';np.savez(root/filename,vertices=vertices,faces=mesh.faces)
 source_id=f'synthetic#solid:{i:04}:{names[role]}'
 parts.append({'source_id':source_id,'frame':'source_assembly_world','geometry_file':filename,'geometry_sha256':sha(root/filename),'required':True,'bounds_m':[vertices.min(0).tolist(),vertices.max(0).tolist()]})
 source_map.append({'source_id':source_id,'mesh_paths':[str(child_path)],'body_path':path,'source_to_body':ident})
stage.GetRootLayer().Save()
dump(root/'source_inventory.json',{'case_id':'02_conveyor','source_files':[],'source_coverage_complete':True,'parts':parts,'coverage_note':'Synthetic fixture only; no online source solution.'})
bindings={'schema_version':1,'case_id':'02_conveyor','bodies':[{'path':p,'role':r,'moving':r!='base'} for r,p in roles.items()],'joints':[{'path':'/World/Joint','role':'rail_rotation'}],'source_map':source_map,'assembly_from_source':ident};dump(root/'bindings.json',bindings)
results=[]
for mutation in ['positive','hidden_source','payload_attached','kinematic_carrier']:
 target=root/(mutation+'.usda');stage.Flatten().Export(str(target));s=Usd.Stage.Open(str(target))
 if mutation=='hidden_source':UsdGeom.Imageable(s.GetPrimAtPath('/World/Payload/Box')).MakeInvisible()
 if mutation=='payload_attached':
  j=UsdPhysics.FixedJoint.Define(s,'/World/FakeAttachment');j.CreateBody0Rel().SetTargets(['/World/Carrier']);j.CreateBody1Rel().SetTargets(['/World/Payload'])
 if mutation=='kinematic_carrier':UsdPhysics.RigidBodyAPI(s.GetPrimAtPath('/World/Carrier')).CreateKinematicEnabledAttr(True)
 s.GetRootLayer().Save()
 args=SimpleNamespace(usd=target,bindings=root/'bindings.json',inventory=root/'source_inventory.json',source_root=str(root),output=root/(mutation+'_evaluation'),solver_python='/opt/astra-content-value-20260921/ovphysx-venv/bin/python')
 result=evaluate(args);expected='accepted' if mutation=='positive' else 'not_accepted';results.append({'fixture':mutation,'expected':expected,'actual':result['status'],'pass':result['status']==expected,'concrete_failures':result['concrete_failures'],'inconclusive':result['inconclusive_checks']})
dump(root/'test_report.json',{'synthetic_only':True,'tests':results,'all_pass':all(x['pass'] for x in results)})
print(json.dumps(results,indent=2))
