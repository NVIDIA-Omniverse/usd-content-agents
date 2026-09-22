"""Read back native authored bytes against the predeclared capstone contract."""
import argparse,datetime,hashlib,json
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade

ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--output',type=Path);args=ap.parse_args()
root=Path('/opt/astra-content-value-20260921/capstone');run=root/'runs'/args.run
assert run.resolve().is_relative_to(root/'runs')
prior=root/'runs/drawer_physics_03/physics.usda';asset=run/'physics.usda'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
a=Usd.Stage.Open(str(prior));b=Usd.Stage.Open(str(asset));reference09=Usd.Stage.Open(str(root/'runs/drawer_physics_09/physics.usda'));checks=[]
def check(name,passed,detail):checks.append({'name':name,'pass':bool(passed),'detail':detail})
def attr(prim,name):return prim.GetAttribute(name).Get()
body='/Asset/drawer_cabinet_drawer_01_1'
joint='/Asset/Joints/candidate_0001_ed11071c24ea'
for name in ['physics:mass','physics:density','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes']:
 x=attr(a.GetPrimAtPath(body),name);y=attr(b.GetPrimAtPath(body),name)
 check('unchanged_'+name,str(x)==str(y),{'original':str(x),'authored':str(y)})
old=a.GetPrimAtPath(joint);new_joints=[p for p in b.Traverse() if p.IsA(UsdPhysics.Joint)];assert len(new_joints)==1;new=new_joints[0]
check('joint_exists',bool(new and new.IsA(UsdPhysics.PrismaticJoint)),str(new))
def api_tokens(prim):
 metadata=prim.GetMetadata('apiSchemas');return sorted(metadata.GetAppliedItems()) if metadata else []
physical_name=lambda n:n.startswith(('physics:','physxJoint:')) or ':physics:' in n
old_names=sorted(x.GetName() for x in old.GetAttributes() if physical_name(x.GetName()))
new_names=sorted(x.GetName() for x in new.GetAttributes() if physical_name(x.GetName()))
check('exact_joint_physical_attribute_inventory',old_names==new_names,{'original':old_names,'authored':new_names})
check('exact_joint_api_inventory',api_tokens(old)==api_tokens(new),{'original':api_tokens(old),'authored':api_tokens(new)})
check('no_joint_drive_api',not any('DriveAPI' in x for x in api_tokens(new)),api_tokens(new))

for x in old.GetAttributes():
 if x.GetName().startswith('physics:'):
  y=new.GetAttribute(x.GetName()).Get();check('joint_'+x.GetName(),str(x.Get())==str(y),{'original':str(x.Get()),'authored':str(y)})
for x in old.GetRelationships():
 expected=['/Asset/drawer_cabinet_0/Primitive_0'] if x.GetName()=='physics:body0' else [str(v) for v in x.GetTargets()]
 check('joint_'+x.GetName(),expected==[str(v) for v in new.GetRelationship(x.GetName()).GetTargets()],{'original':[str(v) for v in x.GetTargets()],'expected':expected,'authored':[str(v) for v in new.GetRelationship(x.GetName()).GetTargets()],'declared_endpoint_protocol':'unscored-native-static-endpoint-repair-v1'})
dynamic=[str(p.GetPath()) for p in b.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI) and p.GetAttribute('physics:rigidBodyEnabled').Get() is not False]
check('only_upper_drawer_dynamic',dynamic==[body],dynamic)
all_rigid=[str(p.GetPath()) for p in b.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
check('no_other_rigid_body_api',all_rigid==[body],all_rigid)
colliders=[str(p.GetPath()) for p in b.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
original_meshes=sorted(str(p.GetPath()) for p in a.Traverse() if p.IsA(UsdGeom.Mesh))
check('exact_original_mesh_collider_inventory',sorted(colliders)==original_meshes,{'expected':original_meshes,'authored':colliders})
groups=[str(p.GetPath()) for p in b.Traverse() if p.IsA(UsdPhysics.CollisionGroup)]
check('no_additional_collision_groups',not groups,groups)
def physical_values(prim):
 return {attr.GetName():str(attr.Get()) for attr in prim.GetAttributes() if attr.GetName().startswith(('physics:','physx')) or ':physics:' in attr.GetName()}
for path in [body,*original_meshes]:
 old_prim=reference09.GetPrimAtPath(path);new_prim=b.GetPrimAtPath(path)
 check('exact09_physical_values_'+path,physical_values(old_prim)==physical_values(new_prim),{'prior09':physical_values(old_prim),'authored':physical_values(new_prim)})
 check('exact09_api_tokens_'+path,api_tokens(old_prim)==api_tokens(new_prim),{'prior09':api_tokens(old_prim),'authored':api_tokens(new_prim)})
old_scenes=[physical_values(p) for p in reference09.Traverse() if p.IsA(UsdPhysics.Scene)]
new_scenes=[physical_values(p) for p in b.Traverse() if p.IsA(UsdPhysics.Scene)]
check('exact09_physics_scene_values',old_scenes==new_scenes,{'prior09':old_scenes,'authored':new_scenes})


check('nonkinematic',b.GetPrimAtPath(body).GetAttribute('physics:kinematicEnabled').Get() is not True,None)
check('no_additional_pair_filters',not any(p.GetRelationship('physics:filteredPairs') and p.GetRelationship('physics:filteredPairs').GetTargets() for p in b.Traverse()),None)
check('no_articulation_root',not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in b.Traverse()),None)
for p in b.Traverse():
 if not p.IsA(UsdGeom.Mesh):continue
 path=str(p.GetPath());moving=path.startswith(body+'/')
 check('enabled_collider_'+path,p.HasAPI(UsdPhysics.CollisionAPI) and attr(p,'physics:collisionEnabled') is not False,None)
 approx=attr(p,'physics:approximation');expected='convexDecomposition' if moving else 'none'
 check('collision_'+path,approx==expected,{'expected':expected,'authored':approx})
 material_binding,relationship=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')
 material=material_binding.GetPrim() if material_binding else None
 check('resolved_physics_material_'+path,bool(material and material.HasAPI(UsdPhysics.MaterialAPI)),{'material':str(material.GetPath()) if material else None,'binding':str(relationship.GetPath()) if relationship else None})
 if material:
  old_mat,_=UsdShade.MaterialBindingAPI(reference09.GetPrimAtPath(path)).ComputeBoundMaterial(materialPurpose='physics')
  check('exact09_resolved_physics_material_'+path,bool(old_mat) and physical_values(old_mat.GetPrim())==physical_values(material),{'prior09':physical_values(old_mat.GetPrim()) if old_mat else None,'authored':physical_values(material)})
  for key,value in [('physics:staticFriction',.6),('physics:dynamicFriction',.6),('physics:restitution',0.)]:
   actual=attr(material,key);check(key+'_'+path,actual is not None and np.isfinite(actual) and abs(actual-value)<=1e-6,{'expected':value,'authored':actual})
 if moving:
  options={key:attr(p,'physxConvexDecompositionCollision:'+key) for key in ['shrinkWrap','errorPercentage','hullVertexLimit','maxConvexHulls','voxelResolution']}
  authored_apis=p.GetMetadata('apiSchemas');tokens=list(authored_apis.GetAppliedItems()) if authored_apis else []
  supported=options['shrinkWrap'] is True and options['hullVertexLimit']==64 and options['errorPercentage'] is not None and abs(options['errorPercentage']-.1)<=1e-6 and options['maxConvexHulls'] is not None and options['maxConvexHulls']==128 and options['voxelResolution'] is not None and options['voxelResolution']==4000000
  check('explicit_supported_tighter_cooking',supported and 'PhysxConvexDecompositionCollisionAPI' in tokens,{'options':options,'authored_api_tokens':tokens,'boundary':'Authored values only; separate actual cooking log, source fidelity and task acceptance required.'})
fidelity_path=root/'evidence'/('original_source_'+args.run+'.json')
fidelity=json.loads(fidelity_path.read_text())
check('original_source_geometry',fidelity['passed'] and fidelity['candidate_sha256']==sha(asset),{'receipt':str(fidelity_path),'sha256':sha(fidelity_path)})
record={'scope':'Read-only post-author audit of declared static-Mesh endpoint repair; all original physical values and09 cooking settings retained. No physical acceptance or USD mutation.',
 'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'asset':str(asset),'asset_sha256':sha(asset),
 'original_mass_reference_sha256':sha(prior),'checks':checks,'pass':all(x['pass'] for x in checks)}
output=args.output or root/'evidence'/('authored_contract_'+args.run+'.json');assert not output.exists();output.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'pass':record['pass'],'failures':[x for x in checks if not x['pass']],'asset_sha256':record['asset_sha256']},indent=2))
raise SystemExit(0 if record['pass'] else 1)
