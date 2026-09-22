"""Independent static delivery evidence, never a replacement acceptance verdict."""
import argparse, collections, datetime, hashlib, json, sys, time
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils

VERSION = 'supplementary_structural_delivery_v1'
CASES = ['07_robot_arm','08_excavator','09_printer','10_complex']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def stamp(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
    p=Path(p);assert not p.exists(),'Refusing overwrite: '+str(p)
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')

def observe(scene, bindings, public):
    """Inspect actual composed APIs and required public body/joint role mappings."""
    stage=Usd.Stage.Open(str(scene),load=Usd.Stage.LoadAll)
    if stage is None:raise ValueError('Submitted stage could not be loaded')
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(str(scene))
    if unresolved:raise ValueError('Unresolved dependencies: '+str(unresolved))
    prims=list(Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
    bodies=[];joints=[];colliders=[]
    for p in prims:
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb=UsdPhysics.RigidBodyAPI(p)
            bodies.append({'path':str(p.GetPath()),'enabled':bool(rb.GetRigidBodyEnabledAttr().Get()),'kinematic':bool(rb.GetKinematicEnabledAttr().Get())})
        if p.IsA(UsdPhysics.Joint):
            j=UsdPhysics.Joint(p)
            joints.append({'path':str(p.GetPath()),'type':p.GetTypeName(),'enabled':bool(j.GetJointEnabledAttr().Get()),
                           'body0':[str(x) for x in j.GetBody0Rel().GetTargets()],'body1':[str(x) for x in j.GetBody1Rel().GetTargets()]})
        if p.HasAPI(UsdPhysics.CollisionAPI):
            colliders.append({'path':str(p.GetPath()),'enabled':bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())})
    checks=[]
    def check(name,passed,evidence):checks.append({'name':name,'passed':bool(passed),'evidence':evidence})
    body_entries=bindings.get('bodies',[]);joint_entries=bindings.get('joints',[])
    if not isinstance(body_entries,list) or not isinstance(joint_entries,list):raise ValueError('Malformed body/joint binding collections')
    body_roles=collections.Counter(x.get('role') for x in body_entries);joint_roles=collections.Counter(x.get('role') for x in joint_entries)
    required_bodies=public['required_body_roles'];required_joints=public['required_joint_roles']
    required_paths=[];bound_bodies={}
    for role in required_bodies:
        entries=[x for x in body_entries if x.get('role')==role]
        unique=len(entries)==1;entry=entries[0] if unique else {};path=entry.get('path');prim=stage.GetPrimAtPath(path) if isinstance(path,str) else None
        exists=bool(prim);check('required_body_role_exists:'+role,unique and exists,{'binding_count':len(entries),'path':path,'prim_exists':exists})
        if unique and exists:required_paths.append(path);bound_bodies[role]=path
        # These four scoped tasks explicitly require all non-base bodies to move.
        if role!='base':
            api=UsdPhysics.RigidBodyAPI(prim) if exists and prim.HasAPI(UsdPhysics.RigidBodyAPI) else None
            dynamic=bool(api and api.GetRigidBodyEnabledAttr().Get() and not api.GetKinematicEnabledAttr().Get())
            check('required_moving_body_is_dynamic:'+role,dynamic,{'path':path,'actual_rigid_body_api':bool(api)})
            descendants=[c['path'] for c in colliders if c['enabled'] and path and (c['path']==path or c['path'].startswith(path.rstrip('/')+'/'))]
            check('required_moving_body_has_collider:'+role,bool(descendants),{'path':path,'enabled_collider_paths':descendants})
    check('distinct_required_body_paths',len(required_paths)==len(set(required_paths)),required_paths)
    required_joint_paths=[]
    types={'revolute':UsdPhysics.RevoluteJoint,'prismatic':UsdPhysics.PrismaticJoint,'fixed':UsdPhysics.FixedJoint}
    for role in required_joints:
        entries=[x for x in joint_entries if x.get('role')==role]
        unique=len(entries)==1;entry=entries[0] if unique else {};path=entry.get('path');prim=stage.GetPrimAtPath(path) if isinstance(path,str) else None
        actual=bool(prim and prim.IsA(UsdPhysics.Joint));kind=public['joint_kinds'][role]
        if kind not in types:raise ValueError('Unsupported scoped public joint kind: '+kind)
        correct=bool(actual and prim.IsA(types[kind]));enabled=bool(actual and UsdPhysics.Joint(prim).GetJointEnabledAttr().Get())
        check('required_joint_role_has_enabled_api:'+role,unique and actual and correct and enabled,
              {'binding_count':len(entries),'path':path,'actual_joint_api':actual,'required_kind':kind,'actual_type':prim.GetTypeName() if prim else None,'enabled':enabled})
        if unique and actual:required_joint_paths.append(path)
    check('distinct_required_joint_paths',len(required_joint_paths)==len(set(required_joint_paths)),required_joint_paths)
    failures=[x['name'] for x in checks if not x['passed']]
    return {'schema_counts':{'rigid_bodies':len(bodies),'joints':len(joints),'colliders':len(colliders),
                             'meshes':sum(p.IsA(UsdGeom.Mesh) for p in prims)},
            'bodies':bodies,'joints':joints,'colliders':colliders,'checks':checks,'concrete_missing_delivery_checks':failures,
            'supplementary_status':'DECISIVE_NON_DELIVERY' if failures else 'NO_DECISIVE_NON_DELIVERY_FROM_THESE_CHECKS',
            'required_body_roles':required_bodies,'required_joint_roles':required_joints,'joint_kinds':public['joint_kinds'],
            'used_layer_files':sorted({str(Path(x.realPath).resolve()) for x in layers if x.realPath}),
            'unresolved_dependencies':list(unresolved),
            'scope':'Necessary static physical-delivery conditions only. A positive presence observation is not physical acceptance. Source fidelity, geometry distances, native dynamics, limits, loads and contacts were not evaluated.'}

def audit(args):
    root=args.root.resolve();run=root/'runs/pilot-v1'/args.case/'content_agents'
    qualification=read(args.qualification)
    assert qualification['all_passed'] and qualification['auditor_sha256']==sha(__file__),'Unqualified auditor bytes'
    assert qualification['usd_version']==list(Usd.GetVersion()),'Unqualified USD runtime version'
    paths={k:run/k for k in ['output_manifest.json','submission.json','launch.json']}
    paths['public_task']=root/'protocol/tasks'/(args.case+'.json')
    initial={k:sha(p) for k,p in paths.items()};started=time.perf_counter()
    manifest=read(paths['output_manifest.json']);mapping={x['path']:x for x in manifest['files']}
    submission=read(paths['submission.json']);launch=read(paths['launch.json']);task=read(paths['public_task'])
    assert task['frozen'] is True and task['case_id']==args.case
    assert initial['public_task']==launch['task_sha256'],'Task differs from launched frozen brief'
    result={'version':VERSION,'case_id':args.case,'arm':'content_agents','started_utc':stamp(),
            'auditor_sha256':sha(__file__),'qualification_sha256':sha(args.qualification),
            'moving_role_policy':'For scoped public07-10 tasks, every required body role except base is a physically moving link, digit or axis component. Binding moving flags cannot waive the task.',
            'no_author_assertion_used_as_observed_evidence':True,'original_evaluations_modified':False,
            'scored_files_modified':False,'source_fidelity_status':'NOT_EVALUATED','native_physics_status':'NOT_EVALUATED'}
    scene_key=submission.get('final_scene');bindings_key=submission.get('bindings')
    if not all(isinstance(x,str) and x.strip() for x in [scene_key,bindings_key]):
        result.update(supplementary_status='DECISIVE_NON_DELIVERY',concrete_missing_delivery_checks=['no_submitted_scene_bindings_pair'],schema_counts=None)
    else:
        for key,relative in [('scene',scene_key),('bindings',bindings_key)]:
            rel=Path(relative);p=(run/rel).resolve()
            assert not rel.is_absolute() and p.is_relative_to(run.resolve()) and p.is_file() and not (run/rel).is_symlink()
            assert str(rel) in mapping and sha(p)==mapping[str(rel)]['sha256'],key+' not manifest bound'
            paths[key]=p;initial[key]=sha(p)
        observation=observe(paths['scene'],read(paths['bindings']),task['public_acceptance'])
        layer_hashes=[]
        for item in observation.pop('used_layer_files'):
            p=Path(item);assert p.is_relative_to(run.resolve()),'Composed USD layer outside submitted run'
            rel=str(p.relative_to(run));digest=sha(p)
            assert rel in mapping and mapping[rel]['sha256']==digest,'Composed layer not manifest bound: '+rel
            layer_hashes.append({'path':str(p.relative_to(root)),'sha256':digest})
        result.update(observation,composed_layer_hashes=layer_hashes)
    assert all(sha(p)==initial[k] for k,p in paths.items()),'Audited inputs changed'
    for item in result.get('composed_layer_hashes',[]):assert sha(root/item['path'])==item['sha256']
    result.update(finished_utc=stamp(),elapsed_seconds=time.perf_counter()-started,inputs_unchanged=True,
                  inputs={k:{'path':str(p.relative_to(root)),'sha256':initial[k]} for k,p in paths.items()})
    write(args.output,result)
    print(json.dumps({k:result.get(k) for k in ['case_id','supplementary_status','schema_counts','elapsed_seconds']}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--qualification',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args())
