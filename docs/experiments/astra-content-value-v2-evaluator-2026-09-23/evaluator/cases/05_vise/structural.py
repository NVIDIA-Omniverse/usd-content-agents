"""USD parsing happens here, in repo Python, NEVER inside ovphysx Python."""
from pathlib import Path
import numpy as np
import json, time
from v2_policy import auxiliary_checks
from v2_process import InspectionDeadline
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from common import check, dump, finite_positive, rigid_matrix, sha, transform
from geometry import surface_compare

def world_matrix(prim):
    return np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim),float).T

def quat(q):
    return list(q.GetImaginary()) + [q.GetReal()]

def pose(prim):
    tr = Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
    return list(tr.GetTranslation()) + quat(tr.GetRotation().GetQuat())

def mesh_world(prim):
    m = UsdGeom.Mesh(prim)
    points = transform(np.asarray(m.GetPointsAttr().Get(),float), world_matrix(prim))
    counts = list(m.GetFaceVertexCountsAttr().Get())
    idx = list(m.GetFaceVertexIndicesAttr().Get())
    triangles, offset = [], 0
    for count in counts:
        if count < 3:
            raise ValueError('invalid mesh face')
        f = idx[offset:offset+count]
        triangles.extend([[f[0],f[k],f[k+1]] for k in range(1,count-1)])
        offset += count
    return points, np.asarray(triangles)

def combine(stage, paths):
    points, faces, offset = [], [], 0
    for path in paths:
        p = stage.GetPrimAtPath(path)
        if not p or not p.IsA(UsdGeom.Mesh):
            raise ValueError('source-map path is not a mesh: ' + path)
        v, f = mesh_world(p)
        points.append(v); faces.append(f+offset); offset += len(v)
    return np.concatenate(points), np.concatenate(faces)

def inspect(final_usd, bindings, contract, source_inventory, source_root, inventory_root, output):
    checks = []
    stage = Usd.Stage.Open(str(final_usd), load=Usd.Stage.LoadAll)
    check(checks,'usd_stage_load',stage is not None)
    if not stage:
        return checks, None
    check(checks,'meters_and_z_up',UsdGeom.GetStageMetersPerUnit(stage)==1 and UsdGeom.GetStageUpAxis(stage)=='Z',
          {'meters_per_unit':UsdGeom.GetStageMetersPerUnit(stage),'up_axis':str(UsdGeom.GetStageUpAxis(stage))})
    timed = [str(a.GetPath()) for p in stage.Traverse() for a in p.GetAttributes() if a.GetNumTimeSamples() and (a.GetName().startswith('xformOp') or a.GetName() in ('points','physics:kinematicEnabled'))]
    check(checks,'no_prescribed_time_sampled_motion_or_geometry',not timed,timed)
    check(checks,'frozen_source_coverage_complete',source_inventory.get('source_coverage_complete') is True,source_inventory.get('coverage_note'),insufficient=True)
    for f in source_inventory['source_files']:
        p = source_root/f['path']
        check(checks,'source_bytes:'+f['path'],p.is_file() and sha(p)==f['sha256'],insufficient=True)
    body_entries = bindings.get('bodies',[])
    body_paths = [b['path'] for b in body_entries]
    check(checks,'unique_body_bindings',len(body_paths)==len(set(body_paths)))
    body_roles = {b['role']:b['path'] for b in body_entries}
    check(checks,'required_body_roles_unique',all(sum(e['role']==r for e in body_entries)==1 for r in contract['required_body_roles']))
    check(checks,'required_body_roles',set(contract['required_body_roles'])<=set(body_roles),{'required':contract['required_body_roles'],'given':body_roles})
    bodies = {}
    for entry in body_entries:
        path = entry['path']; prim = stage.GetPrimAtPath(path)
        valid = bool(prim)
        check(checks,'body_exists:'+path,valid)
        if not valid: continue
        rb = UsdPhysics.RigidBodyAPI(prim)
        moving = bool(entry.get('moving'))
        check(checks,'rigid_body_binding:'+path,not moving or (prim.HasAPI(UsdPhysics.RigidBodyAPI) and bool(rb.GetRigidBodyEnabledAttr().Get()) and not bool(rb.GetKinematicEnabledAttr().Get())))
        check(checks,'unobserved_dynamic_body:'+path,moving or not prim.HasAPI(UsdPhysics.RigidBodyAPI) or bool(rb.GetKinematicEnabledAttr().Get()),'A body declared nonmoving must be static or explicitly kinematic; otherwise its native state would escape observation.')
        matrix = world_matrix(prim)
        check(checks,'finite_rigid_transform:'+path,rigid_matrix(matrix))
        colliders = [str(p.GetPath()) for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
        check(checks,'body_collider:'+path,bool(colliders),colliders)
        if moving:
            mass = UsdPhysics.MassAPI(prim)
            values = [mass.GetMassAttr().Get() or 0, *(mass.GetDiagonalInertiaAttr().Get() or [0,0,0])]
            check(checks,'positive_finite_mass_inertia:'+path,finite_positive(values),values)
            inertia = np.asarray(values[1:])
            check(checks,'inertia_triangle_inequality:'+path,bool(np.max(inertia)<=sum(inertia)-np.max(inertia)+1e-10))
        bodies[path] = {'role':entry['role'],'moving':moving,'pose':pose(prim),'colliders':colliders}
        prim.AddAppliedSchema('PhysxContactReportAPI')
        prim.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.0)
    actual_rb = {str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)}
    check(checks,'all_rigid_bodies_observed',actual_rb<=set(body_paths),sorted(actual_rb-set(body_paths)))
    bound_joint = {j['path']:j['role'] for j in bindings.get('joints',[])}
    check(checks,'unique_joint_bindings',len(bound_joint)==len(bindings.get('joints',[])))
    joints = {}
    for p in stage.Traverse():
        if not p.IsA(UsdPhysics.Joint): continue
        path = str(p.GetPath()); joint=UsdPhysics.Joint(p)
        kind = 'revolute' if p.IsA(UsdPhysics.RevoluteJoint) else ('prismatic' if p.IsA(UsdPhysics.PrismaticJoint) else ('fixed' if p.IsA(UsdPhysics.FixedJoint) else 'unsupported'))
        check(checks,'joint_observer_supported:'+path,kind!='unsupported',kind,insufficient=True)
        if kind=='unsupported': continue
        check(checks,'joint_enabled:'+path,bool(joint.GetJointEnabledAttr().Get()))
        record={'path':path,'kind':kind,'role':bound_joint.get(path),'axis':'X'}
        for i in [0,1]:
            targets=getattr(joint,'GetBody'+str(i)+'Rel')().GetTargets()
            body=str(targets[0]) if targets else None
            check(checks,'joint_body_known:'+path+':'+str(i),body is None or body in bodies,body)
            record['body'+str(i)]=body
            pos=getattr(joint,'GetLocalPos'+str(i)+'Attr')().Get() or Gf.Vec3f(0)
            rot=getattr(joint,'GetLocalRot'+str(i)+'Attr')().Get() or Gf.Quatf(1)
            mat=np.asarray(Gf.Matrix4d().SetRotate(Gf.Quatd(rot.GetReal(),Gf.Vec3d(rot.GetImaginary()))),float).T
            mat[:3,3]=list(pos)
            record['frame'+str(i)]=mat.tolist()
        if kind!='fixed':
            api=UsdPhysics.RevoluteJoint(p) if kind=='revolute' else UsdPhysics.PrismaticJoint(p)
            scale=np.pi/180 if kind=='revolute' else 1
            record['axis']=str(api.GetAxisAttr().Get())
            lo,hi=api.GetLowerLimitAttr().Get(),api.GetUpperLimitAttr().Get()
            record['lower']=float(lo*scale) if lo is not None and np.isfinite(lo) else None
            record['upper']=float(hi*scale) if hi is not None and np.isfinite(hi) else None
        joints[path]=record
    check(checks,'all_joints_declared',set(joints)==set(bound_joint),{'undeclared':sorted(set(joints)-set(bound_joint)),'missing':sorted(set(bound_joint)-set(joints))})
    joint_roles={j['role']:p for p,j in joints.items() if j['role']}
    check(checks,'required_joint_roles',set(contract['required_joint_roles'])<=set(joint_roles),joint_roles)
    for role,kind in contract.get('joint_kinds',{}).items():
        if role in joint_roles:check(checks,'joint_kind:'+role,joints[joint_roles[role]]['kind']==kind)
    for role,body_role in contract.get('joint_moving_body_roles',{}).items():
        if role in joint_roles and body_role in body_roles:
            j=joints[joint_roles[role]];target=body_roles[body_role]
            check(checks,'joint_drives_intended_body:'+role,target in (j['body0'],j['body1']) and bodies[target]['moving'],{'joint':j['path'],'intended_body':target})
    # Connected component / closed chain identity uses actual USD endpoints.
    edges=[(j['body0'],j['body1']) for j in joints.values()]
    nodes=set(v for edge in edges for v in edge)
    components=[]
    while nodes:
        reached={nodes.pop()}
        changed=True
        while changed:
            changed=False
            for a,b in edges:
                if a in reached or b in reached:
                    old=len(reached);reached.update([a,b]);changed|=len(reached)>old
        nodes-=reached;components.append(reached)
    if contract.get('require_joint_graph_cycle'):
        cyclic=any(sum(a in c and b in c for a,b in edges)>=len(c) for c in components)
        check(checks,'actual_joint_graph_has_closed_loop',cyclic)
    if contract.get('require_base_to_effector_chain'):
        base,ee=body_roles.get('base'),body_roles.get('end_effector')
        selected=[joints[joint_roles[r]] for r in contract['required_joint_roles'] if r in joint_roles]
        degree={}; adjacency={}
        for j in selected:
            for a,b in [(j['body0'],j['body1']),(j['body1'],j['body0'])]:
                degree[a]=degree.get(a,0)+1;adjacency.setdefault(a,set()).add(b)
        seen={base};front=[base]
        while front:
            for nxt in adjacency.get(front.pop(),set())-seen:seen.add(nxt);front.append(nxt)
        check(checks,'six_joint_base_to_effector_chain',len(selected)==6 and len(degree)==7 and ee in seen and degree.get(base)==1 and degree.get(ee)==1 and all(v<=2 for v in degree.values()),degree)
    source_map=bindings.get('source_map',[])
    mapped={m['source_id']:m for m in source_map}
    required={p['source_id'] for p in source_inventory['parts'] if p['required']}
    check(checks,'source_parts_mapped_once',len(mapped)==len(source_map) and set(mapped)==required,{'required':len(required),'mapped':len(mapped),'missing':sorted(required-set(mapped))})
    for ident,role in source_inventory.get('source_body_role_requirements',{}).items():
        check(checks,'source_part_correct_mechanism_role:'+ident,ident in mapped and mapped[ident]['body_path'] in bodies and bodies[mapped[ident]['body_path']]['role']==role,{'required_role':role,'mapped_body':mapped.get(ident,{}).get('body_path')})
    for name,passed,evidence in auxiliary_checks(bindings,bodies,joints,source_map,contract):
        check(checks,name,passed,evidence)
    used_mesh=[]
    assembly=np.asarray(bindings.get('assembly_from_source',np.eye(4)),float)
    check(checks,'assembly_alignment_is_rigid',rigid_matrix(assembly))
    for part in source_inventory['parts']:
        ident=part['source_id']
        if not part['required'] or ident not in mapped:continue
        mapping=mapped[ident];paths=mapping.get('mesh_paths',[]);used_mesh.extend(paths)
        pfile=inventory_root/part['geometry_file']
        check(checks,'frozen_geometry_hash:'+ident,sha(pfile)==part['geometry_sha256'],insufficient=True)
        source=np.load(pfile)
        body=stage.GetPrimAtPath(mapping['body_path'])
        check(checks,'source_mesh_body_binding:'+ident,mapping['body_path'] in bodies and all(str(p).startswith(mapping['body_path']+'/') or str(p)==mapping['body_path'] for p in paths))
        if not body or not paths:continue
        check(checks,'source_visuals_visible:'+ident,all(UsdGeom.Imageable(stage.GetPrimAtPath(p)).ComputeVisibility()!='invisible' and UsdGeom.Imageable(stage.GetPrimAtPath(p)).ComputePurpose() not in ('guide','proxy') for p in paths))
        v,f=combine(stage,paths)
        if part['frame']=='source_assembly_world':
            expected=transform(source['vertices'],assembly)
        else:
            local=np.asarray(mapping.get('source_to_body',[]),float)
            check(checks,'part_alignment_is_rigid:'+ident,rigid_matrix(local))
            if not rigid_matrix(local):continue
            expected=transform(source['vertices'],world_matrix(body)@local)
        diag=np.linalg.norm(np.ptp(expected,axis=0))
        tol=max(contract['common']['max_mesh_surface_error_m'],diag*contract['common']['max_mesh_surface_error_fraction'])
        started=time.monotonic()
        with (output/'source_progress.jsonl').open('a') as progress:progress.write(json.dumps({'source_id':ident,'state':'started'})+'\n')
        try:
            okay,detail=surface_compare(expected,source['faces'],v,f,tol)
            check(checks,'source_surface_retained:'+ident,okay,detail)
        except InspectionDeadline:raise
        except Exception as exc:
            okay=False;detail={'error':type(exc).__name__+': '+str(exc)}
            check(checks,'source_measurement_completed:'+ident,False,detail,insufficient=True)
        with (output/'source_progress.jsonl').open('a') as progress:progress.write(json.dumps({'source_id':ident,'state':'completed','passed':okay,'elapsed_s':time.monotonic()-started,'measurement':detail})+'\n')
    check(checks,'source_visual_meshes_unique',len(used_mesh)==len(set(used_mesh)))
    visible=[]
    for p in stage.Traverse():
        if p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).GetPurposeAttr().Get() not in ('guide','proxy'):
            visible.append(str(p.GetPath()))
        elif p.IsA(UsdGeom.Gprim) and not p.IsA(UsdGeom.Mesh) and UsdGeom.Imageable(p).ComputeVisibility()!='invisible' and UsdGeom.Imageable(p).ComputePurpose() not in ('guide','proxy'):
            check(checks,'no_unmapped_visible_primitive:'+str(p.GetPath()),False,'Collision approximations must use proxy/guide purpose or be invisible; original visual geometry must be retained as mapped meshes.')
    check(checks,'no_unmapped_visible_geometry',set(visible)<=set(used_mesh),sorted(set(visible)-set(used_mesh)))
    scenes=[UsdPhysics.Scene(p) for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    check(checks,'exactly_one_physics_scene',len(scenes)==1)
    if len(scenes)==1:
        direction=scenes[0].GetGravityDirectionAttr().Get();magnitude=scenes[0].GetGravityMagnitudeAttr().Get()
        check(checks,'earth_gravity_authored',direction is not None and np.allclose(direction,[0,0,-1],atol=1e-5) and magnitude is not None and abs(magnitude-9.81)<0.01,{'direction':list(direction) if direction is not None else None,'magnitude':magnitude})
    # Evaluator-owned collision-free gravity witness near origin; never writes original.
    witness='/__EvaluatorGravityWitness'
    if stage.GetPrimAtPath(witness):raise ValueError('Reserved evaluator witness path already exists')
    cube=UsdGeom.Cube.Define(stage,witness);cube.CreateSizeAttr(.01)
    cube.AddTranslateOp().Set(Gf.Vec3d(1,1,1))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
    # Collision-free evaluator witness avoids source interactions at any pose.
    cube.CreateVisibilityAttr('invisible')
    mass=UsdPhysics.MassAPI.Apply(cube.GetPrim());mass.CreateMassAttr(.1);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1.666667e-6))
    scenario=output/'instrumented.usda'
    stage.Flatten().Export(str(scenario))
    config={'bodies':bodies,'joints':joints,'body_roles':body_roles,'joint_roles':joint_roles,'gravity_witness':witness,'scenario':str(scenario),'contract':contract}
    dump(output/'runtime_config.json',config)
    return checks,config
