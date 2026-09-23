"""Independent source-faithful drawer acceptance. Never imports an arm's Python code."""
from __future__ import annotations
import argparse, collections, hashlib, json, math, os, pathlib, subprocess, sys, time
import numpy as np
from scipy.spatial import cKDTree
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
HERE=pathlib.Path(__file__).resolve().parent
SPEC=json.loads((HERE/'drawer_acceptance.json').read_text())

def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write_json(p,d): pathlib.Path(p).write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def value(p,name,default=None):
    a=p.GetAttribute(name)
    v=a.Get() if a else None
    return default if v is None else v

def gltf_reference():
    path=HERE/'reference/drawer_cabinet_1k.gltf'; raw=path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SPEC['source_gltf_sha256']
    g=json.loads(raw); data=(HERE/'reference/drawer_cabinet.bin').read_bytes()
    assert hashlib.sha256(data).hexdigest()==SPEC['source_bin_sha256']
    def access(i):
        a=g['accessors'][i]; bv=g['bufferViews'][a['bufferView']]
        dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a['componentType']]
        n={'SCALAR':1,'VEC3':3}[a['type']]; size=np.dtype(dtype).itemsize
        return np.ndarray((a['count'],n),dtype=dtype,buffer=data,offset=bv.get('byteOffset',0)+a.get('byteOffset',0),strides=(bv.get('byteStride',size*n),size)).copy()
    out={}
    for node in g['nodes']:
        tris=[]
        for primitive in g['meshes'][node['mesh']]['primitives']:
            points=access(primitive['attributes']['POSITION']); inds=access(primitive['indices']).reshape(-1,3)
            tris.append(points[inds].astype(np.float64))
        out[node['name']]=np.concatenate(tris)
    return out

def mesh_triangles(stage, paths):
    result=[]; cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    for path in paths:
        prim=stage.GetPrimAtPath(path); mesh=UsdGeom.Mesh(prim)
        if not mesh: raise ValueError('not a Mesh: '+path)
        if UsdGeom.Imageable(prim).ComputeVisibility()==UsdGeom.Tokens.invisible: raise ValueError('source visual hidden: '+path)
        if UsdGeom.Imageable(prim).GetPurposeAttr().Get() in ('proxy','guide'): raise ValueError('source mesh is proxy/guide: '+path)
        points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
        if not np.isfinite(points).all(): raise ValueError('nonfinite vertices')
        matrix=np.array(cache.GetLocalToWorldTransform(prim)); points=np.c_[points,np.ones(len(points))]@matrix; points=points[:,:3]
        counts=list(mesh.GetFaceVertexCountsAttr().Get()); inds=np.array(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int64)
        if sum(counts)!=len(inds) or inds.min()<0 or inds.max()>=len(points): raise ValueError('invalid topology')
        cursor=0
        for n in counts:
            face=inds[cursor:cursor+n];cursor+=n
            if n<3: raise ValueError('face has fewer than 3 points')
            for j in range(1,n-1): result.append(points[[face[0],face[j],face[j+1]]])
    return np.asarray(result,dtype=np.float64)

def geometry_stats(tris):
    points=tris.reshape(-1,3); welded=np.round(points/SPEC['source_fidelity']['topology_weld_tolerance_m']).astype(np.int64)
    _,inverse=np.unique(welded,axis=0,return_inverse=True); faces=inverse.reshape(-1,3)
    edges=np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True)
    return {'triangle_count':len(tris),'bounds_min_m':points.min(0).tolist(),'bounds_max_m':points.max(0).tolist(),'area_m2':float(np.linalg.norm(np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]),axis=1).sum()/2),'welded_boundary_edges':int((counts==1).sum()),'welded_nonmanifold_edges':int((counts>2).sum()),'welded_closed':bool((counts==2).all())}

def fidelity(source,actual):
    s=geometry_stats(source);a=geometry_stats(actual)
    def symmetric_distance(p,q): return float(max(cKDTree(p).query(q)[0].max(),cKDTree(q).query(p)[0].max()))
    vh=symmetric_distance(source.reshape(-1,3),actual.reshape(-1,3));ch=symmetric_distance(source.mean(1),actual.mean(1))
    bounds=max(abs(np.array(s['bounds_min_m'])-a['bounds_min_m']).max(),abs(np.array(s['bounds_max_m'])-a['bounds_max_m']).max())
    ar=abs(s['area_m2']-a['area_m2'])/s['area_m2'];f=SPEC['source_fidelity']
    ok=(s['triangle_count']==a['triangle_count'] and vh<=f['vertex_hausdorff_tolerance_m'] and ch<=f['triangle_centroid_hausdorff_tolerance_m'] and bounds<=f['bounds_tolerance_m'] and ar<=f['relative_area_tolerance'] and s['welded_boundary_edges']==a['welded_boundary_edges'] and s['welded_nonmanifold_edges']==a['welded_nonmanifold_edges'])
    return {'pass':bool(ok),'source':s,'submitted':a,'vertex_hausdorff_m':vh,'centroid_hausdorff_m':ch,'bounds_error_m':float(bounds),'relative_area_error':ar}

def subtree(prim): return list(Usd.PrimRange(prim)) if prim else []
def under(path,parent): return path==parent or path.startswith(parent+'/')
def enabled_colliders(prim): return [p for p in subtree(prim) if p.HasAPI(UsdPhysics.CollisionAPI) and value(p,'physics:collisionEnabled',True)]

def enabled_rigid_ancestors(prim):
    result=[]
    while prim and not prim.IsPseudoRoot():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and value(prim,'physics:rigidBodyEnabled',True):result.append(str(prim.GetPath()))
        prim=prim.GetParent()
    return result

def validate_component_bindings(stage,b):
    errors=[]
    for name,paths in b.get('source_components',{}).items():
        for path in paths:
            prim=stage.GetPrimAtPath(path)
            if not prim:
                errors.append('source component prim absent: '+path);continue
            dynamic=enabled_rigid_ancestors(prim)
            if name=='drawer_cabinet_drawer_01':
                if not under(path,b['drawer_body']) or dynamic!=[b['drawer_body']]:errors.append('upper source mesh must inherit only target drawer rigid body: '+path)
            else:
                if dynamic:errors.append('non-target source component inherits moving rigid body: '+name+' '+path+' '+str(dynamic))
                if name=='drawer_cabinet' and not under(path,b['cabinet_body']):errors.append('cabinet source mesh must be under cabinet_body: '+path)
    return errors

def validate_core(stage,b):
    errors=[];details={}; check=lambda condition,msg: errors.append(msg) if not condition else None
    check(UsdGeom.GetStageUpAxis(stage)=='Y','stage must be Y-up')
    check(abs(UsdGeom.GetStageMetersPerUnit(stage)-1)<1e-9,'stage units must be metres')
    drawer=stage.GetPrimAtPath(b['drawer_body']); cabinet=stage.GetPrimAtPath(b['cabinet_body']);joint=stage.GetPrimAtPath(b['drawer_joint'])
    check(bool(drawer),'drawer body not found');check(bool(cabinet),'cabinet root not found');check(bool(joint),'joint not found')
    if not drawer or not cabinet or not joint:return errors,details
    check(drawer.HasAPI(UsdPhysics.RigidBodyAPI),'drawer missing RigidBodyAPI')
    check(value(drawer,'physics:rigidBodyEnabled',True),'drawer rigid body disabled')
    check(not value(drawer,'physics:kinematicEnabled',False),'drawer is kinematic')
    check(not enabled_rigid_ancestors(cabinet),'cabinet must be anchored static geometry without a rigid-body ancestor')
    for p in stage.Traverse():
        if str(p.GetPath()).startswith('/__Evaluator'):errors.append('reserved evaluator namespace in submission')
        if any('ForceAPI' in x for x in p.GetAppliedSchemas()) or any(a.GetName().startswith('physxForce:') for a in p.GetAttributes()):errors.append('authored force controller is not permitted')
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):errors.append('standalone joint required, articulation root unsupported')
        if p.HasAPI(UsdPhysics.RigidBodyAPI) and value(p,'physics:rigidBodyEnabled',True) and str(p.GetPath())!=b['drawer_body']:errors.append('unexpected dynamic body '+str(p.GetPath()))
        for a in p.GetAttributes():
            if a.GetNumTimeSamples() and (a.GetName().startswith(('xformOp:','physics:','physx','points','faceVertex'))):errors.append('time-sampled physics/geometry '+str(a.GetPath()))
    mass=value(drawer,'physics:mass',0);inertia=np.asarray(value(drawer,'physics:diagonalInertia',[0,0,0]),float);com=value(drawer,'physics:centerOfMass',None)
    check(bool(np.isfinite(mass) and SPEC['drawer_mass_range_kg'][0]<=mass<=SPEC['drawer_mass_range_kg'][1]),'drawer mass outside explicit 0.5–10kg range')
    check(bool(inertia.shape==(3,) and np.isfinite(inertia).all() and (inertia>1e-8).all() and 2*inertia.max()<=inertia.sum()+1e-7),'drawer inertia missing/nonphysical')
    check(com is not None and bool(np.isfinite(np.asarray(com,float)).all()),'drawer explicit COM missing/nonfinite')
    check(joint.IsA(UsdPhysics.PrismaticJoint),'joint must be prismatic')
    check(value(joint,'physics:jointEnabled',True),'joint disabled')
    targets=[str(x) for x in UsdPhysics.Joint(joint).GetBody1Rel().GetTargets()] if joint.IsA(UsdPhysics.Joint) else []
    check(targets==[b['drawer_body']],'joint body1 must be target drawer')
    if joint.IsA(UsdPhysics.Joint):
        p0=UsdPhysics.Joint(joint).GetBody0Rel().GetTargets()
        check(not p0 or str(p0[0])==b['cabinet_body'],'joint body0 must be world or cabinet')
        local=np.eye(3)[{'X':0,'Y':1,'Z':2}.get(value(joint,'physics:axis','X'),0)]
        q=value(joint,'physics:localRot0',Gf.Quatf(1));axis=np.array(Gf.Rotation(Gf.Quatd(q)).TransformDir(Gf.Vec3d(*local)))
        if p0:
            mat=UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(p0[0]));axis=np.array(mat.TransformDir(Gf.Vec3d(*axis)))
        axis/=np.linalg.norm(axis);details['joint_world_axis']=axis.tolist()
        check(float(axis@np.array([0,0,1]))>=SPEC['axis_alignment_cos_min'],'joint axis not positive world Z')
    lo=value(joint,'physics:lowerLimit',float('nan'));hi=value(joint,'physics:upperLimit',float('nan'))
    check(SPEC['joint_lower_range_m'][0]<=lo<=SPEC['joint_lower_range_m'][1],'joint lower limit incompatible with closed source')
    check(SPEC['joint_upper_range_m'][0]<=hi<=SPEC['joint_upper_range_m'][1],'joint upper limit must be 0.22–0.45m')
    details['limits_m']=[float(lo) if math.isfinite(lo) else None,float(hi) if math.isfinite(hi) else None]
    for name,root in [('drawer',drawer),('cabinet',cabinet)]:
        colliders=enabled_colliders(root)
        if name=='cabinet':colliders=[p for p in colliders if not enabled_rigid_ancestors(p)]
        details[name+'_colliders']=[str(p.GetPath()) for p in colliders]
        check(bool(colliders),name+' has no enabled colliders')
        for c in colliders:
            check(c.GetTypeName() in ['Mesh','Cube','Sphere','Capsule','Cylinder','Cone'], 'unsupported collider type '+str(c.GetPath()))
            if c.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(c);pts=np.asarray(mesh.GetPointsAttr().Get(),float)
                check(pts.ndim==2 and len(pts)>=4 and bool(np.isfinite(pts).all()),'invalid collider mesh '+str(c.GetPath()))
    return errors,details

def inspect_submission(b,source=True):
    stage=Usd.Stage.Open(b['_usd_path'])
    if not stage:raise ValueError('Cannot open final USD')
    errors,details=validate_core(stage,b);report={'pass':False,'errors':errors,'physics':details,'components':{}}
    if source:
        errors.extend(validate_component_bindings(stage,b))
        refs=gltf_reference();mapping=b.get('source_components',{})
        if set(mapping)!=set(refs):errors.append('source_components must bind exactly the five original source nodes')
        for name,tris in refs.items():
            try:
                paths=mapping[name]
                if not isinstance(paths,list) or not paths:raise ValueError('component mapping must be nonempty path list')
                if name=='drawer_cabinet_drawer_01' and not all(under(p,b['drawer_body']) for p in paths):errors.append('moving source visual not under target body')
                r=fidelity(tris,mesh_triangles(stage,paths));report['components'][name]=r
                if not r['pass']:errors.append('source surface not preserved: '+name)
            except Exception as e:errors.append('source check '+name+': '+str(e))
    report['pass']=not errors
    return stage,report

def add_raw_schema(p,name):p.AddAppliedSchema(name)

def collision_preflight(stage,b,output,solver):
    directory=output/'initial_collision_preflight';directory.mkdir()
    path=directory/'scene.usda';stage.Flatten().Export(str(path));s=Usd.Stage.Open(str(path))
    center=[10.0,10.0,10.0]
    witness=UsdGeom.Cube.Define(s,'/__EvaluatorQueryWitness');witness.CreateSizeAttr(.2)
    UsdGeom.Xformable(witness).AddTranslateOp().Set(Gf.Vec3d(*center))
    UsdPhysics.CollisionAPI.Apply(witness.GetPrim());s.GetRootLayer().Save()
    request={'scene_usd':str(path),'scene_sha256':sha(path),'drawer_body':b['drawer_body'],
             'witness_center':center,'spec':SPEC,'output':str(directory/'report.json')}
    write_json(directory/'request.json',request)
    with (directory/'native.log').open('w') as log:
        proc=subprocess.run([solver,str(HERE/'drawer_collision_preflight.py'),str(directory/'request.json')],
                            stdout=log,stderr=subprocess.STDOUT,timeout=300)
    if not (directory/'report.json').exists():
        return {'status':'INCONCLUSIVE','pass':False,'error':'native query report missing','returncode':proc.returncode}
    return json.loads((directory/'report.json').read_text())

def contact_report(p):
    add_raw_schema(p,'PhysxContactReportAPI');p.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0)

def trial_scene(stage,b,seed,out):
    layer=stage.Flatten();path=out/'scene.usda';layer.Export(str(path));s=Usd.Stage.Open(str(path))
    for p in list(s.Traverse()):
        for schema in list(p.GetAppliedSchemas()):
            if schema.startswith('PhysicsDriveAPI:'):p.RemoveAPI(UsdPhysics.DriveAPI,schema.split(':',1)[1])
        if p.IsA(UsdPhysics.Scene):s.RemovePrim(p.GetPath())
    scene=UsdPhysics.Scene.Define(s,'/__EvaluatorScene');scene.CreateGravityDirectionAttr(Gf.Vec3f(0,-1,0));scene.CreateGravityMagnitudeAttr(9.81)
    drawer=s.GetPrimAtPath(b['drawer_body']);contact_report(drawer);contact_report(s.GetPrimAtPath(b['cabinet_body']))
    add_raw_schema(drawer,'PhysxRigidBodyAPI');drawer.CreateAttribute('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int).Set(16);drawer.CreateAttribute('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int).Set(8)
    UsdPhysics.RigidBodyAPI(drawer).CreateVelocityAttr(Gf.Vec3f(0));UsdPhysics.RigidBodyAPI(drawer).CreateAngularVelocityAttr(Gf.Vec3f(0))
    drawer.CreateAttribute('physxRigidBody:disableGravity',Sdf.ValueTypeNames.Bool).Set(False)
    rng=np.random.default_rng(seed);payload=SPEC['payload'];pos=np.array(payload['initial_center_m'],float);pos[0]+=rng.uniform(-payload['initial_x_jitter_m'],payload['initial_x_jitter_m']);pos[2]+=rng.uniform(-payload['initial_z_jitter_m'],payload['initial_z_jitter_m']);yaw=rng.uniform(-payload['initial_yaw_jitter_deg'],payload['initial_yaw_jitter_deg'])
    cube=UsdGeom.Cube.Define(s,'/__EvaluatorPayload');cube.CreateSizeAttr(1);xf=UsdGeom.Xformable(cube);xf.AddTranslateOp().Set(Gf.Vec3d(*pos));xf.AddRotateYOp().Set(float(yaw));xf.AddScaleOp().Set(Gf.Vec3f(*payload['size_m']))
    UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim()).CreateKinematicEnabledAttr(False);UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    ma=UsdPhysics.MassAPI.Apply(cube.GetPrim());m=payload['mass_kg'];a=np.array(payload['size_m']);ma.CreateMassAttr(m);ma.CreateDiagonalInertiaAttr(Gf.Vec3f(*(m/12*(a.dot(a)-a*a))));ma.CreateCenterOfMassAttr(Gf.Vec3f(0));contact_report(cube.GetPrim())
    mat=UsdPhysics.MaterialAPI.Apply(s.DefinePrim('/__EvaluatorMaterial','Material'));mat.CreateStaticFrictionAttr(payload['surface_friction']);mat.CreateDynamicFrictionAttr(payload['surface_friction']);mat.CreateRestitutionAttr(0)
    cube.GetPrim().CreateRelationship('material:binding:physics').SetTargets(['/__EvaluatorMaterial'])
    s.GetRootLayer().Save()
    initial=UsdGeom.XformCache().GetLocalToWorldTransform(drawer);pos0=list(initial.ExtractTranslation());rot=initial.ExtractRotationQuat();quat=list(rot.GetImaginary())+[rot.GetReal()]
    req={'scene_usd':str(path),'scene_sha256':sha(path),'drawer_body':b['drawer_body'],'cabinet_body':b['cabinet_body'],'payload_body':'/__EvaluatorPayload','initial_drawer_pose':pos0+quat,'seed':seed,'output':str(out),'spec':SPEC}
    write_json(out/'request.json',req)
    return req

def replay(stage_path,trace_path,out_path,body_paths):
    st=Usd.Stage.CreateNew(str(out_path));st.GetRootLayer().subLayerPaths=[str(stage_path)];st.SetTimeCodesPerSecond(240);st.SetStartTimeCode(0)
    ops={}
    for p in body_paths:
        scale=Gf.Transform(UsdGeom.XformCache().GetLocalToWorldTransform(st.GetPrimAtPath(p))).GetScale()
        x=UsdGeom.Xformable(st.OverridePrim(p));x.ClearXformOpOrder();x.SetResetXformStack(True)
        ops[p]=(x.AddTranslateOp(),x.AddOrientOp());x.AddScaleOp().Set(Gf.Vec3f(*scale))
    rows=0
    for line in pathlib.Path(trace_path).read_text().splitlines():
        row=json.loads(line);tc=row['step'];rows+=1
        for p,key in zip(body_paths,['drawer_pose','payload_pose']):
            a=row[key];ops[p][0].Set(Gf.Vec3d(*a[:3]),tc);ops[p][1].Set(Gf.Quatf(float(a[6]),Gf.Vec3f(*a[3:6])),tc)
    st.SetEndTimeCode(max(0,rows-1));st.GetRootLayer().Save()

def run(bindings,output,solver):
    output.mkdir(parents=True,exist_ok=False);b=json.loads(bindings.read_text());b['_usd_path']=str((bindings.parent/b['final_usd']).resolve())
    report={'protocol':SPEC['protocol_id'],'pass':False,'status':'INCONCLUSIVE','started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'inputs':{'bindings_sha256':sha(bindings),'usd_sha256':sha(b['_usd_path']),'protocol_sha256':sha(HERE/'drawer_acceptance.json'),'evaluator_sha256':sha(__file__),'solver_adapter_sha256':sha(HERE/'drawer_solver.py'),'metrics_sha256':sha(HERE/'drawer_metrics.py'),'collision_preflight_sha256':sha(HERE/'drawer_collision_preflight.py')},'trials':[]}
    try:
        stage,structural=inspect_submission(b);report['inputs']['composed_stage_sha256']=hashlib.sha256(stage.Flatten().ExportToString().encode()).hexdigest();write_json(output/'structural_report.json',structural)
        if structural['pass']:
            query=collision_preflight(stage,b,output,solver);report['initial_collision_preflight']=query
            if not query['pass']:
                report['status']=query['status'];report['failure']='initialized cooked-shape clearance rejected or unavailable'
                write_json(output/'report.json',report);return report
            for seed in SPEC['seeds']:
                trial=output/f'seed_{seed}';trial.mkdir();req=trial_scene(stage,b,seed,trial)
                with (trial/'solver.log').open('w') as log:
                    p=subprocess.run([solver,str(HERE/'drawer_solver.py'),str(trial/'request.json')],stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'UV_NO_CACHE':'1'},timeout=300)
                result=json.loads((trial/'trial_report.json').read_text()) if (trial/'trial_report.json').exists() else {'pass':False,'status':'INCONCLUSIVE','error':'solver missing report','returncode':p.returncode}
                result['seed']=seed;report['trials'].append(result)
                if (trial/'trace.jsonl').exists():replay(trial/'scene.usda',trial/'trace.jsonl',trial/'replay.usda',[b['drawer_body'],'/__EvaluatorPayload'])
            report['status']='FAIL' if any(x.get('status')=='FAIL' for x in report['trials']) else ('PASS' if all(x.get('status')=='PASS' for x in report['trials']) else 'INCONCLUSIVE')
            report['pass']=len(report['trials'])==len(SPEC['seeds']) and all(x['pass'] for x in report['trials'])
        else:report['failure']='structural/source rejection';report['status']='FAIL'
    except Exception as e:report['error']=type(e).__name__+': '+str(e)
    write_json(output/'report.json',report);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--bindings',type=pathlib.Path,required=True);ap.add_argument('--output',type=pathlib.Path,required=True);ap.add_argument('--solver',required=True);args=ap.parse_args()
    r=run(args.bindings.resolve(),args.output.resolve(),args.solver);print(json.dumps({'pass':r['pass'],'report':str(args.output/'report.json')}));sys.exit(0 if r['pass'] else 1)
