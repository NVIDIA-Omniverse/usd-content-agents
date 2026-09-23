"""Gripper-specific source identity, independent payload and contact observer."""
import json
from pathlib import Path
import numpy as np
from common import check,dump,joint_measure,transform
HERE=Path(__file__).resolve().parent

def prepare(config,bindings,inventory,checks,output):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics
    c=json.loads((HERE/'case04_contract.json').read_text());config['case04']=c
    roles=config['body_roles'];jroles=config['joint_roles'];assembly=np.array(bindings['assembly_from_source'])
    check(checks,'gripper_upright',np.allclose(assembly[:3,2],[0,0,1],atol=1e-5))
    required=c['required_body_roles'];available=all(r in roles for r in required)
    if not available:return False
    check(checks,'gripper_roles_distinct',len({roles[r] for r in required})==len(required))
    for r in required:check(checks,'gripper_dynamic_identity:'+r,config['bodies'][roles[r]]['moving']==(r!='base'))
    mapped={x['source_id']:x for x in bindings['source_map']}
    for index,role in c['source_body_role_by_solid'].items():
        ident=[p['source_id'] for p in inventory['parts'] if '#solid:'+str(index).zfill(4)+':' in p['source_id']]
        good=len(ident)==1 and ident[0] in mapped and mapped[ident[0]]['body_path']==roles[role]
        check(checks,'gripper_source_group:'+index,good,{'source_id':ident,'required_role':role})
    for role,endpoints in c['joint_endpoints'].items():
        if role not in jroles:continue
        j=config['joints'][jroles[role]];expected={roles[r] for r in endpoints};actual={j['body0'],j['body1']}
        check(checks,'gripper_joint_endpoints:'+role,expected==actual,{'expected':list(expected),'actual':list(actual)})
    if not all(r in jroles for r in c['required_joint_roles']):return False
    poses={p:b['pose'] for p,b in config['bodies'].items()};grip=assembly[:3,0];closing=assembly[:3,2]
    for role in c['required_joint_roles']:
        j=config['joints'][jroles[role]];axis=np.array(joint_measure(j,poses)['axis_world']);desired=grip if role.endswith('slide') else closing
        check(checks,'gripper_joint_axis:'+role,abs(np.dot(axis,desired))>.999,axis.tolist())
        check(checks,'gripper_bounded_joint:'+role,j['lower'] is not None and j['upper'] is not None and j['lower']<j['upper'])
    act=config['joints'][jroles['actuator']];axis=np.array(joint_measure(act,poses)['axis_world'])
    # USD q is body1 relative to body0; positive sourceZ rotation of cam closes.
    config['actuator_sign']=float(np.sign(np.dot(axis,closing)))*(1 if act['body1']==roles['turntable'] else -1)
    config['grip_axis_world']=grip.tolist();config['closing_axis_world']=closing.tolist()
    stage=Usd.Stage.Open(config['scenario']);payload='/__EvaluatorGripperPayload';config['payload_path']=payload
    for mapping in bindings['source_map']:
        expected=mapping['body_path'];moving=config['bodies'].get(expected,{}).get('moving',False)
        for path in mapping['mesh_paths']:
            prim=stage.GetPrimAtPath(path);enabled=[];dynamic=[]
            while prim and not prim.IsPseudoRoot():
                if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    rb=UsdPhysics.RigidBodyAPI(prim)
                    if rb.GetRigidBodyEnabledAttr().Get():
                        enabled.append(str(prim.GetPath()))
                        if not rb.GetKinematicEnabledAttr().Get():dynamic.append(str(prim.GetPath()))
                prim=prim.GetParent()
            good=(enabled==[expected] and dynamic==[expected]) if moving else not dynamic
            check(checks,'gripper_source_dynamic_ancestry:'+path,good,{'expected_body':expected,'enabled_rigid_ancestors':enabled,'dynamic_ancestors':dynamic})
    external=[]
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            if attr.GetName().startswith('physxForce:') and attr.GetName().split(':')[-1] in ('force','torque'):
                value=attr.Get()
                if value is not None and np.any(np.asarray(value)!=0):external.append(str(attr.GetPath()))
    check(checks,'gripper_no_authored_external_force',not external,external)
    check(checks,'gripper_evaluator_namespace_free',not stage.GetPrimAtPath(payload))
    p=UsdGeom.Xform.Define(stage,payload);p.AddTranslateOp().Set(Gf.Vec3d(*transform([c['payload']['center_source_m']],assembly)[0]))
    # Only upright source rotation is allowed, so yaw suffices.
    yaw=np.degrees(np.arctan2(assembly[1,0],assembly[0,0]));p.AddRotateZOp().Set(float(yaw))
    body=p.GetPrim();UsdPhysics.RigidBodyAPI.Apply(body).CreateKinematicEnabledAttr(False)
    mass=UsdPhysics.MassAPI.Apply(body);m=c['payload']['mass_kg'];x,y,z=c['payload']['size_m'];mass.CreateMassAttr(m);mass.CreateDiagonalInertiaAttr(Gf.Vec3f(m*(y*y+z*z)/12,m*(x*x+z*z)/12,m*(x*x+y*y)/12));mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
    cube=UsdGeom.Cube.Define(stage,payload+'/Shape');cube.CreateSizeAttr(1);cube.AddScaleOp().Set(Gf.Vec3f(x,y,z));UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    body.AddAppliedSchema('PhysxContactReportAPI');body.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
    # Normalize preexisting drives/initial velocities in the evaluator copy.
    # The only active controller is the frozen bounded wrench controller.
    for prim in stage.Traverse():
        for a in prim.GetAttributes():
            name=a.GetName()
            if name.startswith('drive:') and name.endswith((':stiffness',':damping')):a.Set(0.)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb=UsdPhysics.RigidBodyAPI(prim);rb.CreateVelocityAttr(Gf.Vec3f(0));rb.CreateAngularVelocityAttr(Gf.Vec3f(0))
    dest=Path(output)/'case04_instrumented.usda';stage.Flatten().Export(str(dest));config['scenario']=str(dest)
    dump(Path(output)/'runtime_config.json',config);return True

def observe(result,config):
    checks=[];prefix='seed'+str(result['seed'])+':'
    def chk(n,p,e=None,ins=False):check(checks,prefix+n,p,e,ins)
    if 'infrastructure_error' in result:chk('gripper_native_solver',False,result['infrastructure_error'],True);return checks
    c=config['case04'];lim=c['limits'];m=result['metrics']
    chk('gripper_direct_run_completed',result['completed_steps']==result['expected_steps'])
    chk('gripper_finite_state',result['finite_states_and_contacts'])
    for field,threshold in [('max_contact_penetration_m',lim['max_contact_penetration_m']),('max_linear_speed_m_s',lim['max_linear_speed_m_s']),('max_angular_speed_rad_s',lim['max_angular_speed_rad_s']),('max_joint_closure_m',lim['max_joint_closure_m']),('max_joint_axis_error_rad',lim['max_joint_axis_error_rad'])]:chk('gripper_'+field,result[field]<=threshold,{'measured':result[field],'limit':threshold})
    for name in ['native_mass','native_inertia']:
        good=bool(result[name])
        for value in result[name].values():
            a=np.array(value);good &= bool(np.isfinite(a).all())
            if name=='native_mass':good &= bool((a>0).all())
            else:good &= bool((np.linalg.eigvalsh(a.reshape(3,3) if a.size==9 else np.diag(a))>0).all())
        chk('gripper_'+name,good)
    w=result['gravity_witness'];now=w['at_point_one_s'];good=now is not None
    if good:good=abs(now['velocity'][2]+9.81*now['t'])<.03 and abs(now['pose'][2]-w['initial_pose'][2]+.5*9.81*now['t']**2)<.004
    chk('gripper_independent_gravity_witness',good,w,True)
    chk('gripper_bilateral_contact_under_load',m['bilateral_fraction']>=lim['min_bilateral_frame_fraction'],m['bilateral_fraction'])
    chk('gripper_payload_retained_without_fixture',m['max_hold_displacement_m']<=lim['max_hold_displacement_m'],m['max_hold_displacement_m'])
    chk('gripper_both_fingers_closed',min(m['minimum_each_finger_closure_m'])>=lim['min_each_finger_closure_m'],m['minimum_each_finger_closure_m'])
    chk('gripper_release_drop',m['release_drop_m']>=lim['minimum_release_drop_m'],m['release_drop_m'])
    chk('gripper_fixture_removed',m['max_force_after_loading_N']==0.,m['max_force_after_loading_N'])
    chk('gripper_torque_bounded',m['max_torque_Nm']<=c['controller']['max_torque_Nm']+1e-6,m['max_torque_Nm'])
    for p,j in config['joints'].items():
        if j['kind']=='fixed':continue
        lo,hi=result['joint_extrema'][p];tol=lim['joint_limit_tolerance_m'] if j['kind']=='prismatic' else lim['joint_limit_tolerance_rad']
        good=j['lower'] is not None and j['upper'] is not None and lo>=j['lower']-tol and hi<=j['upper']+tol
        chk('gripper_joint_limits:'+p,good,{'range':[lo,hi],'authored':[j['lower'],j['upper']]})
    return checks

def replay(result,config,output):
    from pxr import Gf,Sdf,Usd,UsdGeom
    stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=[str(Path(config['scenario']).resolve())];stage.SetTimeCodesPerSecond(1);stage.SetStartTimeCode(0);stage.SetEndTimeCode(config['case04']['phases_s']['end'])
    for row in result.get('trace',[]):
        for path,pose in row['poses'].items():
            if path not in config['bodies'] and path!=config['payload_path']:continue
            xf=UsdGeom.Xformable(stage.GetPrimAtPath(path));xf.ClearXformOpOrder();op=xf.MakeMatrixXform();xf.SetResetXformStack(True);matrix=Gf.Matrix4d().SetRotate(Gf.Quatd(pose[6],Gf.Vec3d(*pose[3:6])));matrix.SetTranslateOnly(Gf.Vec3d(*pose[:3]));op.Set(matrix,Usd.TimeCode(row['t']))
    stage.GetRootLayer().Save()
