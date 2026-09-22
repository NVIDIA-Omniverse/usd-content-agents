"""Independent source-frozen printer-bed payload preparation and observation."""
import json
from pathlib import Path
import numpy as np
from common import check,dump,pose_matrix,transform,rigid_matrix
HERE=Path(__file__).resolve().parent

def prepare(config,bindings,inventory,checks,output,placement_record=None):
    """Production uses frozen source placement. Explicit records are for fixtures."""
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    spec=json.loads((HERE/'case09_payload_contract.json').read_text())
    if placement_record is None:
        path=HERE/'case09_placement.json'
        if not path.exists():check(checks,'printer_independent_placement_available',False,'Source-derived bed placement has not been frozen',insufficient=True);return False
        placement_record=json.loads(path.read_text())
    reference=placement_record;placement=reference['placement'];bed=config['body_roles'].get('bed');parts={p['source_id']:p for p in inventory['parts']};mapped={m['source_id']:m for m in bindings['source_map']}
    check(checks,'printer_bed_dynamic',bed in config['bodies'] and config['bodies'][bed]['moving'])
    if bed not in config['bodies']:return False
    for ident in reference['bed_source_ids']:
        good=ident in parts and ident in mapped and mapped[ident]['body_path']==bed
        check(checks,'printer_payload_on_actual_source_bed:'+ident,good,{'required_body':bed,'mapped_body':mapped.get(ident,{}).get('body_path')})
    for ident,digest in reference.get('source_geometry_hashes_consulted',{}).items():
        check(checks,'printer_placement_reference:'+ident,ident in parts and parts[ident]['geometry_sha256']==digest,insufficient=True)
    assembly=np.asarray(bindings['assembly_from_source'],float)
    if not rigid_matrix(assembly):check(checks,'printer_payload_alignment_rigid',False);return False
    normal=assembly[:3,:3]@np.asarray(placement['source_top_normal']);upright=np.allclose(normal,[0,0,1],atol=1e-5)
    check(checks,'printer_original_bed_upright',upright,normal.tolist())
    if not upright:return False
    center=transform([placement['source_center_m']],assembly)[0];bed_matrix=pose_matrix(config['bodies'][bed]['pose']);initial_relative=transform([center],np.linalg.inv(bed_matrix))[0]
    payload='/__EvaluatorPrinterPayload';stage=Usd.Stage.Open(config['scenario'])
    if stage.GetPrimAtPath(payload):check(checks,'printer_evaluator_payload_namespace_free',False);return False
    for ident in reference['bed_source_ids']:
        if ident not in mapped:continue
        for path in mapped[ident]['mesh_paths']:
            p=stage.GetPrimAtPath(path);issues=[];reached=False
            while p and not p.IsPseudoRoot():
                if str(p.GetPath())==bed:reached=True;break
                if p.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(p).GetRigidBodyEnabledAttr().Get():issues.append('nested rigid body '+str(p.GetPath()))
                xf=UsdGeom.Xformable(p)
                if xf and xf.GetResetXformStack():issues.append('visual transform reset '+str(p.GetPath()))
                p=p.GetParent()
            check(checks,'printer_source_bed_follows_body:'+path,reached and not issues,issues)
    x=UsdGeom.Xform.Define(stage,payload);x.AddTranslateOp().Set(Gf.Vec3d(*center));yaw=float(np.degrees(np.arctan2(assembly[1,0],assembly[0,0])));x.AddRotateZOp().Set(yaw);prim=x.GetPrim();rb=UsdPhysics.RigidBodyAPI.Apply(prim);rb.CreateKinematicEnabledAttr(False);rb.CreateVelocityAttr(Gf.Vec3f(0));rb.CreateAngularVelocityAttr(Gf.Vec3f(0));m=spec['mass_kg'];sx,sy,sz=spec['size_m'];ma=UsdPhysics.MassAPI.Apply(prim);ma.CreateMassAttr(m);ma.CreateCenterOfMassAttr(Gf.Vec3f(0));ma.CreateDiagonalInertiaAttr(Gf.Vec3f(m*(sy*sy+sz*sz)/12,m*(sx*sx+sz*sz)/12,m*(sx*sx+sy*sy)/12))
    cube=UsdGeom.Cube.Define(stage,payload+'/Shape');cube.CreateSizeAttr(1);cube.AddScaleOp().Set(Gf.Vec3f(sx,sy,sz));UsdPhysics.CollisionAPI.Apply(cube.GetPrim());prim.AddAppliedSchema('PhysxContactReportAPI');prim.CreateAttribute('physxContactReport:threshold',Sdf.ValueTypeNames.Float).Set(0.)
    material_path='/__EvaluatorPrinterContactMaterial'
    if stage.GetPrimAtPath(material_path):check(checks,'printer_evaluator_material_namespace_free',False);return False
    material=UsdShade.Material.Define(stage,material_path);physical=UsdPhysics.MaterialAPI.Apply(material.GetPrim());physical.CreateStaticFrictionAttr(spec['friction']);physical.CreateDynamicFrictionAttr(spec['friction']);physical.CreateRestitutionAttr(spec['restitution'])
    for path in [payload+'/Shape']+config['bodies'][bed]['colliders']:
        collider=stage.GetPrimAtPath(path)
        if not collider:continue
        UsdShade.MaterialBindingAPI.Apply(collider).Bind(material,UsdShade.Tokens.strongerThanDescendants,'physics')
        collider.AddAppliedSchema('PhysxCollisionAPI');collider.CreateAttribute('physxCollision:contactOffset',Sdf.ValueTypeNames.Float).Set(spec['contact_offset_m']);collider.CreateAttribute('physxCollision:restOffset',Sdf.ValueTypeNames.Float).Set(spec['rest_offset_m'])
    q=Gf.Rotation(Gf.Vec3d(0,0,1),yaw).GetQuat();pose=center.tolist()+list(q.GetImaginary())+[q.GetReal()]
    config['bodies'][payload]={'role':'evaluator_payload','moving':True,'pose':pose,'colliders':[payload+'/Shape']};config['body_roles']['evaluator_payload']=payload
    config['case09']={'payload_path':payload,'bed_path':bed,'initial_payload_pose':pose,'initial_bed_pose':config['bodies'][bed]['pose'],'initial_relative_center_m':initial_relative.tolist(),'spec':spec,'source_placement':placement,'bed_source_ids':reference['bed_source_ids'],'normalize_contact_paths':[payload,bed]}
    destination=Path(output)/'case09_instrumented.usda';stage.Flatten().Export(str(destination));config['scenario']=str(destination);dump(Path(output)/'runtime_config.json',config);return True

def observe(result,config):
    checks=[];seed=result['seed'];prefix=f'seed{seed}:printer_payload:'
    def chk(name,passed,evidence=None,insufficient=False):check(checks,prefix+name,passed,evidence,insufficient)
    if 'infrastructure_error' in result:chk('native_payload_observer',False,result['infrastructure_error'],True);return checks
    rows=result.get('case09_contact_samples')
    if rows is None:chk('native_contact_trace_available',False,'Runtime did not produce independent per-step bed/payload contact observations',True);return checks
    spec=config['case09']['spec'];dt=config['contract']['common']['dt_s'];duration=config['contract']['common']['duration_s'];expected=round(duration/dt)
    chk('every_step_observed',len(rows)==expected,{'observed':len(rows),'expected':expected})
    finite=all(np.isfinite(row['payload_pose']).all() and np.isfinite(row['bed_pose']).all() and np.isfinite(row['impulse_world']).all() and np.isfinite(row['t']) for row in rows)
    chk('finite_payload_contact_trace',finite)
    if not finite:return checks
    times=np.array([row['t'] for row in rows]);chk('monotonic_physics_timestamps',len(times)>1 and np.allclose(np.diff(times),dt,rtol=1e-5,atol=1e-7))
    selected=[row for row in rows if row['t']>=spec['observation_start_s']]
    if not selected:chk('payload_observation_window',False,'No post-settle samples');return checks
    relative=[];forces=[];bed_heights=[]
    for row in selected:
        local=transform([row['payload_pose'][:3]],np.linalg.inv(pose_matrix(row['bed_pose'])))[0];relative.append(local);forces.append(float(np.linalg.norm(row['impulse_world']))/dt);bed_heights.append(row['bed_pose'][2])
    relative=np.array(relative);reference=np.array(config['case09']['initial_relative_center_m']);drift=np.linalg.norm(relative-reference,axis=1);fraction=float(np.mean(np.array(forces)>=spec['minimum_contact_force_N']));rise=float(max(bed_heights)-config['case09']['initial_bed_pose'][2])
    chk('identified_bed_contact',fraction>=spec['minimum_contact_frame_fraction'],{'frame_fraction':fraction,'minimum':spec['minimum_contact_frame_fraction'],'threshold_N':spec['minimum_contact_force_N'],'mean_contact_force_N':float(np.mean(forces))})
    chk('retained_relative_to_original_bed',float(max(drift))<=spec['maximum_payload_drift_relative_to_bed_m'],{'max_relative_displacement_m':float(max(drift)),'limit_m':spec['maximum_payload_drift_relative_to_bed_m']})
    chk('bed_lifted_with_payload',rise>=spec['minimum_bed_world_Z_rise_m'],{'world_Z_rise_m':rise,'required_m':spec['minimum_bed_world_Z_rise_m']})
    return checks
