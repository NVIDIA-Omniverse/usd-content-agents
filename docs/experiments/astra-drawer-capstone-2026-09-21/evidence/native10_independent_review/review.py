"""Independent read-only native10 startup/source/property/query review."""
import argparse,datetime,hashlib,json,math,re
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root;cap=root/'capstone';assert not a.output.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
relative=['runs/drawer_physics_10/physics.usda','runs/drawer_physics_03/physics.usda','evidence/native10_critical_cooking_log_audit.json','evidence/native10_cooking_log_review.json','evidence/native10_cooked_clearance_v1.json','evidence/native10_cooked_clearance_v1.log','evidence/authored_contract_drawer_physics_10.json','evidence/original_source_drawer_physics_10.json','evaluator/source_clear_v1/drawer_acceptance.json','review_native09_cooking_log.py','audit_cooking_log.py']
hashes={x:sha(cap/x) for x in relative};strict=json.loads((cap/relative[2]).read_text());prior=json.loads((cap/relative[3]).read_text());q=json.loads((cap/relative[4]).read_text());log=(cap/relative[5]).read_text();authored=json.loads((cap/relative[6]).read_text());source_receipt=json.loads((cap/relative[7]).read_text());spec=json.loads((cap/relative[8]).read_text());checks=[]
def check(n,v,detail=None):checks.append({'name':n,'pass':bool(v),'detail':detail})
expected='0ef7038845d569a2f502f38af9fb5b1e4e99cf06c5f4e04035e4c259b1b07156';check('exact_native10_asset',hashes[relative[0]]==expected)
check('all_receipts_bind_same_asset',all(x['asset_sha256']==expected for x in [strict,prior,q,authored]) and source_receipt['candidate_sha256']==expected)
check('strict_false_preserved',strict['critical_error_absent'] is False and prior['initial_strict_audit_sha256']==hashes[relative[2]])
check('review_binds_exact_query_and_log',prior['query_positive_control_sha256']==hashes[relative[4]] and strict['logs'][0]['sha256']==hashes[relative[5]])
doc=root/'ovphysx-venv/lib/python3.12/site-packages/ovphysx/docs/ovphysx_overview.md';text=doc.read_text();exact='[Error] [omni.physx.cooking.plugin] registry is false.'
check('installed_documentation_exact_exception','`[Error] [omni.physx.cooking.plugin] registry is false` — a non-fatal initialization' in text and sha(doc)==prior['documentation_sha256'])
def blocks(line):
 if line==exact:return False
 return bool(re.search(r'\[error\]|\[fatal\]|invalid.*(?:must be|parameter|configuration)|failed to (?:parse|load)',line,re.I))
regressions=[(exact,False),(exact+' additional failure',True),('[Error] other registry failure',True),('[Fatal] backend failure',True),('[Warning] Invalid max hull vertices(128) must be between 8 and 64',True),('[Warning] Failed to load collider',True)]
check('narrow_exception_regressions',all(blocks(line)==expected for line,expected in regressions))
blockers=[{'line':i,'message':s} for i,s in enumerate(log.splitlines(),1) if blocks(s)]
check('no_remaining_explicit_error_or_invalid_configuration',not blockers,blockers)
warning_lines=[{'line':i,'message':s} for i,s in enumerate(log.splitlines(),1) if '[Warning]' in s]
check('original_warning_inventory_retained',all(any(m['line']==x['line'] and m['verbatim']==x['message'] for m in strict['logs'][0]['messages']) for x in warning_lines))
source=root/'assets/01_drawer/source/drawer_cabinet_1k.gltf';g=json.loads(source.read_text());buf=(source.parent/g['buffers'][0]['uri']).read_bytes();check('original_source_hashes',sha(source)==spec['source_gltf_sha256'] and hashlib.sha256(buf).hexdigest()==spec['source_bin_sha256'])
def acc(i):
 v=g['accessors'][i];b=g['bufferViews'][v['bufferView']];dt=np.dtype({5123:'<u2',5125:'<u4',5126:'<f4'}[v['componentType']]);w={'SCALAR':1,'VEC3':3}[v['type']]
 return np.ndarray((v['count'],w),dt,buffer=buf,offset=b.get('byteOffset',0)+v.get('byteOffset',0),strides=(b.get('byteStride',dt.itemsize*w),dt.itemsize)).copy()
stage=Usd.Stage.Open(str(cap/relative[0]));old=Usd.Stage.Open(str(cap/relative[1]));cache=UsdGeom.XformCache();ids=[];upper='/Asset/drawer_cabinet_drawer_01_1'
for prim in stage.Traverse():
 if not prim.IsA(UsdGeom.Mesh):continue
 node=prim.GetCustomDataByKey('sourceGltfNode');ids.append(node);src=g['meshes'][g['nodes'][node]['mesh']]['primitives'][0];m=UsdGeom.Mesh(prim)
 check('exact_source_arrays_'+str(node),np.array_equal(acc(src['attributes']['POSITION']),np.asarray(m.GetPointsAttr().Get())) and np.array_equal(acc(src['indices']).reshape(-1),np.asarray(m.GetFaceVertexIndicesAttr().Get())) and np.all(np.asarray(m.GetFaceVertexCountsAttr().Get())==3))
 check('identity_source_transform_'+str(node),not any(k in g['nodes'][node] for k in ['matrix','translation','rotation','scale']) and np.array_equal(np.asarray(cache.GetLocalToWorldTransform(prim)),np.eye(4)))
 check('enabled_collider_'+str(node),prim.HasAPI(UsdPhysics.CollisionAPI) and prim.GetAttribute('physics:collisionEnabled').Get() is not False)
check('exact_five_source_node_bijection',sorted(ids)==list(range(5)))
body=stage.GetPrimAtPath(upper);priorbody=old.GetPrimAtPath(upper)
for name in ['physics:mass','physics:density','physics:centerOfMass','physics:diagonalInertia','physics:principalAxes']:
 check('preserved_'+name,str(body.GetAttribute(name).Get())==str(priorbody.GetAttribute(name).Get()))
mesh=stage.GetPrimAtPath(upper+'/Primitive_0');options={k:mesh.GetAttribute('physxConvexDecompositionCollision:'+k).Get() for k in ['shrinkWrap','errorPercentage','hullVertexLimit','maxConvexHulls','voxelResolution']}
check('actual_saved_tighter_options',options['shrinkWrap'] is True and options['hullVertexLimit']==64 and options['maxConvexHulls']==128 and options['voxelResolution']==4000000 and abs(options['errorPercentage']-.1)<1e-6,options)
check('saved_cooking_api_token','PhysxConvexDecompositionCollisionAPI' in mesh.GetMetadata('apiSchemas').GetAppliedItems())
check('authored_numeric_audit_pass',authored['pass'] and all(x['pass'] for x in authored['checks']))
check('query_binds_frozen_spec',q['spec_sha256']==hashes[relative[8]])
check('query_initialized_no_steps',q['simulation_steps']==0 and np.asarray(q['initial_pose']).shape==(1,7) and np.isfinite(q['initial_pose']).all())
check('known_overlap_positive_control',bool(q['original_invalid_spawn_envelope']['hits']))
check('all25_floor_rays_populated',len(q['floor_rays'])==25 and all(x['hits'] for x in q['floor_rays']))
half=np.array(spec['payload']['size_m'])/2;t=math.radians(spec['payload']['initial_yaw_jitter_deg']);extent=[half[0]*math.cos(t)+half[2]*math.sin(t)+spec['payload']['initial_x_jitter_m'],half[1],half[2]*math.cos(t)+half[0]*math.sin(t)+spec['payload']['initial_z_jitter_m']]
check('exact_payload_yaw_jitter_envelope',np.array_equal(extent,q['payload_envelope']['half_extent']) and q['payload_envelope']['center']==spec['payload']['initial_center_m'] and 0<=t<=min(math.atan2(half[2],half[0]),math.atan2(half[0],half[2])))
check('envelope_clear',not q['payload_envelope']['hits'])
expected_spheres=sorted([x,spec['interior_queries']['y_m'],z] for x in spec['interior_queries']['x_m'] for z in spec['interior_queries']['z_m'])
check('frozen_six_spheres_clear',sorted(x['position'] for x in q['frozen_interior_queries'])==expected_spheres and all(not x['hits'] for x in q['frozen_interior_queries']))
y=[hit['position'][1] for ray in q['floor_rays'] for hit in ray['hits']]
from pxr import Sdf, UsdShade
from content_agent_workflows.simready.asset_identity import build_asset_dependency_manifest
from world_understanding.functions.physics.native_behavior_validation import prepare_native_physics_bundle, validate_native_physics_bundle
run=cap/'runs/drawer_physics_10'; previous=cap/'runs/drawer_physics_09'
def bind(path):
    key=str(path.relative_to(cap));hashes[key]=sha(path);return hashes[key]
prior_stage=Usd.Stage.Open(str(previous/'physics.usda'));bind(previous/'physics.usda')
dependency_before=build_asset_dependency_manifest(run/'physics.usda')
def values(prim,physical=False):
    result={}
    for attr in prim.GetAttributes():
        if physical != (attr.GetName().startswith('physics:') or attr.GetName().startswith('physx')):continue
        if not physical and not attr.HasAuthoredValueOpinion():continue
        result[attr.GetName()]={'type':str(attr.GetTypeName()),'value':str(attr.Get()),'times':attr.GetTimeSamples()}
    return result
def apis(prim):
    return sorted(prim.GetMetadata('apiSchemas').GetAppliedItems() if prim.GetMetadata('apiSchemas') else [])
mesh_paths=[str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
for path in mesh_paths:
    current=stage.GetPrimAtPath(path);before=prior_stage.GetPrimAtPath(path)
    check('actual09_to10_mesh_attributes_'+path,values(current)==values(before))
    check('actual09_to10_physical_attributes_'+path,values(current,True)==values(before,True))
    check('actual09_to10_mesh_apis_'+path,apis(current)==apis(before))
    mat,binding=UsdShade.MaterialBindingAPI(current).ComputeBoundMaterial(materialPurpose='physics')
    oldmat,oldbinding=UsdShade.MaterialBindingAPI(before).ComputeBoundMaterial(materialPurpose='physics')
    check('actual09_to10_material_values_'+path,bool(mat) and bool(oldmat) and values(mat.GetPrim(),True)==values(oldmat.GetPrim(),True))
check('actual09_to10_body_properties',values(body,True)==values(prior_stage.GetPrimAtPath(upper),True) and apis(body)==apis(prior_stage.GetPrimAtPath(upper)))
joints=[p for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)];oldjoints=[p for p in prior_stage.Traverse() if p.IsA(UsdPhysics.Joint)]
check('one_prismatic_joint',len(joints)==len(oldjoints)==1 and joints[0].IsA(UsdPhysics.PrismaticJoint))
j=UsdPhysics.Joint(joints[0]);oldj=UsdPhysics.Joint(oldjoints[0]);cabinet='/Asset/drawer_cabinet_0/Primitive_0'
check('joint_physical_attributes_identical09',values(j.GetPrim(),True)==values(oldj.GetPrim(),True) and apis(j.GetPrim())==apis(oldj.GetPrim()))
check('declared_only_endpoint_relationship_change',list(map(str,j.GetBody0Rel().GetTargets()))==[cabinet] and list(map(str,oldj.GetBody0Rel().GetTargets()))==['/Asset/drawer_cabinet_0'] and list(j.GetBody1Rel().GetTargets())==list(oldj.GetBody1Rel().GetTargets())==[Sdf.Path(upper)])
check('static_mesh_endpoint_identity_world_frame',stage.GetPrimAtPath(cabinet).IsA(UsdGeom.Mesh) and np.array_equal(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(cabinet))),np.eye(4)))
check('only_upper_rigid_body_and_five_colliders',[str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]==[upper] and sorted(str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))==sorted(mesh_paths))
check('no_hidden_drive_filters_articulation',not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) or p.IsA(UsdPhysics.CollisionGroup) or any(x.startswith('PhysicsDriveAPI:') for x in apis(p)) or any(r.GetName()=='physics:filteredPairs' and r.GetTargets() for r in p.GetRelationships()) for p in stage.Traverse()))
report_path=run/'runtime/runtime_validation_report.json';report=json.loads(report_path.read_text());bind(report_path)
oldreport_path=previous/'runtime/runtime_validation_report.json';oldreport=json.loads(oldreport_path.read_text());bind(oldreport_path)
check('native_acceptance_thresholds_identical09',report['acceptance']==oldreport['acceptance'],report['acceptance'])
info=report['scene_info'];check('mounted_preserves_authored_origin',info['placement_mode']=='mounted' and info['drop_height_m_resolved']==0 and info['rest_position']==[0,0,0])
check('explicit_gravity_and_exact_support_geometry',info['gravity_magnitude_m_per_s2']==9.81 and info['world_up']==[0,1,0] and info['ground_clearance_support_decision']['exact'] is True and info['ground_clearance_support_decision']['fallback_accepted'] is False)
trajectory_path=run/'runtime/usd_cli_simulation/trajectory.jsonl';bind(trajectory_path)
rows=[json.loads(x) for x in trajectory_path.read_text().splitlines() if x.strip()]
poses=np.asarray([x['pose'] for x in rows]);vel=np.asarray([x['vel'] for x in rows]);times=np.asarray([x['t'] for x in rows]);summary=report['summary']
check('native91_finite_contiguous_samples',len(rows)==91 and [x['frame'] for x in rows]==list(range(91)) and poses.shape==(91,7) and vel.shape==(91,6) and np.isfinite(poses).all() and np.isfinite(vel).all() and np.max(np.abs(times-np.arange(91)/30))<1e-12)
measured={'maximum_translation_from_authored_origin_m':float(np.max(np.linalg.norm(poses[:,:3],axis=1))),'first_recorded_displacement_from_initial_m':float(np.linalg.norm(poses[1,:3]-poses[0,:3])),'maximum_linear_speed_m_s':float(np.max(np.linalg.norm(vel[:,:3],axis=1))),'maximum_angular_speed_rad_s':float(np.max(np.linalg.norm(vel[:,3:],axis=1))),'final_speed_m_s':float(np.linalg.norm(vel[-1,:3]))}
check('native_trajectory_summary_recomputed',math.isclose(measured['maximum_linear_speed_m_s'],summary['max_linear_speed'],abs_tol=1e-15) and math.isclose(measured['maximum_angular_speed_rad_s'],summary['max_angular_speed'],abs_tol=1e-15) and math.isclose(measured['first_recorded_displacement_from_initial_m'],summary['first_step_displacement_m'],abs_tol=1e-15) and list(poses[-1,:3])==summary['final_position'],measured)
manifest_path=run/'workflow_run_manifest.json';manifest=json.loads(manifest_path.read_text());bind(manifest_path)
manifest_checks=[]
for item in manifest['artifacts']:
    path=run/item['path']
    # Hash-only coverage of producer/private artifacts; their content is not copied into this receipt.
    manifest_checks.append(path.is_file() and sha(path)==item['sha256'] and path.stat().st_size==item['size_bytes'])
check('native_manifest_all_artifact_bytes_match',all(manifest_checks),{'artifact_count':len(manifest_checks),'status':manifest['status']})
check('native_source_package_matches_manifest',sha(Path(manifest['source_path']))==manifest['source_sha256'])
launch_path=cap/'evidence/physics_native_execution_10.json';launch=json.loads(launch_path.read_text());bind(launch_path)
check('native_producer_model_code_terminal',launch['model']=='gpt-6-astra' and launch['effort']=='ultra' and launch['code_sha']=='e640b8d6aa667745830db5e9bf87fd1b4cd763cb' and launch['returncode']==0)
capture_path=cap/'evidence/native10_pre_visual_runtime/capture.json';capture=json.loads(capture_path.read_text());bind(capture_path)
check('pre_visual_runtime_bytes_preserved',capture['asset_sha256']==expected and all(sha(Path(x['saved']))==x['sha256'] for x in capture['artifacts']) and sha(report_path)==next(x['sha256'] for x in capture['artifacts'] if Path(x['source']).name=='runtime_validation_report.json'))
bundle=prepare_native_physics_bundle(asset=run/'physics.usda',assessment=run/'physics_behavior_assessment.json',validation_evidence=run/'validation_evidence.json',run_dir=run)
bundle_path=a.output.parent/'native_producer_bundle.json';assert not bundle_path.exists();bundle_path.write_text(json.dumps(bundle,indent=2)+'\n')
native_result=validate_native_physics_bundle(bundle_path,usd_paths=[run/'physics.usda']);check('strict_native_producer_closure',native_result['status']=='passed',native_result)
native_code=Path(__import__('world_understanding.functions.physics.native_behavior_validation',fromlist=['']).__file__)
native_code_sha=sha(native_code)
frames=[]
for path in sorted((run/'runtime/visual_review_frames_1').glob('review_frame_*.png')):
    frames.append({'path':str(path.relative_to(cap)),'sha256':bind(path)})
check('actual_eight_reviewed_frames_retained',len(frames)==8)
dependency_after=build_asset_dependency_manifest(run/'physics.usda')
check('complete_asset_dependency_closure_unchanged',dependency_before==dependency_after)
record={'schema_version':1,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_script_sha256':sha(Path(__file__)),'input_hashes':hashes,'installed_runtime_overview_sha256':sha(doc),'documentation_lines':[137,138],'checks':checks,'bounded_cpu_preflight_supported':all(x['pass'] for x in checks),'blocking_findings':blockers,'strict_original_verdict':'critical_error_absent=false preserved without modification','scoped_exact_documented_exception':exact,'remaining_warning_disposition':'Missing-plugin descriptor, UJITSO-service and emergency-thread messages remain disclosed and not generically whitelisted as harmless. They did not prevent the observed finite CPU query results with both known-intersection and25 positive floor rays. GPU fallback explicitly disclaims GPU/particle/deformable collision.','warning_lines':warning_lines,'saved_options':options,'measured_floor_y_range_m':[min(y),max(y)],'all_bound_inputs_unchanged':all(sha(cap/x)==h for x,h in hashes.items()),'scope':'Read-only independent review of retained startup log and actual USD/source/numeric readback. No solver, renderer, model, asset edit, verdict override or final-retention terminal attestation.','limits':['This is independent read-only producer/source/preflight review, not the actual native Validation review or a physical-task verdict.','The exact documented initialization error exemption does not extend to any other error string or invalid-configuration warning.','Positive controls establish populated initial cooked query geometry, not independent proof that each cooking parameter was honored or that drawer/body clearance is globally valid.','This permits interpreting the measured CPU preflight; it does not establish task success, successful native visual review or workflow acceptance.','Fresh task trials and any later native work remain separate; no terminal assertion is made.']}
record.update({'native_producer_closure_verified':native_result['status']=='passed','native_bundle_sha256':sha(bundle_path),'native_validator_code_sha256':native_code_sha,'asset_dependency_manifest':dependency_before,'native_trajectory_recomputed':measured,'visual_frames':frames,'independent_visual_observation':'All eight actual retained PNGs were directly viewed by this independent reviewer. Nonblank textured cabinet and closed upper drawer remain visibly aligned without visible sag, rotation or separation. The oblique crop omits the base, rear and interior; it establishes no opening, payload or cavity behavior.','physical_change_boundary':'Fresh joint path and static body0 relationship changes from cabinet Xform to its original Mesh. Exact original five position/index arrays and identity world frames retained; all nonphysics authored Mesh attributes match09, as do body/collider values and resolved physics materials. This comparison does not independently establish original GLTF texture/material fidelity.','task_status':'Separate five-seed task evidence is not read or accepted by this receipt.','review_scope_status':'SUPPORTED_WITH_DISCLOSED_LIMITS' if all(x['pass'] for x in checks) else 'BLOCKED','blocker_check_names':[x['name'] for x in checks if not x['pass']]})
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'review_sha256':sha(a.output),'status':record['review_scope_status'],'blockers':record['blocker_check_names'],'inputs_unchanged':record['all_bound_inputs_unchanged']}))
