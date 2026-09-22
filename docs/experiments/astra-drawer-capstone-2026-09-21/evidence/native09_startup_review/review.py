"""Independent read-only native09 startup/source/property/query review."""
import argparse,datetime,hashlib,json,math,re
from pathlib import Path
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.root;cap=root/'capstone';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
relative=['runs/drawer_physics_09/physics.usda','runs/drawer_physics_03/physics.usda','evidence/native09_critical_cooking_log_audit.json','evidence/native09_cooking_log_review.json','evidence/native09_cooked_clearance_v1.json','evidence/native09_cooked_clearance_v1.log','evidence/authored_contract_drawer_physics_09.json','evidence/original_source_drawer_physics_09.json','evaluator/source_clear_v1/drawer_acceptance.json','review_native09_cooking_log.py','audit_cooking_log.py']
hashes={x:sha(cap/x) for x in relative};strict=json.loads((cap/relative[2]).read_text());prior=json.loads((cap/relative[3]).read_text());q=json.loads((cap/relative[4]).read_text());log=(cap/relative[5]).read_text();authored=json.loads((cap/relative[6]).read_text());source_receipt=json.loads((cap/relative[7]).read_text());spec=json.loads((cap/relative[8]).read_text());checks=[]
def check(n,v,detail=None):checks.append({'name':n,'pass':bool(v),'detail':detail})
expected='f38b48d703dd71904668edcae0ffe539ca6795978b44897ecbad00f7e973972d';check('exact_native09_asset',hashes[relative[0]]==expected)
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
record={'schema_version':1,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'review_script_sha256':sha(Path(__file__)),'input_hashes':hashes,'installed_runtime_overview_sha256':sha(doc),'documentation_lines':[137,138],'checks':checks,'bounded_cpu_preflight_supported':all(x['pass'] for x in checks),'blocking_findings':blockers,'strict_original_verdict':'critical_error_absent=false preserved without modification','scoped_exact_documented_exception':exact,'remaining_warning_disposition':'Missing-plugin descriptor, UJITSO-service and emergency-thread messages remain disclosed and not generically whitelisted as harmless. They did not prevent the observed finite CPU query results with both known-intersection and25 positive floor rays. GPU fallback explicitly disclaims GPU/particle/deformable collision.','warning_lines':warning_lines,'saved_options':options,'measured_floor_y_range_m':[min(y),max(y)],'all_bound_inputs_unchanged':all(sha(cap/x)==h for x,h in hashes.items()),'scope':'Read-only independent review of retained startup log and actual USD/source/numeric readback. No solver, renderer, model, asset edit, verdict override or final-retention terminal attestation.','limits':['The exact documented initialization error exemption does not extend to any other error string or invalid-configuration warning.','Positive controls establish populated initial cooked query geometry, not independent proof that each cooking parameter was honored or that drawer/body clearance is globally valid.','This permits interpreting the measured CPU preflight; it does not establish task success, successful native visual review or workflow acceptance.','Fresh task trials and any later native work remain separate; no terminal assertion is made.']}
a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'review_sha256':sha(a.output),'bounded_cpu_preflight_supported':record['bounded_cpu_preflight_supported'],'inputs_unchanged':record['all_bound_inputs_unchanged'],'floor_y_range':record['measured_floor_y_range_m']}))
