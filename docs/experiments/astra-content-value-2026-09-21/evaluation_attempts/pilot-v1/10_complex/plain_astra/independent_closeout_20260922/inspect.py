import datetime,hashlib,json,os,sys
from pathlib import Path
r=Path('/opt/astra-content-value-20260921')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def record(p):return {'path':str(p.relative_to(r)),'sha256':sha(p),'bytes':p.stat().st_size}
case='10_complex';run=r/'runs/pilot-v1'/case/'plain_astra';folder=r/'evaluations/pilot-v1'/case/'plain_astra';corrected=r/'evaluation_adjudications/pilot-v1'/case/'plain_astra/source_distance_v1_1';snapshot=r/'evaluator/case08_10/frozen'/case;inventory=r/'evaluator/case08_10/references'/case/'source_inventory.json';task=r/'protocol/tasks'/f'{case}.json'
frozen=load(snapshot/'frozen_manifest.json');v1=load(folder/'acceptance.json');sub=load(run/'submission.json');manifest=load(run/'output_manifest.json');mapped={x['path']:x for x in manifest['files']};scene=run/sub['final_scene'];bindings=run/sub['bindings'];checks=[]
def check(name,passed):checks.append({'name':name,'passed':bool(passed)})
check('task_frozen_snapshot_binding',load(task)['private_evaluator_snapshot_sha256']==sha(snapshot/'frozen_manifest.json'))
code_records=[]
for name,expected in frozen['code_sha256'].items():
 p=snapshot/name;actual=sha(p);check('frozen_code:'+name,actual==expected);code_records.append(record(p))
check('reference_inventory_frozen_binding',sha(inventory)==frozen.get('reference_inventory_sha256',frozen.get('source_inventory_sha256')))
for key,p in [('scene',scene),('bindings',bindings)]:check('author_manifest:'+key,sha(p)==mapped[str(p.relative_to(run))]['sha256'])
check('frozen_scene_binding',sha(scene)==v1['input_sha256']);check('frozen_bindings_binding',sha(bindings)==v1['bindings_sha256']);check('frozen_inventory_binding',sha(inventory)==v1['inventory_sha256'])
seed_rows=[];physics_files=[]
for seed in [11,23,47,83,131]:
 p=folder/f'seed_{seed}.json';x=load(p);complete=x.get('completed_steps')==x.get('expected_steps')==2880 and 'infrastructure_error' not in x
 check(f'seed_complete:{seed}',complete);check(f'seed_finite:{seed}',x.get('finite_states_and_contacts') is True)
 physics_files.append(record(p));seed_rows.append({'seed':seed,'completed_steps':x.get('completed_steps'),'expected_steps':x.get('expected_steps'),'complete':complete,'finite_states_and_contacts':x.get('finite_states_and_contacts'),'only_control_writes':x.get('only_control_writes'),'max_joint_closure_m':x.get('max_joint_closure_m'),'max_joint_axis_error_rad':x.get('max_joint_axis_error_rad'),'max_contact_penetration_m':x.get('max_contact_penetration_m')})
for name in ['runtime_config.json','instrumented.usda']:physics_files.append(record(folder/name))
non_distance=[x for x in v1['checks'] if not x['name'].startswith('source_surface_retained:')];check('all_frozen_non_distance_checks_pass',all(x['passed'] for x in non_distance))
adjudication=None
if (corrected/'acceptance.json').exists():
 a=load(corrected/'acceptance.json');q=r/'evaluator/errata/source_distance_v1_1/qualification_r2.json';worker=q.with_name('worker.py');qual=load(q)
 for name,passed in [('corrected_v1_binding',a['v1_acceptance_sha256']==sha(folder/'acceptance.json')),('corrected_worker_binding',a['worker_sha256']==sha(worker)),('corrected_qualified_worker_binding',qual['worker_sha256']==sha(worker)),('corrected_qualification_binding',a['qualification_sha256']==sha(q)),('qualification_passed',qual['all_passed'] is True),('corrected_scene_binding',a['input_scene_sha256']==sha(scene)),('corrected_bindings_binding',a['bindings_sha256']==sha(bindings)),('corrected_inventory_binding',a['source_inventory_sha256']==sha(inventory)),('corrected_frozen_binding',a['frozen_snapshot_manifest_sha256']==sha(snapshot/'frozen_manifest.json')),('corrected_no_changes',all(a[k] is False for k in ['tolerances_changed','frozen_files_modified','scored_artifacts_modified'])),('corrected_native_complete',a['native_physics']['all_five_complete'] is True)]:check(name,passed)
 for path,expected in a['native_physics']['configuration_and_result_hashes'].items():check('corrected_reused_native_hash:'+str(Path(path).relative_to(r)),sha(Path(path))==expected)
 check('corrected_non_distance_checks_exact',[x for x in a['checks'] if not x['name'].startswith('source_surface_retained:')]==non_distance)
 adjudication={'status':a['status'],'accepted':a['accepted'],'false_pass':a['false_pass'],'source_surface_checks_replaced':a['source_surface_checks_replaced'],'geometry_passed':sum(x['v1_1_passed'] for x in a['geometry_parts']),'native_physics_origin':a['native_physics_origin'],'all_five_native_complete':a['native_physics']['all_five_complete'],'concrete_failures':a['concrete_failures'],'inconclusive_checks':a['inconclusive_checks'],'acceptance':record(corrected/'acceptance.json'),'provenance':record(corrected/'provenance.json'),'qualification':record(q),'worker':record(worker)}
result={'version':'independent_pair_closeout10_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_id':case,'arm':'plain_astra','scored_or_frozen_files_modified':False,'corrected_receipt_ready':adjudication is not None,'all_integrity_checks_passed':all(x['passed'] for x in checks),'checks':checks,'frozen_status':v1['status'],'frozen_failures_only_source_distance':all(x.startswith('source_surface_retained:') for x in v1['concrete_failures']),'frozen_non_distance_checks':len(non_distance),'native_seeds':seed_rows,'native_evidence':physics_files,'inputs':[record(p) for p in [run/'output_manifest.json',run/'submission.json',scene,bindings,task,snapshot/'frozen_manifest.json',inventory,folder/'acceptance.json',folder/'evaluation_launch.json']],'frozen_code':code_records,'corrected':adjudication,'scope':'Read-only byte, configuration, complete-native-trajectory and acceptance-chain verification. No scoring or physics rerun.'}
print(json.dumps(result,indent=2))
