"""Independent read-only protocol/measurement review. Does not execute the solver."""
from pathlib import Path
import argparse,datetime,hashlib,json,math
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();R=a.root
E=R/'evaluations/pilot-v1/07_robot_arm/plain_astra';U=R/'runs/pilot-v1/07_robot_arm/plain_astra';F=R/'evaluator/general/frozen/07_robot_arm';D=R/'evaluator/errata/robot_arm_controller_review_v1';N=D/'no_jitter'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def ref(p):return {'path':str(p.relative_to(R)),'sha256':sha(p)}
task=read(R/'protocol/tasks/07_robot_arm.json');contract=task['public_acceptance'];c=read(E/'runtime_config.json');r=read(N/'receipt.json');nom=read(N/'seed11_no_jitter.json');checks=read(N/'observer_checks.json');acc=read(E/'acceptance.json');report=read(R/'report/robot_arm_false_pass_evidence.json');sub=read(U/'submission.json');role=c['joint_roles']['joint5'];tol=contract['common']['angle_tolerance_rad']
assert r['diagnostic_sha256']==sha(D/'diagnostic.py') and r['native_result_sha256']==sha(N/'seed11_no_jitter.json')
assert r['before']==r['after']=={'solver_sha256':sha(F/'solver.py'),'config_sha256':sha(E/'runtime_config.json')}
assert r['replacement_counts']=={'effort_multiplier':1,'startup_perturbations':72}
assert c['contract']['common']==contract['common']
for key in ['control_default','target_offsets_rad']:assert c['contract'][key]==contract['task_parameters'][key]
assert c.get('external_load_scale',1)==1
assert sub['claimed_accepted'] and sha(U/'submission.json')==report['author_submission_sha256']
assert sha(E/'acceptance.json')==report['frozen_acceptance_sha256'] and sha(R/'protocol/tasks/07_robot_arm.json')==report['public_task_sha256']
scene=(U/sub['final_scene']).resolve();bindings=(U/sub['bindings']).resolve();assert scene.is_relative_to(U) and bindings.is_relative_to(U)
assert sha(scene)==acc['input_sha256'] and sha(bindings)==acc['bindings_sha256']
fm=read(F/'frozen_manifest.json');frozen_checks={k:sha(F/k)==v for k,v in fm['code_sha256'].items()};assert all(frozen_checks.values())
def measure(result):
 rows=[(x['t'],x['joints'][role]) for x in result['trace'] if 5.7<=x['t']<6.5]
 worst=max(rows,key=lambda x:abs(x[1]['q_unwrapped']-x[1]['target']))
 return {'samples':len(rows),'window_s':[5.7,6.5],'end_exclusive':True,'worst_time_s':worst[0],'actual_angle_rad':worst[1]['q_unwrapped'],'target_angle_rad':worst[1]['target'],'max_error_rad':abs(worst[1]['q_unwrapped']-worst[1]['target']),'tolerance_rad':tol,'margin_over_tolerance_rad':abs(worst[1]['q_unwrapped']-worst[1]['target'])-tol,'external_torque_at_worst_Nm':worst[1]['external_effort'],'complete_steps':result['completed_steps'],'expected_steps':result['expected_steps']}
rows=[]
for seed in [11,23,47,83,131]:
 f=E/f'seed_{seed}.json';result=read(f);m=measure(result);check=next(x for x in acc['checks'] if x['name']==f'seed{seed}:target_holding_under_load:joint5')
 assert m['complete_steps']==m['expected_steps']==2880 and m['max_error_rad']>tol and abs(m['max_error_rad']-check['evidence']['max_error'])<1e-14
 rows.append({'seed':seed,'evidence':ref(f),'measurement':m})
m=measure(nom);check=next(x for x in checks if x['name']=='seed11:target_holding_under_load:joint5');assert m['complete_steps']==m['expected_steps']==2880 and m['max_error_rad']>tol and abs(m['max_error_rad']-check['evidence']['max_error'])<1e-14
expected_load=-min(.01*contract['task_parameters']['control_default']['max_effort'],nom['controller_effective_inertia'][role]);assert abs(m['external_torque_at_worst_Nm']-expected_load)<1e-14
assert all(j.get('external_effort',0)==0 for x in nom['trace'] if x['t']<.05 for j in x['joints'].values())
assert next(x['passed'] for x in checks if x['name']=='seed11:independent_freefall_witness')
result={'schema_version':1,'case_id':'07_robot_arm','arm':'plain_astra','reviewer':'Independent Astra Ultra evaluator reviewer','reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AGREE_CONCRETE_PHYSICAL_FALSE_PASS_WITH_PROTOCOL_EXCLUSION','agrees_demonstrated_false_pass':True,'eligible_matched_pair':False,'scope':'The author claimed acceptance, but the submitted artifact fails a disclosed fixed-controller holding criterion. This does not certify experimental eligibility or establish a general robot/hardware failure.','basis':'All five frozen native traces independently recompute to joint5 holding errors above5degrees. A fresh complete nominal-controller diagnostic retains this failure after removing both undisclosed stochastic terms; the geometry-distance numerical bug is outside this native angle measurement.','nominal_measurement':m,'frozen_measurements':rows,'public_contract_checks':{'same_dt_duration_gravity_tolerances':True,'same_disclosed_gains_effort_cap_and_target_offsets':True,'same_inertia_scaled_opposing_load':True,'nominal_startup_external_effort_zero':True,'gravity_witness_passes':True,'only_nominal_diagnostic_failed_check':check['name']},'wrapper_review':'The reviewed wrapper replaces module.np only for the imported frozen solver. Its Generator.uniform intercepts scalar[.98,1.02]→1 and[-.005,.005]→0; all other NumPy calls delegate to the original module. There are no other RNG uses in the frozen solver, no scene/config changes, no state writes added, and no monkeypatch to the native library. Replacements total1+72=one run scale plus12 steps times6 joints.','evidence':[ref(R/'report/robot_arm_false_pass_evidence.json'),ref(R/'protocol/tasks/07_robot_arm.json'),ref(U/'submission.json'),ref(scene),ref(bindings),ref(E/'acceptance.json'),ref(E/'runtime_config.json'),ref(Path(c['scenario'])),ref(F/'frozen_manifest.json'),ref(F/'solver.py'),ref(F/'evaluate.py'),ref(D/'diagnostic.py'),ref(N/'receipt.json'),ref(N/'seed11_no_jitter.json'),ref(N/'observer_checks.json')],'frozen_code_hash_checks':frozen_checks,'limitations':['The no-jitter run is one unscored diagnostic, not replacement all-five acceptance. It establishes a concrete nominal failure and does not promote any asset.','The evaluator chooses inertia-scaled fixed gains; failure is specific to this disclosed controller/task, not proof that every possible controller or real Thor hardware fails.','The separate author cross-case exposure violation excludes an eligible matched-pair attribution. No causal claim is made about the exposure.','Known source-distance geometry rejections are not used as false-pass evidence. Geometry correction remains a separate versioned adjudication.'],'frozen_files_modified':False,'scored_artifacts_modified':False,'acceptance_files_modified':False,'raw_auth_or_model_reasoning_included':False}
q=D/'independent_review.json';q.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'path':str(q),'sha256':sha(q),'agrees':True,'nominal_error_rad':m['max_error_rad']}))
