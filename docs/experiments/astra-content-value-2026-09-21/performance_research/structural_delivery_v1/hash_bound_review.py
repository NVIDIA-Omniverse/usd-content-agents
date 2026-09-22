"""Read-only receipt/code review binding; no USD authoring, simulation or scoring."""
from pathlib import Path
import argparse,datetime,hashlib,json
CASES=['07_robot_arm','08_excavator','09_printer','10_complex']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def review(root,out):
 root=root.resolve();directory=root/'performance_research/structural_delivery_v1';refs=[]
 def bind(p):
  digest=sha(p);r={'path':str(p.relative_to(root)),'sha256':digest};refs.append(r);return r
 auditor=bind(directory/'audit.py');qualifier=bind(directory/'qualify.py');qualifications={};qrows=[]
 expected={'positive','geometry_only','disabled_body','kinematic_body','no_collider','disabled_collider','wrong_joint_kind','disabled_joint','missing_body_binding','missing_joint_binding','fake_body_path','fake_joint_path','duplicate_body_role','duplicate_joint_role','unresolved_dependency_not_decisive'}
 for host in ['codex','pair']:
  p=directory/f'qualification_{host}/qualification.json';q=read(p);ref=bind(p)
  assert q['all_passed'] and q['synthetic_only'] and q['test_count']==15
  assert {t['name'] for t in q['tests']}==expected and all(t['passed'] for t in q['tests'])
  assert q['auditor_sha256']==auditor['sha256'] and q['qualifier_sha256']==qualifier['sha256']
  for t in q['tests']:
   if 'scene_sha256' in t:
    scene=p.parent/(t['name']+'.usda');assert sha(scene)==t['scene_sha256'];bind(scene)
    assert t['actual_non_delivery']==(t['name']!='positive')
   else:assert t['name']=='unresolved_dependency_not_decisive' and 'Unresolved dependencies' in t['diagnostic']
  qualifications[ref['sha256']]=q;qrows.append({'host':host,'qualification':ref,'tests':15,'all_passed':True,'usd_version':q['usd_version']})
 cases=[]
 for c in CASES:
  p=directory/(c+'_content_agents.json');receipt=read(p);ref=bind(p)
  assert receipt['version']=='supplementary_structural_delivery_v1' and receipt['case_id']==c and receipt['arm']=='content_agents'
  assert receipt['auditor_sha256']==auditor['sha256'] and receipt['qualification_sha256'] in qualifications
  assert receipt['supplementary_status']=='DECISIVE_NON_DELIVERY' and receipt['inputs_unchanged']
  assert not receipt['unresolved_dependencies'] and not receipt['scored_files_modified'] and not receipt['original_evaluations_modified']
  assert all(receipt['schema_counts'][k]==0 for k in ['rigid_bodies','joints','colliders'])
  for item in [*receipt['inputs'].values(),*receipt['composed_layer_hashes']]:
   target=root/item['path'];assert sha(target)==item['sha256'],target;bind(target)
  run=root/'runs/pilot-v1'/c/'content_agents';manifest=read(root/receipt['inputs']['output_manifest.json']['path']);mapping={x['path']:x for x in manifest['files']}
  task=read(root/receipt['inputs']['public_task']['path']);launch=read(root/receipt['inputs']['launch.json']['path']);submission=read(root/receipt['inputs']['submission.json']['path'])
  assert task['case_id']==c and task['frozen'] is True and launch['task_sha256']==receipt['inputs']['public_task']['sha256']
  assert 'Intended moving bodies must be dynamic/nonkinematic' in task['submission_contract']
  public=task['public_acceptance']
  for name in ['required_body_roles','required_joint_roles','joint_kinds']:assert public[name]==receipt[name]
  for name,key in [('scene','final_scene'),('bindings','bindings')]:
   target=root/receipt['inputs'][name]['path'];assert target.resolve()==(run/submission[key]).resolve() and target.resolve().is_relative_to(run.resolve())
   rel=str(target.relative_to(run));assert mapping[rel]['sha256']==receipt['inputs'][name]['sha256']
  for item in receipt['composed_layer_hashes']:
   rel=str((root/item['path']).relative_to(run));assert mapping[rel]['sha256']==item['sha256']
  failures={x['name'] for x in receipt['checks'] if not x['passed']}
  required_failures={'required_moving_body_is_dynamic:'+role for role in public['required_body_roles'] if role!='base'} | {'required_joint_role_has_enabled_api:'+role for role in public['required_joint_roles']}
  assert required_failures<=failures and set(receipt['concrete_missing_delivery_checks'])==failures
  assert submission['claimed_accepted'] is False
  cases.append({'case_id':c,'arm':'content_agents','receipt':ref,'public_task':receipt['inputs']['public_task'],'scene':receipt['inputs']['scene'],'bindings':receipt['inputs']['bindings'],'counts':receipt['schema_counts'],'required_body_roles':public['required_body_roles'],'required_joint_roles':public['required_joint_roles'],'joint_kinds':public['joint_kinds'],'supported_rejection_basis':'Actual composed scene contains no rigid bodies or joints although the frozen public task requires dynamic moving bodies and specified joints. Zero colliders corroborates missing physics.','supported_task_outcome':'FAIL_MANDATORY_PHYSICS_ABSENT','source_fidelity_status':'NOT_ASSESSED_BY_THIS_REVIEW','native_runtime_status':'NOT_ASSESSED_BY_THIS_REVIEW','claimed_accepted':False,'demonstrated_false_pass':False})
 # Verify identity again after every read; no mutable output is in the input set.
 assert all(sha(root/r['path'])==r['sha256'] for r in refs)
 result={'schema_version':1,'reviewed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AGREE_NARROW_STATIC_REJECTION','auditor':auditor,'qualifier':qualifier,'review_source':{'path':str(Path(__file__).resolve().relative_to(root)),'sha256':sha(Path(__file__).resolve())},'qualifications':qrows,'cases':cases,'all_reviewed_input_hashes_match':True,'contract_basis':'Required body/joint roles and dynamic/nonkinematic requirement were disclosed in each launched frozen public task; missing all such APIs is a concrete necessary-condition failure independent of geometry distance and solver availability.','qualification_assessment':'Both hosts pass the same15 meaningful synthetic fixtures. Positive API presence is distinguished from geometry-only, disabled/kinematic bodies, absent/disabled colliders, wrong/disabled joints, missing/fake/duplicate bindings. Unresolved composition explicitly raises instead of proving absence. Tests qualify structural observations only, not simulation.','recommended_integration':{'scope':'Only these four Content Agents receipts, bound to the reviewed auditor/qualification bytes and original submission/task/manifest/layers.','decision':'May report FAIL for mandatory physical content absent even while source comparison is unfinished or inconclusive. Preserve the original frozen evaluator result and separate supplementary receipt.','necessary_guards':['Require supported version, case/arm, auditor and qualification hashes, public task/launch consistency, scene/bindings/output-manifest hash binding, complete composed layer closure and unchanged inputs.','For this narrow integration require zero actual rigid bodies and zero actual joints; do not rely on supplementary_status alone.','Never convert importer/parser/hash/dependency/qualification exceptions into concrete failure.','Do not promote acceptance, claim source fidelity or native dynamics qualification, or count a false pass when claimed_accepted=false.','Any separately authorized stop must preserve incomplete attempts and a termination receipt; never call a stopped evaluation complete or claim its source fidelity. This review stops no jobs and changes no scored artifacts.']},'limitations':['No independent solver run was performed by this review.','The collider-descendant presence test can miss collider ownership under another nested rigid body; its positive result is expressly not acceptance. This does not affect these zero-schema scenes.','Outer manifest/path guards are code-reviewed and actual-receipt checked; the15 fixture set primarily exercises observe(), not every wrapper exception.','This is a post-run independent necessary-condition observation, not a rewrite of the frozen evaluator.'],'frozen_files_modified':False,'scored_artifacts_modified':False,'existing_jobs_changed':False,'simulation_runs':0}
 assert not out.exists(),f'Refusing overwrite: {out}'
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'review':str(out),'sha256':sha(out),'cases':len(cases),'status':result['status']}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();review(a.root,a.output)
