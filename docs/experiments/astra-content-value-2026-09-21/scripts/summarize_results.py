#!/usr/bin/env python3
"""Aggregate measured pilot outcomes; never infer success from author claims."""
import argparse,csv,hashlib,json,math
from pathlib import Path
ARMS=['plain_astra','content_agents']
STRUCTURAL_AUDITOR_SHA='387c6608092d1fe0ad3464c677b62451f983e9151443fa899717be879c79fc22'
STRUCTURAL_QUALIFIER_SHA='82e5f900ef170640bbf7e3d629e117cc5520eef1fd228524156715b72f8cbedd'
# Four independently reviewed, immutable observations only. No general rule
# permits an arbitrary future structural receipt to change a result.
STRUCTURAL_RECEIPTS={
 '07_robot_arm':('371b02203288f0e1e9e8fe7d96b81b7f25ec5d9fa67eed955db1fcff7859dd16','qualification_codex','e297a618664654c82f9f40361a0705bcb4bd1d2219bdd73664c1a2bfdee46f62'),
 '08_excavator':('14b0c3fcd05eb04a82ca0a39534ed394d3df106a4c99d610a9adfe2c87aa29f7','qualification_codex','e297a618664654c82f9f40361a0705bcb4bd1d2219bdd73664c1a2bfdee46f62'),
 '09_printer':('dbb94fdf757ba38fee0d26797fdea6323cc2327a91b5fa5e94975be19b6e0ed5','qualification_pair','7b43248633411ee34fb7188bf7c5f34ce678ad5d4bde8c82947a5756f3d7e24b'),
 '10_complex':('90fd749a8ea5357b6e6ec12a38f628f2a7988afe1578e00d737fa65fcb628731','qualification_pair','7b43248633411ee34fb7188bf7c5f34ce678ad5d4bde8c82947a5756f3d7e24b'),
}
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path,default=None):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return default
def outcome(evaluation):
 accepted=evaluation.get('accepted',evaluation.get('pass',False)) is True
 raw=evaluation.get('status','').lower()
 return ('PASS' if accepted else ('FAIL' if raw in ['fail','not_accepted','non_submission'] else 'INCONCLUSIVE')),accepted
def structural_delivery_evidence(root,run_id,case,arm):
 """Validate a pinned mandatory-physics-absence observation; never grant PASS."""
 if run_id!='pilot-v1' or arm!='content_agents' or case not in STRUCTURAL_RECEIPTS:return None
 receipt_path=root/'evaluation_adjudications'/run_id/case/arm/'structural_delivery_v1/receipt.json'
 if not receipt_path.is_file():return None
 def require(condition,reason):
  if not condition:raise ValueError(reason)
 def exact(path,expected,reason):
  require(path.is_file() and not path.is_symlink() and digest(path)==expected,reason)
 try:
  expected_receipt,qualification_folder,expected_qualification=STRUCTURAL_RECEIPTS[case]
  exact(receipt_path,expected_receipt,'unapproved_or_changed_supplementary_receipt')
  receipt=read(receipt_path);package=root/'performance_research/structural_delivery_v1'
  exact(package/'audit.py',STRUCTURAL_AUDITOR_SHA,'qualified_auditor_changed_or_missing')
  exact(package/'qualify.py',STRUCTURAL_QUALIFIER_SHA,'qualifier_changed_or_missing')
  qualification_path=package/qualification_folder/'qualification.json'
  exact(qualification_path,expected_qualification,'qualification_changed_or_missing')
  qualification=read(qualification_path)
  require(qualification.get('all_passed') is True and qualification.get('test_count')==15 and len(qualification.get('tests',[]))==15 and all(x.get('passed') is True for x in qualification['tests']),'qualification_not_passed')
  require(qualification.get('auditor_sha256')==STRUCTURAL_AUDITOR_SHA and qualification.get('qualifier_sha256')==STRUCTURAL_QUALIFIER_SHA,'qualification_code_binding_mismatch')
  require(receipt.get('auditor_sha256')==STRUCTURAL_AUDITOR_SHA and receipt.get('qualification_sha256')==expected_qualification,'receipt_qualification_binding_mismatch')
  require(receipt.get('version')=='supplementary_structural_delivery_v1' and receipt.get('case_id')==case and receipt.get('arm')==arm,'receipt_scope_mismatch')
  require(receipt.get('supplementary_status')=='DECISIVE_NON_DELIVERY','receipt_not_non_delivery')
  require(receipt.get('inputs_unchanged') is True and receipt.get('scored_files_modified') is False and receipt.get('original_evaluations_modified') is False,'receipt_mutation_or_integrity_flags')
  require(receipt.get('source_fidelity_status')=='NOT_EVALUATED' and receipt.get('native_physics_status')=='NOT_EVALUATED','supplementary_scope_incorrect')
  for kind,key in [('rigid_bodies','bodies'),('joints','joints'),('colliders','colliders')]:
   count=receipt.get('schema_counts',{}).get(kind)
   require(type(count) is int and count==0 and receipt.get(key)==[],'not_global_zero_physics_schemas')
  require(receipt.get('unresolved_dependencies')==[],'unresolved_submitted_dependencies')
  run_relative=Path('runs')/run_id/case/arm;run=root/run_relative;inputs=receipt['inputs']
  # These files are deliberately published byte-for-byte after privacy review.
  for key,relative in [('output_manifest.json',run_relative/'output_manifest.json'),('submission.json',run_relative/'submission.json'),('public_task',Path('protocol/tasks')/(case+'.json'))]:
   require(inputs[key]['path']==str(relative),'unexpected_'+key+'_path')
   exact(root/relative,inputs[key]['sha256'],'stale_or_changed_'+key)
  task=read(root/inputs['public_task']['path']);submission=read(run/'submission.json');manifest=read(run/'output_manifest.json')
  public=task.get('public_acceptance',{});required=public.get('required_joint_roles')
  require(task.get('frozen') is True and task.get('case_id')==case,'task_not_frozen_for_case')
  require(isinstance(required,list) and len(required)>0 and all(isinstance(x,str) and x for x in required),'task_has_no_mandatory_joints')
  require(required==receipt.get('required_joint_roles') and public.get('required_body_roles')==receipt.get('required_body_roles') and public.get('joint_kinds')==receipt.get('joint_kinds'),'public_role_requirements_changed')
  # Launch is projected publicly. Check published bytes against the publication
  # manifest, then bind its retained-original digest and unchanged task field.
  launch_relative=run_relative/'launch.json';launch_path=root/launch_relative
  require(inputs['launch.json']['path']==str(launch_relative),'unexpected_launch_path')
  launch_sha=digest(launch_path);launch_binding='exact_retained_bytes'
  if launch_sha!=inputs['launch.json']['sha256']:
   publication=read(root/'publication_manifest.json',{});entries=[x for x in publication.get('files',[]) if x.get('path')==str(launch_relative)]
   require(len(entries)==1 and entries[0].get('projection') is True and entries[0].get('sha256')==launch_sha and entries[0].get('original_retained_sha256')==inputs['launch.json']['sha256'],'launch_projection_not_hash_bound')
   launch_binding='verified_public_projection'
  require(read(launch_path).get('task_sha256')==inputs['public_task']['sha256'],'launch_task_binding_mismatch')
  # Frozen evaluator bytes remain available publicly; no CAD arrays required.
  family='case08_10' if case in ['08_excavator','10_complex'] else 'general'
  snapshot=root/'evaluator'/family/'frozen'/case;freeze_path=snapshot/'frozen_manifest.json'
  exact(freeze_path,task['private_evaluator_snapshot_sha256'],'frozen_manifest_changed')
  frozen=read(freeze_path);require(bool(frozen.get('code_sha256')),'empty_frozen_code_manifest')
  for name,expected in frozen['code_sha256'].items():
   require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe_frozen_code_path')
   exact(snapshot/name,expected,'frozen_code_changed:'+name)
  entries=manifest.get('files',[]);mapping={x['path']:x['sha256'] for x in entries}
  require(len(entries)==len(mapping),'duplicate_output_manifest_paths')
  retained_only=[]
  def manifest_bound(relative,expected):
   path=Path(relative);require(not path.is_absolute() and '..' not in path.parts,'unsafe_submitted_path')
   local=str(path.relative_to(run_relative));require(mapping.get(local)==expected,'scene_or_layer_manifest_digest_mismatch')
   p=root/path
   if p.exists():exact(p,expected,'present_scene_binding_or_layer_changed')
   else:retained_only.append(str(path))
  for key,submission_key in [('scene','final_scene'),('bindings','bindings')]:
   relative=submission.get(submission_key);require(isinstance(relative,str) and relative,'no_submitted_scene_bindings_pair')
   require(inputs[key]['path']==str(run_relative/relative),'submission_pointer_mismatch')
   manifest_bound(inputs[key]['path'],inputs[key]['sha256'])
  layers=receipt.get('composed_layer_hashes');require(isinstance(layers,list) and layers,'missing_composed_layer_bindings')
  for layer in layers:manifest_bound(layer['path'],layer['sha256'])
  return {'valid':True,'reason':'hash_bound_absence_of_all_mandatory_physics_schemas','receipt_sha256':expected_receipt,
          'source_fidelity_status':'NOT_EVALUATED','native_physics_status':'NOT_EVALUATED','launch_binding':launch_binding,
          'retained_only_input_paths':sorted(set(retained_only))}
 except (OSError,ValueError,TypeError,KeyError,AttributeError) as exc:
  return {'valid':False,'reason':str(exc),'receipt_sha256':digest(receipt_path) if receipt_path.is_file() else None}
def structural_delivery_result(evidence):
 """Stable semantic fields; local/public validation details stay in the helper."""
 if evidence is None:return None
 return {key:evidence[key] for key in ['valid','reason','receipt_sha256','source_fidelity_status','native_physics_status'] if key in evidence}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-id',default='pilot-v1');p.add_argument('--rates',type=Path);a=p.parse_args()
 dataset=read(a.root/'protocol/dataset.json');rates=read(a.rates,{}) if a.rates else {};rows=[]
 for case in dataset['assets']:
  for arm in ARMS:
   run=a.root/'runs'/a.run_id/case['case_id']/arm;execution=read(run/'execution.json');submission=read(run/'submission.json',{});audit=read(run/'model_usage_audit.json',{})
   evaluation_path=a.root/'evaluations'/a.run_id/case['case_id']/arm/'acceptance.json'
   if not evaluation_path.exists():evaluation_path=evaluation_path.with_name('report.json')
   evaluation=read(evaluation_path)
   status='PENDING';accepted=False;concrete=False;distance_correction_pending=False
   if evaluation:
    status,accepted=outcome(evaluation);concrete=status=='FAIL'
   elif execution and not submission:status='NO_SUBMISSION';concrete=True
   frozen_evaluator_status=status
   if evaluation and evaluation.get('measurement_owner')=='harness_bounded_attempt_closeout' and evaluation.get('frozen_evaluator_verdict_produced') is False:
    frozen_evaluator_status='NO_VERDICT'
   if evaluation and case['case_id']!='01_drawer':
    failures=evaluation.get('concrete_failures',[])
    distance_failures=[x for x in failures if x.startswith('source_surface_retained:')]
    if distance_failures:
     distance_correction_pending=True
     if len(distance_failures)==len(failures):status='INCONCLUSIVE';accepted=False;concrete=False
   adjudication=read(a.root/'evaluation_adjudications'/a.run_id/case['case_id']/arm/'source_distance_v1_1/acceptance.json')
   adjudicated_status=None;adjudication_hash_bound=None
   if adjudication:
    adjudication_hash_bound=bool(evaluation and adjudication.get('v1_acceptance_sha256')==hashlib.sha256(evaluation_path.read_bytes()).hexdigest())
    if adjudication_hash_bound:
     adjudicated_status,corrected_accepted=outcome(adjudication)
     if corrected_accepted and not (adjudication.get('native_physics',{}).get('all_five_complete') is True and adjudication.get('frozen_files_modified') is False and adjudication.get('scored_artifacts_modified') is False and adjudication.get('tolerances_changed') is False):
      adjudicated_status='INCONCLUSIVE';corrected_accepted=False
     status,accepted=adjudicated_status,corrected_accepted;concrete=status=='FAIL';distance_correction_pending=False
    else:status='INCONCLUSIVE';accepted=False;concrete=False
   # A later measurement review may conservatively withdraw a rejection, but
   # cannot confer acceptance or silently replace the frozen measurement.
   review=read(a.root/'evaluation_adjudications'/a.run_id/case['case_id']/arm/'measurement_review/assessment.json')
   measurement_review_applied=False
   if review and status=='FAIL':
    adjudication_path=a.root/'evaluation_adjudications'/a.run_id/case['case_id']/arm/'source_distance_v1_1/acceptance.json'
    latest=adjudication if adjudication_hash_bound else evaluation
    failures=set((latest or {}).get('concrete_failures',[]))
    bound=bool(evaluation and review.get('v1_acceptance_sha256')==hashlib.sha256(evaluation_path.read_bytes()).hexdigest())
    if adjudication:
     bound=bound and adjudication_hash_bound and review.get('prior_adjudication_sha256')==hashlib.sha256(adjudication_path.read_bytes()).hexdigest()
    if bound and failures and failures<=set(review.get('covered_rejection_names',[])) and review.get('remaining_supported_rejections')==[] and review.get('status')=='inconclusive' and review.get('accepted') is False and review.get('frozen_files_modified') is False and review.get('scored_artifacts_modified') is False:
     status='INCONCLUSIVE';accepted=False;concrete=False;measurement_review_applied=True
   # A separate, pinned observation can prove missing mandatory delivery while
   # source/native checks remain unassessed. It never grants or overrides PASS.
   structural_delivery=structural_delivery_evidence(a.root,a.run_id,case['case_id'],arm)
   pre_structural_delivery_status=status;structural_delivery_applied=False
   if structural_delivery and structural_delivery['valid'] and status in ['PENDING','INCONCLUSIVE','FAIL'] and frozen_evaluator_status!='PASS' and adjudicated_status!='PASS':
    status='FAIL';accepted=False;concrete=True;structural_delivery_applied=True
   contexts=audit.get('observed_turn_contexts',[])
   model_ok=bool(contexts) and all(c.get('model')=='gpt-6-astra' and c.get('reasoning_effort')=='ultra' for c in contexts)
   audit_dir=a.root/'audits'/a.run_id/case['case_id']/arm
   protocol=read(audit_dir/'protocol_audit.json') or read(run/'protocol_audit.json',{})
   independent_status=status
   independent_false_pass=submission.get('claimed_accepted') is True and concrete
   if protocol.get('valid') is False or (contexts and not model_ok):status='INVALID_PROTOCOL';accepted=False;concrete=False
   elif accepted and protocol.get('valid') is not True:status='PENDING_PROTOCOL_AUDIT';accepted=False;concrete=False
   usage=read(audit_dir/'usage_reconciled.json') or read(run/'usage_reconciled.json')
   if usage is None:
    sessions=audit.get('session_usage',[])
    usage={k:sum(s.get('total_token_usage',{}).get(k,0) for s in sessions) if sessions else None for k in ['input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens']}
    usage['complete']=False
   elapsed=execution.get('elapsed_seconds') if execution else None;gpu=execution.get('gpu_allocation_seconds') if execution else None
   cost=None
   token_values=[usage.get(k) for k in ['input_tokens','cached_input_tokens','output_tokens']]
   pricing_tokens_valid=all(isinstance(x,int) and not isinstance(x,bool) and x>=0 for x in token_values) and usage.get('cached_input_tokens',0)<=usage.get('input_tokens',-1)
   required=['input_usd_per_million','cached_input_usd_per_million','output_usd_per_million','gpu_usd_per_hour']
   if usage.get('complete') and pricing_tokens_valid and isinstance(gpu,(int,float)) and not isinstance(gpu,bool) and math.isfinite(gpu) and gpu>=0 and rates.get('rate_source') and rates.get('effective_date') and all(isinstance(rates.get(k),(int,float)) and not isinstance(rates[k],bool) and math.isfinite(rates[k]) and rates[k]>=0 for k in required):
    cost=((usage['input_tokens']-usage['cached_input_tokens'])*rates[required[0]]+usage['cached_input_tokens']*rates[required[1]]+usage['output_tokens']*rates[required[2]])/1e6+gpu/3600*rates[required[3]]
   rows.append({'case_id':case['case_id'],'asset':case['title'],'arm':arm,'status':status,'frozen_evaluator_status':frozen_evaluator_status,'adjudicated_evaluator_status':adjudicated_status,'adjudication_hash_bound':adjudication_hash_bound,'distance_correction_pending':distance_correction_pending,'independent_task_status':independent_status,'protocol_audit_valid':protocol.get('valid'),'accepted':accepted,'claimed_accepted':submission.get('claimed_accepted'),'demonstrated_false_pass':independent_false_pass,'elapsed_seconds':elapsed,'allocated_gpu_seconds':gpu,'timed_out':execution.get('timed_out') if execution else None,'repairs_declared':submission.get('repairs_used'),'human_interventions':execution.get('human_interventions') if execution else None,'human_review_seconds':execution.get('human_review_seconds') if execution else None,'observed_model_effort_match':model_ok if contexts else None,'model_contexts_observed':len(contexts),'input_tokens':usage.get('input_tokens'),'cached_input_tokens':usage.get('cached_input_tokens'),'output_tokens':usage.get('output_tokens'),'token_usage_complete':usage.get('complete',False),'cost_usd':cost})
   rows[-1]['measurement_review_applied']=measurement_review_applied
   rows[-1].update(structural_delivery_audit_applied=structural_delivery_applied,pre_structural_delivery_status=pre_structural_delivery_status,
                   structural_delivery_audit=structural_delivery_result(structural_delivery))
 global_audit=read(a.root/'report/global_protocol_audit.json',{'valid':None,'scope':'No global audit supplied'})
 for row in rows:
  pair=[x for x in rows if x['case_id']==row['case_id']]
  row['paired_author_audits_pass']=all(x['protocol_audit_valid'] is True and x['observed_model_effort_match'] is True for x in pair)
  row['global_protocol_verified']=global_audit.get('valid') is True
  row['matched_pair_protocol_eligible']=row['paired_author_audits_pass'] and row['global_protocol_verified']
 aggregate={}
 for arm in ARMS:
  group=[r for r in rows if r['arm']==arm];n=sum(r['accepted'] for r in group);completed=[r for r in group if r['elapsed_seconds'] is not None]
  total_time=sum(r['elapsed_seconds'] for r in completed);costs=[r['cost_usd'] for r in group];dollars=sum(costs) if all(c is not None for c in costs) else None
  aggregate[arm]={'accepted':n,'attempted':len(completed),'planned':len(group),'full_denominator_acceptance_lower_bound':n/len(group),'concrete_rejections':sum(r['status'] in ['FAIL','NO_SUBMISSION'] for r in group),'inconclusive':sum(r['status']=='INCONCLUSIVE' for r in group),'demonstrated_false_passes':sum(r['demonstrated_false_pass'] for r in group),'total_scored_wall_seconds':total_time,'scored_wall_seconds_per_accepted_asset':total_time/n if n else None,'total_allocated_gpu_seconds':sum(r['allocated_gpu_seconds'] or 0 for r in group),'human_interventions':sum(r['human_interventions'] or 0 for r in completed),'human_review_seconds':None,'total_cost_usd':dollars,'cost_usd_per_accepted_asset':dollars/n if n and dollars is not None else None}
  aggregate[arm].update({'runs_with_complete_token_usage':sum(r['token_usage_complete'] is True for r in group),'observed_input_tokens':sum(r['input_tokens'] or 0 for r in completed) if completed else None,'observed_cached_input_tokens':sum(r['cached_input_tokens'] or 0 for r in completed) if completed else None,'observed_output_tokens':sum(r['output_tokens'] or 0 for r in completed) if completed else None,'protocol_valid_runs':sum(r['protocol_audit_valid'] is True for r in group),'protocol_invalid_runs':sum(r['status']=='INVALID_PROTOCOL' for r in group)})
  physically_accepted=sum(r['independent_task_status']=='PASS' for r in group)
  aggregate[arm].update(independently_passing_assets=physically_accepted,wall_seconds_per_independently_passing_asset=total_time/physically_accepted if physically_accepted else None,cost_usd_per_independently_passing_asset=dollars/physically_accepted if physically_accepted and dollars is not None else None,matched_pair_protocol_eligible=sum(r['matched_pair_protocol_eligible'] for r in group),matched_pair_eligible_accepted=sum(r['accepted'] and r['matched_pair_protocol_eligible'] for r in group))
  aggregate[arm].update(concrete_rejections=sum(r['independent_task_status'] in ['FAIL','NO_SUBMISSION'] for r in group),inconclusive=sum(r['independent_task_status']=='INCONCLUSIVE' for r in group))
 out=a.root/'report';out.mkdir(exist_ok=True);(out/'results.json').write_text(json.dumps({'schema_version':1,'run_id':a.run_id,'rows':rows,'aggregate':aggregate,'rates':rates,'notes':['independently_passing_assets counts physical task passes; accepted counts passes with an individual run audit, not global experiment eligibility. Only matched_pair_eligible_accepted applies the full paired/global protocol gate.','NO_VERDICT identifies a harness-owned timeout closeout when the frozen evaluator produced no verdict; independent_task_status is INCONCLUSIVE.','Pending and inconclusive cases are not passes. The full-denominator acceptance fraction is a lower bound while pending/inconclusive cases remain.','False pass requires an explicit author acceptance claim and a concrete independent rejection.','Per-accepted effort includes failed attempts. Zero accepted makes the ratio undefined.','Recorded GPU time is assigned allocation time, not measured GPU-busy time.','Dollar costs exclude setup/evaluation and remain unknown without complete usage and explicit rate provenance.','Unmeasured human review effort is unknown, not zero.']},indent=2)+'\n')
 with (out/'results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print(json.dumps(aggregate,indent=2))
if __name__=='__main__':main()
