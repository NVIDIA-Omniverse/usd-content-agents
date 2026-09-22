#!/usr/bin/env python3
"""Regression checks for evidence accounting, not authoring implementation tests."""
import contextlib,hashlib,importlib.util,io,json,shutil,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest import mock
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('usage',HERE/'reconcile_usage.py');usage=importlib.util.module_from_spec(spec);spec.loader.exec_module(usage)
spec=importlib.util.spec_from_file_location('aggregation',HERE/'summarize_results.py');aggregation=importlib.util.module_from_spec(spec);spec.loader.exec_module(aggregation)
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d)+'\n')
class Accounting(unittest.TestCase):
 def test_generator_keeps_every_usage_dimension(self):
  value=usage.sum_usage(x for x in ({'input_tokens':20,'cached_input_tokens':10,'output_tokens':3},{'input_tokens':30,'cached_input_tokens':15,'output_tokens':4}))
  self.assertEqual((value['input_tokens'],value['cached_input_tokens'],value['output_tokens']),(50,25,7))
 def test_fork_history_and_response_duplicates_not_billed_twice(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'sample.jsonl';own={'input_tokens':20,'cached_input_tokens':10,'output_tokens':3};total=usage.sum_usage([own])
   records=[{'type':'session_meta','payload':{'id':'child'}},{'type':'token_usage_record','timestamp':'01','payload':{'thread_id':'parent','response_id':'p','usage':{'input_tokens':300}}},{'type':'token_usage_record','timestamp':'02','payload':{'thread_id':'child','response_id':'c','usage':own}},{'type':'token_usage_record','timestamp':'02','payload':{'thread_id':'child','response_id':'c','usage':own}},{'type':'event_msg','timestamp':'03','payload':{'type':'token_count','info':{'total_token_usage':total}}},{'type':'event_msg','timestamp':'04','payload':{'type':'task_complete'}}]
   p.write_text('\n'.join(map(json.dumps,records))+'\n');result=usage.extract(p)
   self.assertEqual(result['usage'],total);self.assertEqual(result['request_count'],1);self.assertTrue(result['ledger_matches_final_counter']);self.assertTrue(result['completed_after_final_usage'])
 def fixture(self,root,status='accepted',valid=True):
  write(root/'protocol/dataset.json',{'assets':[{'case_id':'01_test','title':'Test'}]})
  for arm in ('plain_astra','content_agents'):
   run=root/'runs/pilot-v1/01_test'/arm;audit=root/'audits/pilot-v1/01_test'/arm
   write(run/'execution.json',{'elapsed_seconds':60,'gpu_allocation_seconds':3600,'human_interventions':0})
   write(run/'submission.json',{'claimed_accepted':True,'repairs_used':0})
   write(run/'model_usage_audit.json',{'observed_turn_contexts':[{'model':'gpt-6-astra','reasoning_effort':'ultra'}]})
   write(audit/'protocol_audit.json',{'valid':valid});write(audit/'usage_reconciled.json',{'complete':True,'input_tokens':2000000,'cached_input_tokens':1000000,'output_tokens':100000})
   write(root/'evaluations/pilot-v1/01_test'/arm/'acceptance.json',{'status':status,'accepted':status=='accepted'})
 def aggregate(self,root,rates):
  write(root/'rates.json',rates)
  subprocess.run([sys.executable,str(HERE/'summarize_results.py'),'--root',str(root),'--rates',str(root/'rates.json')],check=True,stdout=subprocess.DEVNULL)
  return json.loads((root/'report/results.json').read_text())
 def test_rates_require_provenance_and_cache_is_not_double_billed(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root);rates={'input_usd_per_million':10,'cached_input_usd_per_million':2,'output_usd_per_million':30,'gpu_usd_per_hour':4}
   self.assertIsNone(self.aggregate(root,rates)['rows'][0]['cost_usd'])
   rates.update(rate_source='synthetic test rates only',effective_date='2026-09-21');result=self.aggregate(root,rates)
   self.assertEqual(result['rows'][0]['cost_usd'],19);self.assertEqual(result['aggregate']['plain_astra']['cost_usd_per_accepted_asset'],19)
   write(root/'audits/pilot-v1/01_test/plain_astra/usage_reconciled.json',{'complete':True,'input_tokens':2,'cached_input_tokens':3,'output_tokens':1})
   self.assertIsNone(self.aggregate(root,rates)['rows'][0]['cost_usd'])
 def test_renamed_session_copies_are_counted_once(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);run=root/'runs/pilot-v1/test/plain_astra';write(run/'execution.json',{'timed_out':False})
   directory=root/'private-runs/pilot-v1/test/plain_astra/sessions';directory.mkdir(parents=True)
   totals=usage.sum_usage([{'input_tokens':20,'cached_input_tokens':10,'output_tokens':3}])
   records=[{'type':'session_meta','payload':{'id':'same-session'}},{'type':'token_usage_record','timestamp':'01','payload':{'thread_id':'same-session','response_id':'request','usage':totals}},{'type':'event_msg','timestamp':'02','payload':{'type':'token_count','info':{'total_token_usage':totals}}},{'type':'event_msg','timestamp':'03','payload':{'type':'task_complete'}}]
   for name in ('original.jsonl','renamed.jsonl'):(directory/name).write_text('\n'.join(map(json.dumps,records))+'\n')
   result=usage.reconcile(root,'pilot-v1','test','plain_astra')
   self.assertEqual(result['input_tokens'],20);self.assertTrue(result['complete'])
 def test_inconclusive_is_not_a_false_pass_and_zero_acceptance_is_undefined(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,'inconclusive');result=self.aggregate(root,{})
   self.assertFalse(result['rows'][0]['demonstrated_false_pass']);self.assertIsNone(result['aggregate']['plain_astra']['cost_usd_per_accepted_asset'])
   self.fixture(root,'not_accepted');self.assertTrue(self.aggregate(root,{})['rows'][0]['demonstrated_false_pass'])
 def test_missing_protocol_audit_does_not_promote_physical_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,valid=None);result=self.aggregate(root,{})
   self.assertEqual(result['rows'][0]['status'],'PENDING_PROTOCOL_AUDIT');self.assertFalse(result['rows'][0]['accepted']);self.assertEqual(result['rows'][0]['independent_task_status'],'PASS')
 def test_known_source_distance_defect_cannot_establish_false_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,'not_accepted')
   for arm in ('plain_astra','content_agents'):
    write(root/'evaluations/pilot-v1/01_test'/arm/'acceptance.json',{'status':'not_accepted','accepted':False,'concrete_failures':['source_surface_retained:test']})
   result=self.aggregate(root,{})
   self.assertEqual(result['rows'][0]['status'],'INCONCLUSIVE');self.assertTrue(result['rows'][0]['distance_correction_pending']);self.assertFalse(result['rows'][0]['demonstrated_false_pass'])
 def test_corrected_acceptance_requires_frozen_hash_and_complete_physics(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,'not_accepted')
   frozen=root/'evaluations/pilot-v1/01_test/plain_astra/acceptance.json'
   receipt=root/'evaluation_adjudications/pilot-v1/01_test/plain_astra/source_distance_v1_1/acceptance.json'
   data={'status':'accepted','accepted':True,'v1_acceptance_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'native_physics':{'all_five_complete':True},'frozen_files_modified':False,'scored_artifacts_modified':False,'tolerances_changed':False}
   write(receipt,data);result=self.aggregate(root,{})['rows'][0]
   self.assertEqual(result['frozen_evaluator_status'],'FAIL');self.assertEqual(result['status'],'PASS');self.assertFalse(result['demonstrated_false_pass'])
   data['native_physics']['all_five_complete']=False;write(receipt,data)
   self.assertEqual(self.aggregate(root,{})['rows'][0]['status'],'INCONCLUSIVE')
   data['native_physics']['all_five_complete']=True;data['v1_acceptance_sha256']='bad';write(receipt,data)
   self.assertEqual(self.aggregate(root,{})['rows'][0]['status'],'INCONCLUSIVE')
 def test_unverified_global_budget_does_not_erase_physics_or_certify_pair(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root)
   write(root/'report/global_protocol_audit.json',{'valid':None})
   result=self.aggregate(root,{})
   self.assertEqual(result['rows'][0]['independent_task_status'],'PASS')
   self.assertFalse(result['rows'][0]['matched_pair_protocol_eligible'])
   self.assertEqual(result['aggregate']['plain_astra']['independently_passing_assets'],1)
 def test_review_can_withdraw_only_hash_bound_fully_covered_rejection(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);self.fixture(root,'not_accepted')
   frozen=root/'evaluations/pilot-v1/01_test/plain_astra/acceptance.json'
   write(frozen,{'status':'not_accepted','accepted':False,'concrete_failures':['witness','hidden_gate']})
   review=root/'evaluation_adjudications/pilot-v1/01_test/plain_astra/measurement_review/assessment.json'
   data={'status':'inconclusive','accepted':False,'v1_acceptance_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'covered_rejection_names':['witness','hidden_gate'],'remaining_supported_rejections':[],'frozen_files_modified':False,'scored_artifacts_modified':False}
   write(review,data);row=self.aggregate(root,{})['rows'][0]
   self.assertEqual(row['frozen_evaluator_status'],'FAIL');self.assertEqual(row['independent_task_status'],'INCONCLUSIVE');self.assertFalse(row['demonstrated_false_pass']);self.assertFalse(row['accepted'])
   data['covered_rejection_names']=['witness'];write(review,data)
   self.assertTrue(self.aggregate(root,{})['rows'][0]['demonstrated_false_pass'])
   data['covered_rejection_names'].append('hidden_gate');data['v1_acceptance_sha256']='wrong';write(review,data)
   self.assertFalse(self.aggregate(root,{})['rows'][0]['measurement_review_applied'])
   data['v1_acceptance_sha256']=hashlib.sha256(frozen.read_bytes()).hexdigest();data['status']='accepted';data['accepted']=True;write(review,data)
   self.assertEqual(self.aggregate(root,{})['rows'][0]['independent_task_status'],'FAIL')
 def structural_fixture(self,root,case='07_robot_arm',status=None,project_launch=True):
  """Approved public metadata only; deliberately do not copy any submitted USD."""
  source=HERE.parent;arm='content_agents';run=Path('runs/pilot-v1')/case/arm
  def copy(relative):
   p=root/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/relative,p)
  receipt=Path('evaluation_adjudications/pilot-v1')/case/arm/'structural_delivery_v1/receipt.json';copy(receipt)
  evidence=json.loads((root/receipt).read_text());folder=aggregation.STRUCTURAL_RECEIPTS[case][1]
  for n in ['audit.py','qualify.py',folder+'/qualification.json']:copy(Path('performance_research/structural_delivery_v1')/n)
  task=Path('protocol/tasks')/(case+'.json');copy(task)
  family='case08_10' if case in ['08_excavator','10_complex'] else 'general';snapshot=Path('evaluator')/family/'frozen'/case
  copy(snapshot/'frozen_manifest.json')
  for n in json.loads((root/snapshot/'frozen_manifest.json').read_text())['code_sha256']:copy(snapshot/n)
  for n in ['submission.json','output_manifest.json','launch.json']:copy(run/n)
  if project_launch:
   launch=json.loads((root/run/'launch.json').read_text())
   for k in ['hostname','gpu_uuid','host','thread_ids']:launch.pop(k,None)
   write(root/run/'launch.json',launch)
  # Works in both the retained tree and a public tree where launch is already projected.
  launch_sha=hashlib.sha256((root/run/'launch.json').read_bytes()).hexdigest()
  if launch_sha!=evidence['inputs']['launch.json']['sha256']:
   write(root/'publication_manifest.json',{'files':[{'path':str(run/'launch.json'),'sha256':launch_sha,'original_retained_sha256':evidence['inputs']['launch.json']['sha256'],'projection':True}]})
  write(root/'protocol/dataset.json',{'assets':[{'case_id':case,'title':'Approved structural observation regression fixture'}]})
  write(root/run/'execution.json',{'elapsed_seconds':60,'gpu_allocation_seconds':60,'human_interventions':0})
  write(root/run/'model_usage_audit.json',{'observed_turn_contexts':[{'model':'gpt-6-astra','reasoning_effort':'ultra'}]})
  write(root/'audits/pilot-v1'/case/arm/'protocol_audit.json',{'valid':True})
  if status:write(root/'evaluations/pilot-v1'/case/arm/'acceptance.json',{'status':status,'accepted':status=='accepted'})
  return {'case':case,'run':root/run,'receipt':root/receipt,'evidence':evidence,'snapshot':root/snapshot,'qualification':root/'performance_research/structural_delivery_v1'/folder/'qualification.json','task':root/task}
 def structural_row(self,root):
  return next(x for x in self.aggregate(root,{})['rows'] if x['arm']=='content_agents')
 def test_structural_delivery_all_four_public_metadata_sets_without_usd(self):
  for case in aggregation.STRUCTURAL_RECEIPTS:
   with self.subTest(case=case),tempfile.TemporaryDirectory() as d:
    root=Path(d);f=self.structural_fixture(root,case)
    self.assertFalse((root/f['evidence']['inputs']['scene']['path']).exists())
    row=self.structural_row(root)
    self.assertTrue(row['structural_delivery_audit_applied']);self.assertEqual(row['independent_task_status'],'FAIL')
    self.assertEqual(row['frozen_evaluator_status'],'PENDING');self.assertFalse(row['accepted'])
    self.assertFalse(row['demonstrated_false_pass'])
    diagnostics=aggregation.structural_delivery_evidence(root,'pilot-v1',case,'content_agents')
    self.assertTrue(diagnostics['retained_only_input_paths'])
    self.assertNotIn('retained_only_input_paths',row['structural_delivery_audit'])
    self.assertNotIn('launch_binding',row['structural_delivery_audit'])
    self.assertEqual(row['structural_delivery_audit']['source_fidelity_status'],'NOT_EVALUATED')
    self.assertEqual(row['structural_delivery_audit']['native_physics_status'],'NOT_EVALUATED')
 def test_structural_delivery_can_only_reject_pending_inconclusive_or_fail(self):
  for status in [None,'inconclusive','not_accepted','accepted']:
   with self.subTest(status=status),tempfile.TemporaryDirectory() as d:
    root=Path(d);self.structural_fixture(root,status=status);row=self.structural_row(root)
    self.assertTrue(row['structural_delivery_audit']['valid'])
    self.assertEqual(row['structural_delivery_audit_applied'],status!='accepted')
    self.assertEqual(row['independent_task_status'],'PASS' if status=='accepted' else 'FAIL')
    self.assertEqual(row['accepted'],status=='accepted')
 def test_structural_delivery_does_not_override_prior_pass_after_bad_adjudication(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=self.structural_fixture(root,status='accepted')
   write(root/'evaluation_adjudications/pilot-v1'/f['case']/'content_agents/source_distance_v1_1/acceptance.json',{'status':'inconclusive','accepted':False,'v1_acceptance_sha256':'wrong'})
   row=self.structural_row(root)
   self.assertEqual(row['frozen_evaluator_status'],'PASS');self.assertFalse(row['structural_delivery_audit_applied'])
   self.assertEqual(row['independent_task_status'],'INCONCLUSIVE')
 def test_structural_delivery_preserves_corrected_pass(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=self.structural_fixture(root,status='not_accepted')
   frozen=root/'evaluations/pilot-v1'/f['case']/'content_agents/acceptance.json'
   write(root/'evaluation_adjudications/pilot-v1'/f['case']/'content_agents/source_distance_v1_1/acceptance.json',{'status':'accepted','accepted':True,'v1_acceptance_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'native_physics':{'all_five_complete':True},'frozen_files_modified':False,'scored_artifacts_modified':False,'tolerances_changed':False})
   row=self.structural_row(root)
   self.assertEqual(row['frozen_evaluator_status'],'FAIL');self.assertEqual(row['adjudicated_evaluator_status'],'PASS')
   self.assertFalse(row['structural_delivery_audit_applied']);self.assertEqual(row['independent_task_status'],'PASS');self.assertTrue(row['accepted'])
 def test_structural_delivery_rejects_stale_or_forged_evidence(self):
  for corruption in ['receipt','receipt_qualification','receipt_task','receipt_manifest','qualification','auditor','task','manifest','scene','bindings','frozen_code','launch_bytes','launch_original_digest']:
   with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as d:
    root=Path(d);f=self.structural_fixture(root,status='inconclusive')
    if corruption=='receipt':
     x=json.loads(f['receipt'].read_text());x['inputs']['scene']['sha256']='0'*64;write(f['receipt'],x)
    elif corruption.startswith('receipt_'):
     x=json.loads(f['receipt'].read_text())
     if corruption=='receipt_qualification':x['qualification_sha256']='0'*64
     else:x['inputs']['public_task' if corruption=='receipt_task' else 'output_manifest.json']['sha256']='0'*64
     write(f['receipt'],x)
    elif corruption=='qualification':
     x=json.loads(f['qualification'].read_text());x['auditor_sha256']='0'*64;write(f['qualification'],x)
    elif corruption=='auditor':(root/'performance_research/structural_delivery_v1/audit.py').write_text('# unqualified replacement\n')
    elif corruption=='task':
     x=json.loads(f['task'].read_text());x['public_acceptance']['required_joint_roles']=[];write(f['task'],x)
    elif corruption=='manifest':
     p=f['run']/'output_manifest.json';x=json.loads(p.read_text());x['files'][0]['sha256']='0'*64;write(p,x)
    elif corruption in ['scene','bindings']:
     p=root/f['evidence']['inputs'][corruption]['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_text('changed retained input\n')
    elif corruption=='frozen_code':(f['snapshot']/'structural.py').write_text('# changed frozen implementation\n')
    elif corruption=='launch_bytes':
     p=f['run']/'launch.json';x=json.loads(p.read_text());x['task_sha256']='0'*64;write(p,x)
    else:
     p=root/'publication_manifest.json';x=json.loads(p.read_text());x['files'][0]['original_retained_sha256']='0'*64;write(p,x)
    row=self.structural_row(root)
    self.assertFalse(row['structural_delivery_audit']['valid']);self.assertFalse(row['structural_delivery_audit_applied'])
    self.assertEqual(row['independent_task_status'],'INCONCLUSIVE');self.assertFalse(row['accepted']);self.assertFalse(row['demonstrated_false_pass'])
 def test_structural_delivery_cannot_promote_acceptance_or_expand_scope(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);f=self.structural_fixture(root,status='inconclusive')
   x=json.loads(f['receipt'].read_text());x['supplementary_status']='accepted';x['accepted']=True;write(f['receipt'],x)
   row=self.structural_row(root);self.assertFalse(row['accepted']);self.assertEqual(row['independent_task_status'],'INCONCLUSIVE')
   self.assertIsNone(aggregation.structural_delivery_evidence(root,'pilot-v1',f['case'],'plain_astra'))
   self.assertIsNone(aggregation.structural_delivery_evidence(root,'another-run',f['case'],'content_agents'))
   self.assertIsNone(aggregation.structural_delivery_evidence(root,'pilot-v1','06_engine','content_agents'))
 def test_whole_results_identical_for_retained_and_actual_public_projection(self):
  spec=importlib.util.spec_from_file_location('builder',HERE/'build_public_bundle.py');builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
  with tempfile.TemporaryDirectory() as d:
   retained=Path(d)/'retained';public=Path(d)/'public';f=self.structural_fixture(retained,case='08_excavator',status='inconclusive')
   # Tiny synthetic bound inputs make this regression runnable without private
   # CAD. Only this test process trusts its freshly hashed receipt; production
   # trust anchors and the qualified auditor/qualification files stay unchanged.
   receipt=f['evidence'];manifest_path=f['run']/'output_manifest.json';manifest=json.loads(manifest_path.read_text())
   scene=retained/receipt['inputs']['scene']['path'];bindings=retained/receipt['inputs']['bindings']['path']
   scene.parent.mkdir(parents=True,exist_ok=True);scene.write_text('#usda 1.0\ndef Xform "SyntheticMetadataFixture" {}\n');bindings.write_text('{}\n')
   for key,path in [('scene',scene),('bindings',bindings)]:
    value=hashlib.sha256(path.read_bytes()).hexdigest();receipt['inputs'][key]['sha256']=value
    for item in manifest['files']:
     if item['path']==str(path.relative_to(f['run'])):item['sha256']=value;item['bytes']=path.stat().st_size
   receipt['composed_layer_hashes']=[dict(receipt['inputs']['scene'])]
   write(manifest_path,manifest);receipt['inputs']['output_manifest.json']['sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
   launch=f['run']/'launch.json';data=json.loads(launch.read_text());data['hostname']='synthetic-retained-host';write(launch,data)
   receipt['inputs']['launch.json']['sha256']=hashlib.sha256(launch.read_bytes()).hexdigest();write(f['receipt'],receipt)
   trust=list(aggregation.STRUCTURAL_RECEIPTS[f['case']]);trust[0]=hashlib.sha256(f['receipt'].read_bytes()).hexdigest()
   shutil.copytree(retained,public)
   launch_relative=launch.relative_to(retained);projected=public/launch_relative
   projected.write_text(json.dumps(builder.public_value(data),indent=2)+'\n')
   write(public/'publication_manifest.json',{'files':[{'path':str(launch_relative),'sha256':hashlib.sha256(projected.read_bytes()).hexdigest(),'original_retained_sha256':receipt['inputs']['launch.json']['sha256'],'projection':True}]})
   for path in [scene,bindings]:(public/path.relative_to(retained)).unlink()
   with mock.patch.dict(aggregation.STRUCTURAL_RECEIPTS,{f['case']:tuple(trust)}):
    local_diagnostics=aggregation.structural_delivery_evidence(retained,'pilot-v1',f['case'],'content_agents')
    public_diagnostics=aggregation.structural_delivery_evidence(public,'pilot-v1',f['case'],'content_agents')
    self.assertTrue(local_diagnostics['valid']);self.assertTrue(public_diagnostics['valid'])
    self.assertEqual(local_diagnostics['launch_binding'],'exact_retained_bytes');self.assertEqual(public_diagnostics['launch_binding'],'verified_public_projection')
    self.assertEqual(local_diagnostics['retained_only_input_paths'],[]);self.assertTrue(public_diagnostics['retained_only_input_paths'])
    for root in [retained,public]:
     with mock.patch.object(sys,'argv',['summarize_results.py','--root',str(root)]),contextlib.redirect_stdout(io.StringIO()):aggregation.main()
   for name in ['results.json','results.csv']:
    self.assertEqual((retained/'report'/name).read_bytes(),(public/'report'/name).read_bytes(),name)
if __name__=='__main__':unittest.main()
