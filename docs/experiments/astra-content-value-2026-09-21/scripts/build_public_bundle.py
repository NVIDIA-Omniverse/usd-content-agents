#!/usr/bin/env python3
"""Build an explicit evidence allowlist; never publish retention archives."""
import argparse,hashlib,json,re,shutil
from pathlib import Path
STRUCTURAL_DELIVERY_CASES=['07_robot_arm','08_excavator','09_printer','10_complex']

# The standard Horde container account path is part of the retained harness,
# not a machine address or personal workstation identifier.
BLOCKED=re.compile(r'\b[a-z][a-z0-9]*-(?:codex-[0-9]+gpu|astra-[a-z0-9-]+)\b|\b[a-z0-9-]+\.teleport\.sh\b|\b[a-z0-9.-]*horde[a-z0-9.-]*\.nvidia\.com\b|GPU-[a-f0-9-]{20,}|/(?:Users|home)/(?!horde\b)[^/\s\"\']+|(?:sk-|nvapi-|ghp_|github_pat_)[A-Za-z0-9_-]{20,}|bearer\s+[A-Za-z0-9_+./=-]{20,}|eyJ[A-Za-z0-9_-]{15,}\.[A-Za-z0-9_-]{15,}',re.I)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):return json.loads(p.read_text())
def public_value(v):
 if isinstance(v,dict):
  result={k:public_value(x) for k,x in v.items() if k not in ['thread_ids','session_id','source_session','session_file','sessions','session_usage','usage_events','usd_cli_session','missing_final_sessions','recorded_sessions','extracted_tool_sessions','gpu_uuid','assigned_gpu_uuid','hostname','host']}
  for key,label in [('missing_final_sessions','missing_final_session_count'),('recorded_sessions','recorded_session_count'),('extracted_tool_sessions','extracted_tool_session_count')]:
   if isinstance(v.get(key),list):result[label]=len(v[key])
  return result
 if isinstance(v,list):return [public_value(x) for x in v]
 if isinstance(v,str):return BLOCKED.sub('[private runtime identifier]',v)
 return v
def audit_reference_bindings(root,records):
 """Distinguish retained audit digests from projected or omitted public files."""
 published={x['path']:x for x in records};references=[]
 def visit(value,pointer=''):
  if isinstance(value,dict):
   if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
    yield pointer,value
   for key,child in value.items():yield from visit(child,pointer+'/'+str(key))
  elif isinstance(value,list):
   for index,child in enumerate(value):yield from visit(child,pointer+'/'+str(index))
 for arm in ['plain_astra','content_agents']:
  audit_path='audits/pilot-v1/03_hinge/'+arm+'/protocol_audit.json'
  if not (root/audit_path).is_file():continue
  for pointer,evidence in visit(load(root/audit_path)):
   path=evidence['path'];source=root/path
   assert not source.is_symlink() and source.is_file(),path
   assert source.resolve().is_relative_to(root.resolve()),path
   retained=sha(source);assert retained==evidence['sha256'],path+' retained audit digest mismatch'
   item=published.get(path)
   row={'audit_path':audit_path,'reference_pointer':pointer,'evidence_path':path,
        'expected_retained_sha256':evidence['sha256'],'verified_retained_sha256':retained,
        'publication_status':'retained_only'}
   if item:
    assert item['original_retained_sha256']==retained,path+' publication original digest mismatch'
    row.update({'publication_status':'published_projection' if item['projection'] else 'published_exact',
                'published_path':item['path'],'published_sha256':item['sha256'],
                'published_bytes_equal_retained':item['sha256']==retained})
   references.append(row)
 return {'scope':'SHA256 values inside the canonical03 audit evidence references bind retained originals. Published projections have separate hashes below. Retained-only evidence is deliberately absent from this public subset; its original bytes cannot be independently checked from this bundle alone. This index does not claim recovery of lost audit history.',
         'references':references}
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;o=a.output;o.mkdir(parents=True,exist_ok=True);records=[]
 def copy(relative,target=None,projection=False):
  src=r/relative
  if not src.is_file():return
  assert not src.is_symlink()
  dst=o/(target or relative);dst.parent.mkdir(parents=True,exist_ok=True)
  if projection and src.suffix=='.json':dst.write_text(json.dumps(public_value(load(src)),indent=2)+'\n')
  elif projection:dst.write_text(BLOCKED.sub('[private runtime identifier]',src.read_text()))
  else:shutil.copyfile(src,dst)
  records.append({'path':str(dst.relative_to(o)),'sha256':sha(dst),'original_retained_sha256':sha(src),'projection':projection,'bytes':dst.stat().st_size})
 def tree(relative,extensions):
  for f in sorted((r/relative).rglob('*')):
   if f.is_file() and f.suffix in extensions and '__pycache__' not in f.parts:copy(f.relative_to(r))
 for n in ['protocol.json','dataset.json','source_downloads.json','repository_source_hashes.json','rates.template.json']:copy('protocol/'+n)
 for n in ['pair.json','codex.json']:copy('audits/pilot-v1/final_environment/'+n)
 copy('agents/roster.json',projection=True)
 for n in ['08_launch.json','09_launch.json','10_launch.json']:copy('agents/'+n,projection=True)
 tree('protocol/tasks',{'.json'})
 for family in ['general','case08_10']:
  tree('evaluator/'+family+'/frozen',{'.py','.json'})
  for f in sorted((r/'evaluator'/family/'references').glob('*/source_inventory.json')):copy(f.relative_to(r))
 for n in ['REBUILD.md','common.py','geometry.py','freeze_dataset.py','finalize_roles.py','case09_placement.py','case09_placement.json','case09_payload_contract.json','environment_structural.json','environment_solver.json']:
  copy('evaluator/general/'+n)
 for f in (r/'evaluator/case08_10').glob('*.py'):copy(f.relative_to(r),projection=f.name=='release_qualification.py')
 for n in ['qualification.json','qualification_08_excavator.json']:copy('evaluator/case08_10/'+n,projection=True)
 for f in (r/'evaluator/conveyor').glob('*'):
  if f.suffix in ['.py','.json','.md']:copy(f.relative_to(r),projection=f.name=='freeze.json')
 copy('evaluator/conveyor/reference/source_inventory.json')
 frozen=load(r/'evaluator/drawer_freeze.json')
 for n in frozen['files']:copy('evaluator/'+n)
 for n in frozen['evidence']:copy('evaluator/'+n)
 copy('evaluator/drawer_freeze.json','evaluator/drawer_freeze.public.json',projection=True)
 copy('evaluator/drawer_contact_units_errata.md')
 tree('evaluator/errata',{'.py','.json','.md'})
 # Qualified supplementary observations are separate from frozen evaluations.
 structural='performance_research/structural_delivery_v1/'
 for n in ['audit.py','qualify.py','README.md','independent_review.json','hash_bound_review.py','hash_bound_review.json','summary.json']:
  copy(structural+n)
 for case in STRUCTURAL_DELIVERY_CASES:copy(structural+case+'_content_agents.json')
 for qualification in ['qualification_codex','qualification_pair']:
  copy(structural+qualification+'/qualification.json')
  for fixture in ['positive','geometry_only','disabled_body','kinematic_body','no_collider','disabled_collider','wrong_joint_kind','disabled_joint','missing_body_binding','missing_joint_binding','fake_body_path','fake_joint_path','duplicate_body_role','duplicate_joint_role','unresolved_dependency']:
   copy(structural+qualification+'/'+fixture+'.usda')
 # Synthetic06 witness fixtures/logs supplement the code/results above. Never
 # include binaries or core dumps. Runtime-identifying logs are projections.
 for f in sorted((r/'evaluator/errata/engine_auxiliary_body_review').rglob('*')):
  if f.is_file() and f.suffix in ['.usda','.log']:
   copy(f.relative_to(r),projection=f.suffix=='.log')
 for f in (r/'evaluator/general/tests').glob('*qualification*.json'):copy(f.relative_to(r),projection=True)
 for n in ['case04_qualification.json','case09_placement_qualification.json']:copy('evaluator/general/'+n,projection=True)
 for n in ['build_public_bundle.py','fetch_sources.py','align_reference_rebuild_by_hash.py','check_reference_rebuild.py','summarize_results.py','reconcile_usage.py','test_accounting.py','write_results_report.py','plot_drawer.py','run_arm.py','run_queue.py','evaluate_finished.py','render_drawer_independent_replay.py','render_portable_drawer_snapshots.py','verify_public_bundle.py','check_drawer_portability.py','check_drawer_payload_clearance.py']:
  copy('scripts/'+n)
 for n in ['RESULTS.md','METHOD.md','results.json','results.csv','sources.json','compute_accounting.json','global_protocol_audit.json','workflow_geometry_diagnosis.md','workflow_geometry_diagnosis_evidence.json','independent_method_review.md','optional_backend_review_evidence.json','reference_regeneration.md','drawer_replay_reproduction.md','drawer_portability_receipt.json','robot_arm_false_pass_evidence.json','hardware_verification.json']:
  copy('report/'+n,projection=n.endswith('.json') and n not in ['results.json','sources.json'])
 copy('report/PUBLIC_README.md','README.md')
 copy('report/PUBLIC_GITATTRIBUTES','.gitattributes')
 for case in ['03_hinge','05_vise']:
  for n in ['launch.json','result.json','freezer.stdout.log','freezer.stderr.log']:
   copy('repro-check/reference-regeneration-v1/'+case+'/'+n,projection=True)
 for n in ['aligned/alignment_receipt.json','public_compatible/alignment_receipt.json']:
  copy('repro-check/reference-regeneration-v1/03_hinge/'+n,projection=True)
 copy('audits/pilot-v1/global_concurrency/overlap_02_05.json',projection=True)
 copy('audits/pilot-v1/global_concurrency/hinge_five_workers_direct.json')
 copy('evaluation_attempts/pilot-v1/07_robot_arm/independent_early_stop_20260922.json')
 for folder,names in [
  ('evaluation_attempts/pilot-v1/09_printer/plain_astra/bounded_closeout_20260922', ['before_deadline.json','closeout.py','dispatcher_stop.json','final_evaluator_log.txt','final_worker_status.json','last_dispatch_status.json','original_timeout_error.json','pre_timeout_log.txt','timeout_assessment.json','wrapper_provenance.json']),
  ('evaluation_attempts/pilot-v1/10_complex/plain_astra/independent_closeout_20260922', ['receipt.json','inspect.py'])]:
  for name in names:copy(folder+'/'+name)
 copy('audits/pilot-v1/03_hinge/repository_fresh_20260922.json',projection=True)
 tree('report/drawer',{'.json','.gz','.csv','.png','.svg'})
 for case in load(r/'protocol/dataset.json')['assets']:
  case=case['case_id']
  for f in (r/'audits/pilot-v1'/case).glob('*.json'):
   if case=='03_hinge' and f.name=='repository_fresh_20260922.json':continue
   copy(f.relative_to(r),projection=True)
  for arm in ['plain_astra','content_agents']:
   base='runs/pilot-v1/'+case+'/'+arm
   for n in ['execution.json','submission.json','launch.json','model_usage_audit.json']:
    if (r/base/n).exists():
     exact_submission=case in STRUCTURAL_DELIVERY_CASES and arm=='content_agents' and n=='submission.json'
     if exact_submission:
      raw=(r/base/n).read_text();data=load(r/base/n)
      assert not BLOCKED.search(raw) and public_value(data)==data and not re.search(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',raw,re.I),'Supplementary submission is not safe for exact publication'
     copy(base+'/'+n,projection=not exact_submission)
   if case in STRUCTURAL_DELIVERY_CASES and arm=='content_agents':
    manifest=base+'/output_manifest.json';raw=(r/manifest).read_text();data=load(r/manifest)
    assert not BLOCKED.search(raw) and public_value(data)==data and not re.search(r'[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}',raw,re.I),'Supplementary output manifest contains private metadata'
    copy(manifest)
   audits='audits/pilot-v1/'+case+'/'+arm
   for n in ['artifact_integrity.json','protocol_audit.json','protocol_probe.json','usd_readback.json','usage_reconciled.json']:copy(audits+'/'+n,projection=True)
   if case=='03_hinge':
    # Explicit fresh review receipts only; never private request/output files or archives.
    for n in ['protocol_probe.json','retained_metadata.json','process_output_index.json','usd_readback.json','execution_review.json']:
     copy(audits+'/fresh_review_20260922/'+n,projection=True)
    prior={'plain_astra':('pre_fresh_review_20260922_e2fd479a87ec.json','e2fd479a87ecc1abb51e13caed14a5d1c6cb0ee2bf1558c2884970e8673b1ead'),
           'content_agents':('pre_fresh_review_20260922_678123972352.json','6781239723527527ecc62f7458c5880178701d924113d2e0415aeb9bbe678200')}[arm]
    history=audits+'/audit_history/'+prior[0]
    if (r/history).is_file():
     assert sha(r/history)==prior[1],'Unexpected preserved current03 audit receipt'
     data=load(r/history)
     copy(history,projection=public_value(data)!=data or bool(BLOCKED.search((r/history).read_text())))
   for section in ['evaluations','evaluation_adjudications']:
    folder=r/section/'pilot-v1'/case/arm
    for f in sorted(folder.rglob('*.json')):
     if f.name in ['acceptance.json','report.json','conservative_v1.json','provenance.json'] or re.fullmatch(r'seed_\d+\.json',f.name):
      if any(x in f.parts for x in ['_runtime','vendor']):continue
      copy(f.relative_to(r))
   # Only this explicit conservative review receipt, preserved byte-for-byte.
   # Its v1/prior-adjudication hash bindings are required by public aggregation.
   copy('evaluation_adjudications/pilot-v1/'+case+'/'+arm+'/measurement_review/assessment.json')
   if case in STRUCTURAL_DELIVERY_CASES and arm=='content_agents':
    copy('evaluation_adjudications/pilot-v1/'+case+'/'+arm+'/structural_delivery_v1/receipt.json')
 # CC0 source and accepted drawer are the self-contained public geometry example.
 drawer=next(x for x in load(r/'protocol/dataset.json')['assets'] if x['case_id']=='01_drawer')
 for f in drawer['original_source_files']:copy(drawer['source_root']+'/'+f['path'])
 for n in ['final.usd','bindings.json']:copy('runs/pilot-v1/01_drawer/plain_astra/'+n)
 for f in (r/'runs/pilot-v1/01_drawer/plain_astra/assets').rglob('*'):
  if f.is_file() and f.suffix.lower() in ['.jpg','.jpeg','.png']:copy(f.relative_to(r))
 visuals='visuals/pilot-v1/01_drawer/plain_astra/'
 for frame in ['closed_initial','open_loaded','closed_returned']:
  for ext in ['.usda','.png']:copy(visuals+frame+ext)
  for step in ['open','camera','render']:copy(visuals+frame+'_'+step+'.json',projection=True)
 tree(visuals+'assets',{'.jpg','.jpeg','.png'})
 for n in ['replay_render_config.json','five_seed_summary.json','render_receipt.json','render_provenance.json','render_results.json','render_verification.json','renderer_package_version.json','visual_review.json','lane_reservation.json','lane_release.json']:
  copy(visuals+n,projection=True)
 for n in ['diagnostic.json','render_provenance.json','closed_initial_open.json']:
  copy(visuals+'attempt01_open_policy_failure/'+n,projection=True)
 for f in (r/'repro-check/drawer-portability-v1').glob('*.json'):copy(f.relative_to(r),projection=True)
 # Remove only previous generated allowlist files that this build no longer exports.
 old=o/'publication_manifest.json'
 if old.exists():
  retained={x['path'] for x in records}
  for item in load(old)['files']:
   if item['path'] not in retained:
    path=o/item['path']
    if path.is_file():path.unlink()
 failures=[]
 for item in records:
  f=o/item['path']
  if f.suffix.lower() in ['.json','.jsonl','.csv','.md','.py','.usda','.gltf','.svg','.log']:
   if BLOCKED.search(f.read_text(errors='replace')):failures.append(item['path'])
 if failures:raise SystemExit('Publication scan rejected files: '+json.dumps(failures))
 bindings=audit_reference_bindings(r,records)
 (o/'publication_manifest.json').write_text(json.dumps({'scope':'Explicit public subset; per-file original digests identify private retained originals. Projection=true means runtime identifiers/session detail were removed; projected bytes are not the original hashed receipt. Frozen evaluator code and scored acceptance records remain exact. Non-drawer CAD geometry and private tool/rollout data are not included.','files':records,'audit_reference_bindings':bindings},indent=2)+'\n')
 print(json.dumps({'files':len(records),'bytes':sum(x['bytes'] for x in records),'scan':'passed'}))
if __name__=='__main__':main()
