import datetime,hashlib,json,os,signal,time
from pathlib import Path
r=Path('/opt/astra-content-value-20260921');out=r/'evaluation_attempts/pilot-v1/09_printer/plain_astra/bounded_closeout_20260922';evaldir=r/'evaluations/pilot-v1/09_printer/plain_astra'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
def load(p):return json.loads(p.read_text())
def record(p):return {'path':str(p.relative_to(r)),'sha256':sha(p),'bytes':p.stat().st_size}
def write(p,x):assert not p.exists(),str(p);p.write_text(json.dumps(x,indent=2)+'\n')
assert datetime.datetime.now(datetime.timezone.utc)>=datetime.datetime.fromisoformat('2026-09-22T01:46:08+00:00')
err=r/'evidence/pair-09_printer-plain_astra-evaluation-error.json';error=load(err)
assert 'TimeoutExpired' in error['error'] and '5400' in error['error'] and '/frozen/09_printer/evaluate.py' in error['error']
assert not (evaldir/'acceptance.json').exists() and not (evaldir/'report.json').exists()
assert not Path('/proc/164822').exists(),'Original timed evaluator still exists'
assert not (evaldir/'runtime_config.json').exists(),'Native config unexpectedly produced; review before closeout'
corrected=r/'evaluation_adjudications/pilot-v1/10_complex/plain_astra/source_distance_v1_1/acceptance.json'
assert sha(corrected)=='382566007ee71f8e82520fea1feaebc464a1def89d185f53282ad0849ea04366'
# Only the previously coordinated, now unnecessary adjudication dispatcher.
pid=124238;proc=Path('/proc')/str(pid);termination={'pid':pid,'action':'already_exited','reason':'Direct10 corrected receipt is complete; direct09 exhausted its original bound without a verdict; geometry-only CA09/10 already have qualified mandatory-physics absence receipts. No corrected/source or native completion is inferred.'}
if proc.exists():
 cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip();cwd=os.readlink(proc/'cwd')
 assert str(r/'evaluator/errata/source_distance_v1_1/dispatcher.py') in cmd and '--label pair_r2_20260921' in cmd
 children=[]
 for path in Path('/proc').glob('[0-9]*'):
  try:
   stat=(path/'stat').read_text();ppid=int(stat[stat.rfind(')')+2:].split()[1])
   if ppid==pid:children.append(int(path.name))
  except (OSError,ValueError):pass
 assert not children,('Unexpected active children',children)
 os.kill(pid,signal.SIGTERM)
 for _ in range(30):
  if not proc.exists():break
  time.sleep(.1)
 assert not proc.exists(),'Dispatcher has not exited'
 termination.update(action='SIGTERM',verified_cmdline=cmd,verified_cwd=cwd,children_before=[],confirmed_exited=True)
write(out/'dispatcher_stop.json',dict(termination,observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
for original,name in [(err,'original_timeout_error.json'),(r/'evidence/pair-evaluation-worker.json','final_worker_status.json'),(r/'evaluations/pilot-v1/09_printer/plain_astra-independent.log','final_evaluator_log.txt'),(r/'evaluation_adjudications/pilot-v1/_dispatch/pair_r2_20260921.json','last_dispatch_status.json')]:
 p=out/name;assert not p.exists();p.write_bytes(original.read_bytes())
before=load(out/'before_deadline.json');checks=[]
for entry in before['input_and_frozen_files']:
 if entry['path']=='evidence/pair-evaluation-worker.json':continue
 p=r/entry['path'];checks.append({'path':entry['path'],'before_sha256':entry['sha256'],'after_sha256':sha(p),'unchanged':sha(p)==entry['sha256']})
assert all(x['unchanged'] for x in checks)
script=out/'closeout.py';assert not script.exists();script.write_bytes(Path(__file__).read_bytes())
assessment={'measurement_owner':'harness_bounded_attempt_closeout','acceptance_wrapper':True,'source_checks_complete':False,'native_checks_complete':False,'closeout_script':record(script),'version':'bounded_evaluator_timeout_closeout_v1','observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_id':'09_printer','arm':'plain_astra','status':'inconclusive','accepted':False,'concrete_failures':[],'inconclusive_checks':['independent_evaluator_time_limit_exceeded_before_verdict'],'origin':'Independent bounded-attempt closeout; not a verdict emitted by the frozen evaluator','existing_time_limit_seconds':5400,'original_attempt_started_utc':'2026-09-22T00:16:07Z','original_attempt_deadline_utc':'2026-09-22T01:46:07Z','frozen_evaluator_verdict_produced':False,'native_physics_status':'NOT_EVALUATED','source_fidelity_status':'INCOMPLETE_NOT_ASSESSED','false_pass_supported':False,'partial_output_files_before_assessment':[record(p) for p in sorted(evaldir.rglob('*')) if p.is_file()],'before_deadline_receipt':record(out/'before_deadline.json'),'raw_timeout_error':record(out/'original_timeout_error.json'),'raw_evaluator_log':record(out/'final_evaluator_log.txt'),'final_worker_status':record(out/'final_worker_status.json'),'dispatcher_stop':record(out/'dispatcher_stop.json'),'input_and_frozen_hash_rechecks':checks,'scored_author_artifacts_modified':False,'frozen_implementation_modified':False,'attempt_extended':False,'expensive_scoring_restarted':False,'scope':'Timeout does not establish source mismatch, physical task failure or an author false pass. No native trajectories were produced. The separate CA09 mandatory-physics non-delivery receipt remains independent.'}
write(out/'timeout_assessment.json',assessment)
wrapper=dict(assessment);wrapper['closeout_assessment']=record(out/'timeout_assessment.json');write(evaldir/'acceptance.json',wrapper)
write(out/'wrapper_provenance.json',{'measurement_owner':'harness_bounded_attempt_closeout','frozen_evaluator_emitted_no_verdict':True,'assessment':record(out/'timeout_assessment.json'),'acceptance_wrapper':record(evaldir/'acceptance.json'),'closeout_script':record(script),'original_timeout_error':record(out/'original_timeout_error.json'),'frozen_code_unchanged':True,'author_artifacts_unchanged':True})
print(json.dumps({'assessment':record(out/'timeout_assessment.json'),'acceptance_wrapper':record(evaldir/'acceptance.json'),'dispatcher_stop':record(out/'dispatcher_stop.json')}))
