#!/usr/bin/env python3
"""Extract only usage/model metadata from private sessions; never export model reasoning."""
import argparse,hashlib,json,re,time
from pathlib import Path
KEYS=('input_tokens','cached_input_tokens','cache_write_input_tokens','output_tokens','reasoning_output_tokens','total_tokens')
def read(p,default=None):
    try:return json.loads(p.read_text())
    except (OSError,ValueError):return default
def sum_usage(items):
    items=list(items)
    return {k:sum(x.get(k,0) for x in items) for k in KEYS}
def lines(path):
    with path.open(errors='replace') as stream:
        yield from stream
def extract(path):
    metadata=None;records={};contexts=[];last_counter=None;last_token_at='';completion_at='';requests={};commands=[]
    for line in lines(path):
        try:d=json.loads(line)
        except ValueError:continue
        p=d.get('payload',{});kind=d.get('type');at=d.get('timestamp','')
        if kind=='session_meta' and metadata is None:metadata=p
        own_id=(metadata or {}).get('id')
        if kind=='turn_context':contexts.append({'model':p.get('model'),'reasoning_effort':p.get('effort',p.get('model_reasoning_effort'))})
        if kind=='token_usage_record' and p.get('thread_id')==own_id:
            identity=p.get('response_id')
            if identity:
                records[identity]=p.get('usage',{});last_token_at=max(last_token_at,at)
        if kind=='event_msg' and p.get('type')=='token_count' and p.get('info'):
            last_counter=p['info'].get('total_token_usage')
        if kind=='event_msg' and p.get('type')=='task_complete':completion_at=max(completion_at,at)
    if not metadata:return None
    total=sum_usage(records.values());consistent=bool(records) and last_counter is not None and all(total[k]==last_counter.get(k,0) for k in KEYS)
    return {'session_file':path.name,'session_id':metadata.get('id'),'request_count':len(records),'request_identity_digest':hashlib.sha256('\n'.join(sorted(records)).encode()).hexdigest(),'usage':total if records else last_counter,'cumulative_counter':last_counter,'ledger_matches_final_counter':consistent,'completed_after_final_usage':bool(last_token_at and completion_at>=last_token_at),'last_usage_utc':last_token_at or None,'completion_utc':completion_at or None,'observed_contexts':list({json.dumps(x,sort_keys=True):x for x in contexts}.values())}
def reconcile(root,run_id,case,arm):
    run=root/'runs'/run_id/case/arm;execution=read(run/'execution.json')
    if execution is None:return None
    private=root/'private-runs'/run_id/case/arm;audit=read(run/'model_usage_audit.json',{})
    files=list(private.glob('sessions/**/*.jsonl'))+list(run.rglob('rollout-*.jsonl'))
    sessions={};seen_files=set()
    for path in files:
        if path.is_symlink():continue
        item=extract(path)
        if item:
            seen_files.add(item['session_file'])
            identity=item.get('session_id') or item['session_file']
            old=sessions.get(identity)
            if old is None or item['request_count']>=old['request_count']:sessions[identity]=item
    missing=[]
    for item in audit.get('session_usage',[]):
        name=item['session_file']
        if name not in seen_files:
            missing.append(name);sessions[name]={'session_file':name,'usage':item['total_token_usage'],'complete':False,'source':'last periodic counter; transient session no longer available'}
    result=sum_usage(x['usage'] for x in sessions.values() if x.get('usage'))
    contexts=list({json.dumps(x,sort_keys=True):x for s in sessions.values() for x in s.get('observed_contexts',[])}.values())
    complete=bool(sessions) and not missing and not execution.get('timed_out') and all(s.get('ledger_matches_final_counter') and s.get('completed_after_final_usage') for s in sessions.values())
    result.update({'schema_version':1,'case_id':case,'arm':arm,'complete':complete,'sessions':list(sessions.values()),'missing_final_sessions':missing,'observed_contexts':contexts,'observed_model_effort_match':bool(contexts) and all(x['model']=='gpt-6-astra' and x['reasoning_effort']=='ultra' for x in contexts),'policy':'Sum per-response usage for each distinct session once, checked against its final cumulative counter. Parent turn usage excludes child sessions. Missing or interrupted sessions make completeness false. This is client telemetry, not an invoice.','generated_unix':time.time()})
    out=root/'audits'/run_id/case/arm;out.mkdir(parents=True,exist_ok=True)
    (out/'usage_reconciled.json').write_text(json.dumps(result,indent=2)+'\n');return {k:result[k] for k in ('case_id','arm','complete','input_tokens','cached_input_tokens','output_tokens','observed_model_effort_match')}
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run-id',default='pilot-v1');p.add_argument('--case');a=p.parse_args()
    for run in sorted((a.root/'runs'/a.run_id).glob('*/*/execution.json')):
        case=run.parent.parent.name;arm=run.parent.name
        if a.case and case!=a.case:continue
        print(json.dumps(reconcile(a.root,a.run_id,case,arm)))
if __name__=='__main__':main()
