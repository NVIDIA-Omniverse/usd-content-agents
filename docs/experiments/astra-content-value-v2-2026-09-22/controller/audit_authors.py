"""Prepare private, immutable author review evidence; never infer treatment compliance.

UI association is mechanical. Tool/script review remains an explicit independent
adjudication, separate from task acceptance and terminal billing completeness.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import re
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
from common import read, require, sha256, object_hash, write_new
from ledger import Ledger
from ui_audit import summarize,reconcile
from retain_run import verify_archive

SUSPECT=re.compile(r'content[-_]workflow|content[-_]agent|/workflows|agentic/packages|(?:physics|joint|geometry|validation)[-_]agent',re.I)


def review(cfg_path,jobs_path,destination,drain_receipt=None):
    cfg=read(cfg_path); jobs=read(jobs_path)['jobs']; out=Path(destination)
    require(not out.exists(),'Fresh private review directory required')
    require(len(jobs)==20 and len({j['run_id'] for j in jobs})==20,'Twenty unique runs required')
    db=Ledger(cfg['database'])
    states={row['id']:row['status']for row in db.db.execute('SELECT id,status FROM runs')}
    require(set(states)=={j['run_id']for j in jobs} and all(x=='reaped'for x in states.values()),'Every author must be reaped before final audit')
    drain=read(drain_receipt) if drain_receipt else None
    if db.db.execute("SELECT 1 FROM requests WHERE status='pending' LIMIT 1").fetchone():
        require(drain and drain.get('gateway_process_exited') is True and drain.get('all_author_runs_reaped') is True,
                'Pending usage requires a trusted bounded-drain and gateway-exit receipt')
        require(drain.get('drain_policy')=='All request deadlines expired plus 65 seconds before gateway shutdown',
                'Original upstream budgets must be allowed to finish before gateway exit')
    out.mkdir(mode=0o700,parents=True); results=[]
    try:
        for job in jobs:
            run=job['run_id']; dest=out/run;dest.mkdir(mode=0o700)
            retained=Path(cfg['retained_archives'])/run; proof=read(retained/'verification.json')
            archive=retained/'complete.tgz';verify_archive(archive,run,proof)
            captures=dest/'session_captures';captures.mkdir(mode=0o700)
            metadata={};tools=[];suspects=[];scripts=[]
            with tarfile.open(archive) as tar:
                for member in tar:
                    if not member.isfile():continue
                    rel=Path(member.name).relative_to(run)
                    if len(rel.parts)==3 and rel.parts[:2]==('private','session_captures'):
                        data=tar.extractfile(member).read();(captures/rel.name).write_bytes(data)
                        own=None
                        for n,line in enumerate(data.decode('utf-8',errors='replace').splitlines(),1):
                            try:r=json.loads(line)
                            except ValueError:continue
                            payload=r.get('payload',{})
                            if not isinstance(payload,dict):continue
                            if r.get('type')=='session_meta' and own is None:own=payload.get('id')
                            if r.get('type')=='response_item' and payload.get('type') in ('function_call','custom_tool_call'):
                                item={'capture':rel.name,'line':n,'own_thread':own,'call':payload}
                                tools.append(item)
                                if SUSPECT.search(json.dumps(payload)):suspects.append({'capture':rel.name,'line':n,'name':payload.get('name')})
                    elif str(rel) in ('private/launch.json','private/reap.json','private/author_delivery.json','private/ui_effort_audit.json'):
                        data=tar.extractfile(member).read();metadata[rel.name]=json.loads(data);(dest/rel.name).write_bytes(data)
                    elif rel.parts[0]=='workspace' and rel.suffix in ('.py','.sh','.js','.mjs'):
                        data=tar.extractfile(member).read()
                        scripts.append({'path':str(rel),'bytes':len(data),'sha256':__import__('hashlib').sha256(data).hexdigest(),'workflow_string_present':bool(SUSPECT.search(data.decode('utf-8',errors='replace')))})
            require(set(metadata)=={'launch.json','reap.json','author_delivery.json','ui_effort_audit.json'},'Missing authoritative author metadata')
            launch=metadata['launch.json'];reap=metadata['reap.json']
            for key in ('run_id','case_id','arm','lane_id','protocol_sha256','task_sha256','input_sha256','source_sha256'):
                require(launch.get(key)==job[key],'Author launch differs from assigned job: '+key)
            require(reap['run_id']==run and metadata['author_delivery.json']['run_id']==run,'Author receipt run differs')
            require(reap['lease']==launch['lease'] and reap['cgroup_populated']==0 and reap['namespace_init_exited'] is True,'Author not bound to a quiescent lease')
            capture_errors=metadata['ui_effort_audit.json'].get('capture_errors',[])
            audit=summarize(captures,capture_errors);usage=db.usage(run);association=reconcile(audit,usage)
            if drain:require(drain['usage_sha256'][run]==object_hash(usage),'Usage changed after trusted drain receipt')
            write_new(dest/'ui_metadata.json',audit);write_new(dest/'ui_association.json',association);write_new(dest/'final_usage.json',usage)
            with (dest/'tool_calls.private.jsonl').open('x') as f:
                for item in tools:f.write(json.dumps(item)+'\n')
            write_new(dest/'workspace_script_inventory.json',{'scripts':scripts})
            result={'run_id':run,'case_id':job['case_id'],'arm':job['arm'],'retention_sha256':proof['archive_sha256'],
                    'ui_complete':association['complete'],'terminal_billing_complete':usage['complete'],
                    'terminal_response_id_associations':association['terminal_response_id_associations'],
                    'admission_identity_associations':association['admission_identity_associations'],
                    'source_unchanged':metadata['reap.json']['source_unchanged'],
                    'tool_call_count':len(tools),'workflow_review_candidates':suspects,
                    'author_claim':metadata['author_delivery.json'].get('claimed_accepted'),
                    'treatment_audit':'REQUIRED','protocol_eligible':None,
                    'scope':'Regex findings are review pointers, never an automatic violation or proof of non-use. Review tool calls and authored scripts from immutable archive. Copied ancestor tool calls can repeat in child captures.'}
            write_new(dest/'review_index.json',result);results.append(result)
    finally:db.db.close()
    write_new(out/'index.json',{'runs':results,'all_treatment_audits_pending':True})
    return {'runs':len(results),'ui_complete':sum(x['ui_complete'] for x in results),'billing_complete':sum(x['terminal_billing_complete'] for x in results),'protocol_adjudication':'PENDING'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--jobs',required=True);p.add_argument('--output',required=True);p.add_argument('--drain-receipt');a=p.parse_args()
    print(json.dumps(review(a.config,a.jobs,a.output,a.drain_receipt)))
