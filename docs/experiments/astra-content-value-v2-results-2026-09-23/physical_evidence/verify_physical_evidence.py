"""Create-only exact physical evidence companion; no authored code is executed.

Build from trusted extracted metadata and retention proofs, never by searching an
author workspace. Only explicitly named evaluator reports and numeric traces are
selected. A partial development build is always labeled incomplete.
"""
import argparse
import hashlib
import gzip
import io
import importlib.util
import json
import math
from pathlib import Path
import re
import tarfile

CASES=('01_drawer','02_conveyor','03_hinge','04_gripper','05_vise','06_engine','07_robot_arm','08_excavator','09_printer','10_complex')
ARMS=('plain_astra','content_agents')
RUNS={f'v2_{c}_{a}' for c in CASES for a in ARMS}
SEEDS=(11,23,47,83,131)
SEED='(?:11|23|47|83|131)'
SCANNER_SHA='100339638100c2264c192710164d062f6adfdb99653b7dd48d7663b172ba6581'
SEAL_SHA='ec9ceb5efc6e231df1bda550aa0386e0923a6f534a9a8634b96726535fe30d48'
FREEZE_SHA='5b64456b3aaa06754d38bbea6345c2347a2039af3bdd8f71db9b15c1496af097'
MAX_FILE=64*1024*1024
MAX_TOTAL=512*1024*1024
SELECTED=re.compile(r'private/evaluation\.json|home/evidence/(?:parser_preflight|submission_rejection)\.json|home/evidence/result/(?:acceptance|report|structural_report)\.json|home/evidence/result/initial_collision_preflight/report\.json|home/evidence/result/seed_'+SEED+r'(?:\.json|/(?:report|trial_report)\.json|/trace\.jsonl)\Z')
EXCLUDED_JSON=re.compile(r'home/evidence/result/(?:runtime_config\.json|source_progress\.jsonl|initial_collision_preflight/request\.json|seed_'+SEED+r'(?:\.log\.process\.json|/(?:request\.json|solver\.log\.process\.json)))\Z')
EXTRA_PRIVATE=re.compile(r'(?:\b[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\b|\b(?:resp|req)_[A-Za-z0-9_-]+|/opt/|/Users/)',re.I)
DRAWER_FIELDS=set('step time_s phase target_q_m applied_force_world_n q_m drawer_pose drawer_velocity payload_pose payload_velocity payload_drawer_contact_impulse_ns payload_drawer_contact_force_n payload_center_in_initial_drawer_frame_m contacts'.split())
CONVEYOR_FIELDS=set('step time_s poses velocities points_world_m rail_angle_rad carrier_angle_rad payload_angle_rad torque_nm contact_forces_n max_penetration_m'.split())
PHASES={'settle','open','hold_open','close','hold_closed'}

def require(value,message):
    if not value: raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()
def encode(value):return (json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def parse(raw):
    def pairs(rows):
        out={}
        for k,v in rows:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    return json.loads(raw,object_pairs_hook=pairs)
def safe_rel(value):
    p=Path(value)
    require(isinstance(value,str) and not p.is_absolute() and '..' not in p.parts and p.as_posix()==value,'Unsafe member path')
    return p
def regular(path,root=None):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),'Missing or linked input')
    if root is not None:require(p.resolve().is_relative_to(Path(root).resolve()),'Escaping input')
    require(p.stat().st_size<=MAX_FILE,'Oversized evidence member')
    return p.read_bytes()
def load_scanner(path):
    require(file_sha(path)==SCANNER_SHA,'Frozen privacy scanner differs')
    spec=importlib.util.spec_from_file_location('frozen_privacy_scan',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def numeric_tree(value):
    if isinstance(value,dict):
        require(all(isinstance(k,str) for k in value),'Non-string numeric field')
        for v in value.values():numeric_tree(v)
    elif isinstance(value,list):
        for v in value:numeric_tree(v)
    else:require(value is None or type(value) in (bool,int,float),'Non-numeric trace payload')

def trace_check(raw):
    rows=0;nonfinite=0;previous=None;fields=None
    for line in raw.splitlines():
        require(line,'Blank trace row');record=parse(line)
        require(isinstance(record,dict),'Trace row must be an object')
        current=set(record)
        require(current in (DRAWER_FIELDS,CONVEYOR_FIELDS),'Unexpected trace schema')
        require(fields is None or current==fields,'Trace schema changed midstream');fields=current
        require(type(record['step']) is int and record['step']>=0,'Invalid trace step')
        require(previous is None or record['step']==previous+1,'Noncontiguous trace steps')
        require(previous is not None or record['step']==0,'Trace does not start at zero')
        previous=record['step'];require(type(record['time_s']) in (int,float),'Invalid trace time')
        if 'phase' in record:require(record['phase'] in PHASES,'Unrecognized trace phase')
        numeric_tree({k:v for k,v in record.items() if k!='phase'})
        # Report nonfinite data without rewriting or suppressing failed traces.
        def count(v):
            if isinstance(v,dict):return sum(count(x) for x in v.values())
            if isinstance(v,list):return sum(count(x) for x in v)
            return int(type(v) is float and not math.isfinite(v))
        nonfinite+=count(record);rows+=1
    return {'rows':rows,'nonfinite_numeric_values':nonfinite,'schema':'drawer' if fields==DRAWER_FIELDS else 'conveyor' if fields==CONVEYOR_FIELDS else 'empty',
            'scope':'Original numeric trace bytes; sample count is not a new physical verdict.'}

def compress_trace(raw):
    stream=io.BytesIO()
    with gzip.GzipFile(filename='',fileobj=stream,mode='wb',compresslevel=9,mtime=0) as f:f.write(raw)
    return stream.getvalue()

def decompress_trace(raw,length,digest):
    require(type(length) is int and 0<=length<=MAX_FILE,'Invalid decompression bound')
    require(re.fullmatch('[0-9a-f]{64}',digest),'Invalid decompressed digest')
    require(raw[:4]==b'\x1f\x8b\x08\x00' and raw[4:8]==b'\x00'*4,'Unexpected gzip header metadata')
    with gzip.GzipFile(fileobj=io.BytesIO(raw),mode='rb') as f:
        decoded=f.read(length+1)
        require(len(decoded)==length,'Decompressed length differs or exceeds bound')
        require(f.read(1)==b'','Decompression exceeds bound')
    require(sha(decoded)==digest,'Decompressed retained-member digest differs')
    return decoded

def scan(raw,rel,scanner):
    if not rel.endswith('.py'):
        require(not EXTRA_PRIVATE.search(raw.decode('utf-8')),'Private identity or runtime locator')
    if rel.endswith('.jsonl'):
        for line in raw.splitlines():scanner.scan(line,structured=True)
        return trace_check(raw)
    scanner.scan(raw,structured=rel.endswith('.json'))
    if rel.endswith('.json'):parse(raw)
    return None

def public_path(author,original):
    return 'runs/'+author+'/'+('evaluation_summary.json' if original=='private/evaluation.json' else 'native/'+original.removeprefix('home/evidence/'))

def selected_inventory(manifest):
    out={};seen=set();unknown=[]
    for row in manifest['files']:
        rel=row['path'];safe_rel(rel);require(rel not in seen,'Duplicate retained member');seen.add(rel)
        if SELECTED.fullmatch(rel):
            require(row['kind']=='file','Selected evidence is not a regular file')
            require(type(row['bytes']) is int and 0<=row['bytes']<=MAX_FILE,'Invalid selected size')
            require(re.fullmatch('[0-9a-f]{64}',row['sha256']),'Invalid retained digest');out[rel]=row
        elif rel.startswith('home/evidence/') and rel.endswith(('.json','.jsonl')) and not EXCLUDED_JSON.fullmatch(rel):
            unknown.append(rel)
    require(not unknown,'Unclassified evaluator JSON members require review: '+repr(unknown))
    require('private/evaluation.json' in out,'Missing trusted evaluation summary')
    return out

def extract_traces(archive,eval_run,members):
    wanted={eval_run+'/'+p:row for p,row in members.items() if p.endswith('/trace.jsonl')}
    found={}
    if not wanted:return found
    with tarfile.open(archive,'r|gz') as t:
        for m in t:
            if m.name not in wanted:continue
            row=wanted[m.name]
            require(m.name not in found,'Duplicate selected tar member')
            require(m.isfile() and not m.issym() and not m.islnk() and m.size==row['bytes'],'Unsafe selected tar member')
            raw=t.extractfile(m).read(MAX_FILE+1)
            require(len(raw)==m.size and sha(raw)==row['sha256'],'Retained trace bytes differ')
            found[m.name]=raw
    require(set(found)==set(wanted),'Retained trace missing')
    return {k[len(eval_run)+1:]:v for k,v in found.items()}

def wrapper_status(native,parser):
    value=native.get('status')
    normalized={'accepted':'PASS','not_accepted':'FAIL','inconclusive':'INCONCLUSIVE'}.get(value,value)
    if normalized not in ('PASS','FAIL','INCONCLUSIVE'):normalized='INCONCLUSIVE'
    return 'FAIL' if parser.get('status')=='FAIL' else normalized

def build(root,index_path,out,allow_partial=False):
    root=Path(root).resolve();index_path=Path(index_path).resolve();out=Path(out)
    require(not out.exists(),'Create-only destination exists')
    batch=root/'controller/private/scored_batch01'
    index_raw=regular(index_path,batch);index=parse(index_raw)
    identities=[r['author_run_id'] for r in index['runs']]
    require(len(identities)==len(set(identities)) and set(identities)<=RUNS,'Unknown or duplicate author attempt')
    require(bool(identities) and (allow_partial or set(identities)==RUNS),'All twenty author attempts required')
    seal_raw=regular(batch/'blinded_adjudication_seal.json',batch)
    require(sha(seal_raw)==SEAL_SHA,'Blinded adjudication seal differs')
    seal=parse(seal_raw);require(seal['all_twenty_audits_validated'] is True and seal['independent_outcomes_not_opened_by_root_or_supplied_to_reviewers_before_seal'] is True,'Blind seal incomplete')
    require({r['run_id'] for r in seal['runs']}==RUNS,'Seal missing author identities')
    require(file_sha(root/'protocol/freeze.json')==FREEZE_SHA,'Original freeze differs')
    scanner=load_scanner(root/'controller/publication.py')
    seen=set();staged={};records=[];runs=[];private_inputs={str(index_path):sha(index_raw)}
    def take(path,scope=None):
        raw=regular(path,scope);private_inputs[str(Path(path).resolve())]=sha(raw);return raw
    for entry in index['runs']:
        author=entry['author_run_id'];require(author in RUNS and author not in seen,'Unknown or duplicate author attempt');seen.add(author)
        eval_run='eval_measured-v2_'+author;references=entry['references']
        def metadata(rel):
            ref=references[rel];p=Path(ref['path'])
            require(p.resolve()==(index_path.parent/author/rel).resolve(),'Unexpected extracted reference location')
            raw=take(p,index_path.parent/author);require(sha(raw)==ref['sha256'],'Extracted metadata changed');return raw
        manifest_raw=metadata('private/retention_manifest.json');manifest=parse(manifest_raw)
        require(manifest['run_id']==eval_run and manifest['schema_version']=='retention.v2','Retention identity differs')
        proof_raw=take(batch/'evaluations01'/author/'retained/verification.json',batch);proof=parse(proof_raw)
        require(proof['run_id']==eval_run and proof['local_member_verification'] is True and proof['export_complete'] is True,'Retention proof incomplete')
        require(proof['manifest_sha256']==sha(manifest_raw),'Retention manifest differs')
        selected=selected_inventory(manifest);all_members={r['path']:r for r in manifest['files']}
        def bound_metadata(rel):
            raw=metadata(rel);row=all_members[rel]
            require(row['kind']=='file' and row['bytes']==len(raw) and row['sha256']==sha(raw),'Metadata not bound to retained member');return raw
        terminal_raw=bound_metadata('private/batch_evaluation.json');terminal=parse(terminal_raw)
        request_raw=bound_metadata('private/batch_request.json');request=parse(request_raw)
        summary_raw=bound_metadata('private/evaluation.json');summary=parse(summary_raw)
        require(terminal['author_run_id']==request['author_run_id']==author and terminal['run_id']==request['run_id']==summary['run_id']==eval_run,'Author/evaluator identity differs')
        require(terminal['terminal'] is True and terminal['cgroup_populated']==summary['cgroup_populated']==0,'Evaluator not reaped')
        require(terminal['request_sha256']==sha(request_raw) and terminal['evaluation_sha256']==sha(summary_raw),'Terminal input binding differs')
        require(terminal['status']==summary['status']==entry['status'] and summary['status'] in ('PASS','FAIL','INCONCLUSIVE'),'Summary status differs')
        require(request['freeze_sha256']==FREEZE_SHA,'Evaluation used different freeze')
        author_proof_raw=take(batch/'archives'/author/'verification.json',batch);author_proof=parse(author_proof_raw)
        require(author_proof['run_id']==author and author_proof['local_member_verification'] is True and author_proof['archive_sha256']==request['author_archive_sha256'],'Original author archive differs')
        archive=Path(proof['local_archive'])
        require(archive.resolve()==(batch/'evaluations01'/author/'retained/complete.tgz').resolve() and archive.is_file() and not archive.is_symlink(),'Unexpected archive location')
        require(archive.stat().st_size==proof['archive_bytes'],'Retained archive size changed')
        trace_members={p:v for p,v in selected.items() if p.endswith('/trace.jsonl')}
        traces={}
        if trace_members:
            require(file_sha(archive)==proof['archive_sha256'],'Trace source archive changed')
            traces=extract_traces(archive,eval_run,trace_members)
        files=[]
        for rel,row in sorted(selected.items()):
            raw=traces[rel] if rel in traces else bound_metadata(rel)
            public=public_path(author,rel)
            detail=scan(raw,public,scanner)
            supplied=compress_trace(raw) if detail is not None else raw
            if detail is not None:public+='.gz'
            require(sum(len(v) for v in staged.values())+len(supplied)<=MAX_TOTAL,'Evidence bundle exceeds bound')
            staged[public]=supplied;files.append(public)
            role='trusted_evaluator_summary' if rel=='private/evaluation.json' else 'numeric_trace' if rel.endswith('.jsonl') else 'native_acceptance_report' if rel in ('home/evidence/result/acceptance.json','home/evidence/result/report.json') else 'evaluator_detail'
            record={'path':public,'bytes':len(supplied),'sha256':sha(supplied),
                    'kind':'lossless_gzip_derivative' if detail is not None else 'exact_retained_bytes','role':role,
                    'author_run_id':author,'original_member_path':rel,'original_member_sha256':row['sha256'],
                    'original_member_bytes':len(raw),'original_bytes_provided':detail is None,
                    'retention_manifest_sha256':sha(manifest_raw),'evaluation_archive_sha256':proof['archive_sha256']}
            if detail is not None:
                record['numeric_trace_summary']=detail
                record['compression']={'format':'gzip','level':9,'mtime':0,'filename_metadata':False,
                    'decompressed_sha256':sha(raw),'decompressed_bytes':len(raw),
                    'arithmetic_or_reserialization':False} 
            records.append(record)
        report=summary.get('report');report_sha=summary.get('report_sha256')
        if report_sha is not None:
            require(report in selected and selected[report]['sha256']==report_sha,'Authoritative report binding missing')
        elif 'report' not in summary:
            require('home/evidence/submission_rejection.json' in selected and summary['status']=='FAIL','Missing early submission rejection')
        native_reports=[p for p in selected if p in ('home/evidence/result/acceptance.json','home/evidence/result/report.json')]
        if report_sha is not None:
            native=parse(staged[public_path(author,report)])
            parser=parse(staged[public_path(author,'home/evidence/parser_preflight.json')]) if 'home/evidence/parser_preflight.json' in selected else {}
            require(summary['status']==wrapper_status(native,parser),'Native/summary verdict differs without parser override')
        runs.append({'author_run_id':author,'case_id':request['case_id'],'arm':request['arm'],
          'summary_status':summary['status'],'summary_path':'runs/'+author+'/evaluation_summary.json',
          'authoritative_report_original_path':report,'authoritative_report_sha256':report_sha,
          'native_report_present':bool(native_reports),'submission_rejection_present':'home/evidence/submission_rejection.json' in selected,
          'selected_members':files,'evaluation_archive_sha256':proof['archive_sha256'],'evaluation_archive_bytes':proof['archive_bytes'],
          'retention_manifest_sha256':sha(manifest_raw),'retention_proof_sha256':sha(proof_raw),
          'terminal_receipt_sha256':sha(terminal_raw),'evaluation_input_binding_sha256':sha(request_raw),
          'original_author_archive_sha256':request['author_archive_sha256'],'original_author_retention_proof_sha256':sha(author_proof_raw),
          'submitted_scene_sha256':request['declaration'].get('final_scene_sha256'),
          'submitted_bindings_sha256':request['declaration'].get('bindings_sha256'),
          'selected_trace_archive_rehashed':bool(trace_members),
          'scope':'Retained evidence integrity and original author binding; no new eligibility or physical adjudication.'})
    require(bool(seen) and (allow_partial or seen==RUNS),'All twenty author attempts required')
    require(not allow_partial or seen<=RUNS,'Unexpected partial identity')
    complete=seen==RUNS
    public_index={'schema_version':'independent-physical-evidence-index.v1','complete_twenty_attempts':complete,
      'provided_runs':len(seen),'pending_run_ids':sorted(RUNS-seen),'blinded_adjudication_seal_sha256':SEAL_SHA,
      'frozen_experiment_sha256':FREEZE_SHA,'metadata_index_original_sha256':sha(index_raw),
      'runs':sorted(runs,key=lambda r:r['author_run_id']),
      'scope':'Exact trusted evaluator summaries and exact native reports are distinct. Early submission failures may have no native report. This publication does not decide protocol eligibility.'}
    staged['evidence_index.json']=encode(public_index)
    staged['README.md']=README.encode()
    staged['verify_physical_evidence.py']=Path(__file__).read_bytes()
    staged['privacy_scanner.py']=(root/'controller/publication.py').read_bytes()
    for rel in ('evidence_index.json','README.md','verify_physical_evidence.py','privacy_scanner.py'):
        scan(staged[rel],rel,scanner)
        records.append({'path':rel,'bytes':len(staged[rel]),'sha256':sha(staged[rel]),'kind':'generated_metadata' if rel in ('README.md','evidence_index.json') else 'verification_tool'})
    publication={'schema_version':'independent-physical-evidence-publication.v1','complete_twenty_attempts':complete,
      'files':sorted(records,key=lambda r:r['path']),'frozen_privacy_scanner_sha256':SCANNER_SHA,
      'builder_sha256':sha(Path(__file__).read_bytes()),'source_index_sha256':sha(index_raw),
      'original_archives_provided':False,'original_manifests_provided':False,
      'scope':'Reports are exact retained bytes. Standalone traces are lossless gzip derivatives with exact decompressed member bindings. Archive and private proof hashes attest separately retained originals; their complete bytes are not supplied.'}
    staged['publication_manifest.json']=encode(publication);scan(staged['publication_manifest.json'],'publication_manifest.json',scanner)
    require(all(file_sha(p)==h for p,h in private_inputs.items()),'Input changed during build')
    out.mkdir(parents=True)
    for rel,raw in staged.items():
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(raw)
    result=verify(out)
    return result,{'private_input_sha256':private_inputs,'input_metadata_unchanged':True,
      'archive_policy':'Only archives containing selected evaluator numeric traces were rehashed/read; other report bytes were checked against previously verified retention manifests.',
      'excluded_classes':['geometry','authored_programs','model_history','requests','runtime_configuration','solver_logs','process_logs','worker_configuration'],
      'no_new_scoring':True,'remote_operations':False,'public_worktree_modified':False}

def verify(root):
    root=Path(root).resolve();manifest=parse(regular(root/'publication_manifest.json',root))
    require(manifest['schema_version']=='independent-physical-evidence-publication.v1','Unexpected manifest schema')
    scanner=load_scanner(root/'privacy_scanner.py');scan(encode(manifest),'publication_manifest.json',scanner)
    seen={'publication_manifest.json'};evidence={}
    for row in manifest['files']:
        rel=row['path'];safe_rel(rel);require(rel not in seen,'Duplicate publication member');seen.add(rel)
        raw=regular(root/rel,root)
        require(sha(raw)==row['sha256'] and len(raw)==row['bytes'],'Publication member changed')
        decoded=raw
        compressed=row['kind']=='lossless_gzip_derivative'
        if compressed:
            require(row['original_bytes_provided'] is False and row['original_member_path'].endswith('/trace.jsonl'),'False gzip original-byte claim')
            compression=row['compression']
            require(compression=={'format':'gzip','level':9,'mtime':0,'filename_metadata':False,
                'decompressed_sha256':row['original_member_sha256'],'decompressed_bytes':row['original_member_bytes'],
                'arithmetic_or_reserialization':False},'Compression metadata differs')
            decoded=decompress_trace(raw,row['original_member_bytes'],row['original_member_sha256'])
        detail=scan(decoded,rel[:-3] if compressed else rel,scanner)
        if row['kind'] in ('exact_retained_bytes','lossless_gzip_derivative'):
            require(row['author_run_id'] in RUNS and SELECTED.fullmatch(row['original_member_path']),'Unauthorized original evidence')
            require(sha(decoded)==row['original_member_sha256'],'Original-byte binding differs')
            if not compressed and 'original_member_bytes' in row:
                require(row['original_bytes_provided'] is True and row['original_member_bytes']==len(raw),'Exact-byte size claim differs')
            expected=public_path(row['author_run_id'],row['original_member_path'])+('.gz' if compressed else '')
            require(rel==expected,'Evidence relocation differs')
            if detail is not None:require(detail==row['numeric_trace_summary'],'Trace metadata differs')
            evidence[rel]=row
        else:require(rel in ('README.md','evidence_index.json','verify_physical_evidence.py','privacy_scanner.py'),'Unexpected generated member')
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() or p.is_symlink()}==seen,'Unlisted publication member')
    require(file_sha(root/'verify_physical_evidence.py')==manifest['builder_sha256'],'Builder/verifier binding differs')
    index=parse(regular(root/'evidence_index.json',root));ids=[r['author_run_id'] for r in index['runs']]
    require(len(ids)==len(set(ids)) and set(ids)<=RUNS and index['provided_runs']==len(ids),'Run coverage differs')
    require(index['pending_run_ids']==sorted(RUNS-set(ids)),'Pending attempts omitted')
    require(index['complete_twenty_attempts']==manifest['complete_twenty_attempts']==(set(ids)==RUNS),'False complete coverage')
    require(index['blinded_adjudication_seal_sha256']==SEAL_SHA and index['frozen_experiment_sha256']==FREEZE_SHA,'Protocol bindings differ')
    referenced=set()
    for run in index['runs']:
        require(run['summary_path'] in run['selected_members'],'Missing summary reference')
        summary=parse(regular(root/run['summary_path'],root))
        require(summary['status']==run['summary_status'],'Summary status changed')
        require(summary['run_id']=='eval_measured-v2_'+run['author_run_id'],'Summary author binding differs')
        for rel in run['selected_members']:
            row=evidence[rel];require(rel not in referenced,'Duplicate selected reference');referenced.add(rel)
            require(row['author_run_id']==run['author_run_id'] and row['evaluation_archive_sha256']==run['evaluation_archive_sha256'] and row['retention_manifest_sha256']==run['retention_manifest_sha256'],'Retention references differ')
        if run['authoritative_report_sha256'] is not None:
            rel=public_path(run['author_run_id'],run['authoritative_report_original_path'])
            require(rel in run['selected_members'] and evidence[rel]['sha256']==run['authoritative_report_sha256']==summary['report_sha256'],'Authoritative report differs')
            native=parse(regular(root/rel,root))
            parser_path=public_path(run['author_run_id'],'home/evidence/parser_preflight.json')
            parser=parse(regular(root/parser_path,root)) if parser_path in run['selected_members'] else {}
            require(summary['status']==wrapper_status(native,parser),'Public native/summary status differs')
        native_members=[rel for rel in run['selected_members'] if evidence[rel]['original_member_path'] in ('home/evidence/result/acceptance.json','home/evidence/result/report.json')]
        rejection=any(evidence[rel]['original_member_path']=='home/evidence/submission_rejection.json' for rel in run['selected_members'])
        require(run['native_report_present']==bool(native_members) and run['submission_rejection_present']==rejection,'Report presence flags differ')
        if 'report' not in summary:require(rejection and summary['status']=='FAIL','Missing legitimate early rejection')
    require(referenced==set(evidence),'Unreferenced or omitted selected evidence')
    return {'passed':True,'complete_twenty_attempts':index['complete_twenty_attempts'],'provided_runs':len(ids),
      'files':len(seen),'retained_evidence_members':len(evidence),'exact_report_or_uncompressed_trace_members':sum(r['kind']=='exact_retained_bytes' for r in evidence.values()),'lossless_gzip_trace_members':sum(r['kind']=='lossless_gzip_derivative' for r in evidence.values()),'numeric_traces':sum('numeric_trace_summary' in r for r in evidence.values()),
      'bytes':sum((root/p).stat().st_size for p in seen),'manifest_sha256':file_sha(root/'publication_manifest.json'),
      'scope':'Membership, exact-byte bindings, privacy scan and report/summary linkage; no simulation or new physical verdict.'}

README='''# Independent physical evidence

Read evidence_index.json first. A development bundle with complete_twenty_attempts
false is incomplete and explicitly lists pending attempts. A final edition covers
all twenty author attempts, preserving PASS, FAIL, INCONCLUSIVE and individual
checks without selecting favorable seeds or suppressing failed metrics.

Each evaluation_summary.json is the trusted evaluator wrapper summary. The exact
native acceptance.json or report.json, parser/structural/preflight reports, and
per-seed reports are separate evidence. A summary FAIL for an immutable scene or
bindings that were not submitted legitimately has a submission-rejection report
and no native physical report. Missing reports never become synthetic PASS data.
Native report status and protocol eligibility answer different questions; this
companion does not adjudicate eligibility or comparative benefit.

Reports retain all original numerical thresholds, metrics, checks, and any embedded
trace arrays. Standalone numeric JSONL traces are supplied as lossless gzip derivatives.
Decompression returns their exact original bytes and format; no CSV conversion,
rounding, arithmetic, or reserialization is performed. The manifest separately
binds supplied compressed bytes and exact decompressed original-member bytes.
Decompression is bounded by that retained member length. Trace
sample counts do not independently prove task acceptance. Drawer scene-wide
contact samples do not identify contact actors; sensor reports and their original
limits must be read as supplied. No new contact provenance is inferred.

Run python3 -B verify_physical_evidence.py --verify . to verify provided membership,
hashes, privacy and summary/report bindings. No simulator or author code is run.
The bundled privacy scanner has the exact preregistered digest. Verification of
this companion is an integrity check, not a replay of the physical evaluation.

The manifest distinguishes exact retained reports, lossless gzip trace derivatives,
and generated index/documentation. The gzip bytes are newly generated and are not
falsely described as original supplied bytes. Each original member is bound to its evaluation archive/manifest
and original author run. The full archives and private retention proofs are not
provided: their hashes attest separately retained originals. Extraction never
follows links and selects only bounded regular evaluator reports/traces. Requests,
runtime configurations, solver/process logs, authored programs, model histories,
and geometry are excluded. Cases 05 and 06 contain no geometry payload here.
'''

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--index',type=Path);p.add_argument('--output',type=Path)
    p.add_argument('--private-review',type=Path);p.add_argument('--allow-partial-development',action='store_true')
    a=p.parse_args()
    if a.verify:r=verify(a.verify)
    else:
        require(a.root and a.index and a.output and a.private_review,'Build arguments required')
        require(not a.private_review.exists(),'Create-only review exists')
        r,provenance=build(a.root,a.index,a.output,a.allow_partial_development)
        with a.private_review.open('xb') as f:f.write(encode({'verification':r,'provenance':provenance}))
    print(json.dumps(r,indent=2))
