"""Create-only quantitative projection; no author code, archives or network.

Actual inputs are permitted only after all twenty blinded audits and independent
evaluations finish. Development qualification uses synthetic fixtures exclusively.
This same file serves as the portable verifier copied into the public bundle.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys

PINS = {
    'controller/accounting.py': 'ef1af0b82852ca6c7789031a7301453af09fac6b8ac1d4bc6c8f01d0032b391f',
    'controller/compare_results.py': '944ff010fd544d1e720e5136bddbfdaa842e3d71ff33c7850712b6a8e368af86',
    'protocol/api_reference_rates.json': '5637a2f729c4764e93c60cf166f2273dd09fb6fa98e2f976b0560b12f3cc7e58',
}
CASES = ('01_drawer','02_conveyor','03_hinge','04_gripper','05_vise','06_engine',
         '07_robot_arm','08_excavator','09_printer','10_complex')
ARMS = ('plain_astra','content_agents')
EFFORTS = ('none','minimal','low','medium','high','xhigh','max','ultra')
TIERS = ('auto','default','priority','flex','scale','batch')
STATES = ('PASS','FAIL','INCONCLUSIVE')
HASH_FIELDS = ('task_sha256','input_sha256','source_sha256','protocol_sha256')
PRICE_KEYS = ('observed','long_context','standard_usd_lower','standard_usd_upper',
              'cached_input_known','cache_write_known','response_tier','known_tier_multiplier',
              'returned_tier_reference_usd_lower','returned_tier_reference_usd_upper')
TOTAL_KEYS = ('run_id','request_count','requests_with_terminal_usage','usage_complete',
              'standard_api_equivalent_usd_lower','standard_api_equivalent_usd_upper',
              'invoice_usd','horde_usd','human_usd')
BOUNDARY = ('Only public assigned run identities and explicit quantitative/categorical allowlists. '
            'Request and intervention ordinals preserve counts/order, not private identity. '
            'Original input hashes attest separately retained private inputs; they do not imply public availability.')


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()


def parse(data):
    def pairs(items):
        out={}
        for k,v in items:
            require(k not in out,'Duplicate JSON key');out[k]=v
        return out
    def nonfinite(value):raise ValueError('Nonfinite JSON number')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def number(value, nullable=False):
    require((nullable and value is None) or (type(value) in (int,float) and
            math.isfinite(value) and value>=0),'Invalid quantitative value')


def integer(value, maximum=None):
    require(type(value) is int and value>=0 and (maximum is None or value<=maximum),'Invalid token count')


def enum(value, allowed):
    require(value is None or value in allowed,'Unexpected public metadata enumeration')
    return value


def tokens(value):
    if value is None:return None
    out={k:value[k] for k in ('input_tokens','output_tokens','total_tokens')}
    for v in out.values():integer(v)
    require(out['total_tokens']==out['input_tokens']+out['output_tokens'],'Token total mismatch')
    for field,keys,maximum in [('input_tokens_details',('cached_tokens','cache_write_tokens'),out['input_tokens']),
                                ('output_tokens_details',('reasoning_tokens',),out['output_tokens'])]:
        if field not in value:continue
        detail=value[field]
        if detail is None:out[field]=None;continue
        require(isinstance(detail,dict),'Invalid token detail record')
        out[field]={k:detail[k] for k in keys if k in detail}
        for v in out[field].values():
            if v is not None:integer(v,maximum)
        if field=='input_tokens_details':
            require(sum(v or 0 for v in out[field].values())<=maximum,'Overlapping input token partitions')
    return out


def request_projection(request, ordinal):
    out={'ordinal':ordinal}
    for key,allowed in [('model',('gpt-6-astra',)),('wire_reasoning_effort',EFFORTS),
                        ('requested_tier',TIERS),('response_tier',TIERS),
                        ('status',('pending','completed','failed','incomplete','cancelled'))]:
        if key in request:out[key]=enum(request[key],allowed)
    if 'usage' in request:out['usage']=tokens(request['usage'])
    return out


def project_rows(rows):
    require(isinstance(rows,list) and len(rows)==20,'All twenty original rows required')
    seen=set();out=[]
    for row in rows:
        case,arm=row['case_id'],row['arm']
        require(case in CASES and arm in ARMS,'Unexpected assigned case/arm')
        identity=(case,arm);require(identity not in seen,'Duplicate case/arm');seen.add(identity)
        # These assigned run names are public experiment identities, not model IDs.
        run_id='v2_'+case+'_'+arm
        require(row['run_id']==run_id and row['lane_id'] in ('lane0','lane1','lane2','lane3'),'Unexpected public run/lane identity')
        require(type(row['protocol_eligible']) is bool and row['independent_status'] in STATES,'Invalid explicit verdict/eligibility')
        result={k:row[k] for k in ('run_id','case_id','arm','lane_id','protocol_eligible','independent_status')}
        for key in HASH_FIELDS:
            require(isinstance(row[key],str) and re.fullmatch('[0-9a-f]{64}',row[key]),'Invalid public source/protocol hash')
            result[key]=row[key]
        if 'claimed_accepted' in row:
            claim=row['claimed_accepted'];require(claim is None or type(claim) is bool,'Invalid author claim')
            result['claimed_accepted']=claim
        for key in ('author_claim_present','ui_association_complete'):
            if key in row:
                require(type(row[key]) is bool,'Invalid metadata boolean');result[key]=row[key]
        if 'protocol_compliance_status' in row:
            result['protocol_compliance_status']=enum(row['protocol_compliance_status'],
                ('observed_compliance','observed_noncompliance','insufficient_evidence'))
        for key in ('author_elapsed_seconds','allocated_lane_seconds','human_review_minutes'):
            if key in row:number(row[key],nullable=True);result[key]=row[key]
        require(isinstance(row['human_interventions'],list),'Explicit human intervention list required')
        result['human_interventions']=[{'ordinal':i+1} for i in range(len(row['human_interventions']))]
        ledger=row['usage_ledger'];require(ledger['run_id']==run_id,'Accounting run identity mismatch')
        require(type(ledger['complete']) is bool and isinstance(ledger['requests'],list),'Invalid accounting ledger')
        public_ledger={'run_id':run_id,'complete':ledger['complete'],
                       'requests':[request_projection(r,i+1) for i,r in enumerate(ledger['requests'])]}
        for key in ('requested_ui_effort','resolved_wire_effort'):
            if key in ledger:public_ledger[key]=enum(ledger[key],EFFORTS)
        if 'request_count' in ledger:
            require(type(ledger['request_count']) is int and ledger['request_count']==len(ledger['requests']),'Request count mismatch')
            public_ledger['request_count']=ledger['request_count']
        result['usage_ledger']=public_ledger;out.append(result)
    require(seen=={(c,a) for c in CASES for a in ARMS},'Missing assigned attempt')
    return out


def calculators(root):
    require(sys.implementation.name == 'cpython' and sys.version_info[:3] == (3, 9, 6),
            'Exact historical replay requires CPython 3.9.6; newer float summation can change final bits')
    require(__debug__,'Frozen comparator uses assertions; Python -O is not supported')
    root=Path(root)
    for path,digest in PINS.items():
        p=root/path
        require(p.is_file() and not p.is_symlink() and sha(p.read_bytes())==digest,'Pinned comparator/accounting/rate bytes differ')
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,root/path)
        module=importlib.util.module_from_spec(spec)
        previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True
        try:spec.loader.exec_module(module)
        finally:sys.dont_write_bytecode=previous
        return module
    accounting=load('frozen_results_accounting','controller/accounting.py')
    previous=sys.modules.get('accounting');sys.modules['accounting']=accounting
    try:comparison=load('frozen_results_comparison','controller/compare_results.py')
    finally:
        if previous is None:sys.modules.pop('accounting',None)
        else:sys.modules['accounting']=previous
    return accounting,comparison


def account_rows(rows,rates,accounting):
    outputs=[]
    for row in rows:
        raw=accounting.reference_run(row['usage_ledger'],rates)
        result={k:raw[k] for k in TOTAL_KEYS}
        result['requests']=[{'ordinal':i+1,**{k:v[k] for k in PRICE_KEYS if k in v}}
                            for i,v in enumerate(raw['requests'])]
        outputs.append(result)
    return {'runs':outputs}


def read_ref(base,ref):
    require(set(ref)=={'path','sha256'},'Exact path/hash reference required')
    require(isinstance(ref['sha256'],str) and re.fullmatch('[0-9a-f]{64}',ref['sha256']),'Invalid reference digest')
    path=Path(base)/ref['path'];require(path.is_file() and not path.is_symlink(),'Missing or symlink input')
    data=path.read_bytes();require(sha(data)==ref['sha256'],'Input hash mismatch')
    return parse(data)


def build(root,input_path,output):
    output=Path(output);require(not output.exists(),'Fresh public output required')
    root=Path(root);input_path=Path(input_path);spec=parse(input_path.read_bytes())
    require(set(spec)=={'schema_version','input_kind','all_twenty_audits_and_evaluations_complete','rows','rates','comparison'},'Unexpected input schema fields')
    require(spec['schema_version']=='results-metadata-inputs.v1' and spec['input_kind'] in
            ('synthetic_fixture','final_assembled_results'),'Unexpected input schema/kind')
    require(spec['all_twenty_audits_and_evaluations_complete'] is True,'Explicit completed audit/evaluation barrier required')
    assembled=read_ref(input_path.parent,spec['rows']);rates=read_ref(input_path.parent,spec['rates'])
    expected=read_ref(input_path.parent,spec['comparison'])
    require(assembled.get('schema_version')=='comparison-rows.v1','Final assembled row schema required')
    require(spec['rates']['sha256']==PINS['protocol/api_reference_rates.json'],'Rates are not the exact frozen rates')
    accounting,comparison=calculators(root)
    original=assembled['runs'];projected=project_rows(original)
    # Recompute, do not trust a supplied favorable comparison or select attempts.
    require(encode(comparison.compare(original,rates))==encode(expected),'Final comparison does not match original rows')
    require(encode(comparison.compare(projected,rates))==encode(expected),'Projection altered comparison')
    priced=account_rows(projected,rates,accounting)
    require(encode(priced)==encode(account_rows(original,rates,accounting)),'Projection altered per-request accounting')
    public={'schema_version':'public-comparison-rows.v1','runs':projected}
    provenance={'schema_version':'results-projection-provenance.v1','input_kind':spec['input_kind'],
        'original_input_sha256':{k:spec[k]['sha256'] for k in ('rows','rates','comparison')},
        'input_manifest_sha256':sha(input_path.read_bytes()),'projector_sha256':sha(Path(__file__).read_bytes()),
        'pinned_calculation_files':PINS,'comparison_exactly_reproduced':True,'per_request_accounting_exactly_reproduced':True,
        'boundary':BOUNDARY}
    files={'rows.json':encode(public),'comparison.json':encode(expected),'per_request_accounting.json':encode(priced),
           'provenance.json':encode(provenance),'README.md':README.encode(),
           'verify_results_metadata.py':Path(__file__).read_bytes()}
    for rel in PINS:files[rel]=(root/rel).read_bytes()
    manifest={'schema_version':'public-results-metadata.v1','input_kind':spec['input_kind'],
              'files':[{'path':rel,'sha256':sha(data),'bytes':len(data)} for rel,data in sorted(files.items())]}
    files['manifest.json']=encode(manifest)
    output.mkdir(parents=True,exist_ok=False)
    for rel,data in files.items():
        p=output/rel;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as stream:stream.write(data)
    return verify(output)


def verify(root):
    root=Path(root).resolve();manifest=parse((root/'manifest.json').read_bytes())
    require(set(manifest)=={'schema_version','input_kind','files'} and
            manifest['schema_version']=='public-results-metadata.v1' and
            manifest['input_kind'] in ('synthetic_fixture','final_assembled_results'),'Unexpected publication manifest')
    names=set()
    for row in manifest['files']:
        require(set(row)=={'path','sha256','bytes'},'Unexpected publication member fields')
        require(isinstance(row['sha256'],str) and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Invalid member digest')
        integer(row['bytes'])
        rel=row['path'];path=Path(rel)
        require(not path.is_absolute() and '..' not in path.parts and path.as_posix()==rel and rel not in names,'Unsafe/duplicate member')
        names.add(rel);p=root/path
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root),'Missing/escaping member')
        raw=p.read_bytes();require(len(raw)==row['bytes'] and sha(raw)==row['sha256'],'Published member changed')
    required={'rows.json','comparison.json','per_request_accounting.json','provenance.json','README.md','verify_results_metadata.py',*PINS}
    require(names==required,'Unexpected publication allowlist')
    require({str(p.relative_to(root)) for p in root.rglob('*') if p.is_file() or p.is_symlink()}==names|{'manifest.json'},'Unlisted publication member')
    accounting,comparison=calculators(root)
    rows_doc=parse((root/'rows.json').read_bytes());rows=rows_doc['runs']
    require(set(rows_doc)=={'schema_version','runs'} and rows_doc['schema_version']=='public-comparison-rows.v1','Unexpected public row fields')
    require(encode(project_rows(rows))==encode(rows),'Public rows contain non-allowlisted or noncanonical data')
    rates=parse((root/'protocol/api_reference_rates.json').read_bytes())
    actual=comparison.compare(rows,rates)
    require(encode(actual)==encode(parse((root/'comparison.json').read_bytes())),'Comparison reproduction differs')
    require(encode(account_rows(rows,rates,accounting))==encode(parse((root/'per_request_accounting.json').read_bytes())),'Per-request accounting differs')
    provenance=parse((root/'provenance.json').read_bytes())
    require(set(provenance)=={'schema_version','input_kind','original_input_sha256','input_manifest_sha256',
            'projector_sha256','pinned_calculation_files','comparison_exactly_reproduced',
            'per_request_accounting_exactly_reproduced','boundary'},'Unexpected provenance fields')
    require(provenance['schema_version']=='results-projection-provenance.v1' and
            provenance['input_kind']==manifest['input_kind'],'Publication kind mismatch')
    require(set(provenance['original_input_sha256'])=={'rows','rates','comparison'},'Unexpected original digest fields')
    for value in [*provenance['original_input_sha256'].values(),provenance['input_manifest_sha256'],provenance['projector_sha256']]:
        require(isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value),'Invalid provenance digest')
    require(provenance['original_input_sha256']['rates']==PINS['protocol/api_reference_rates.json'] and
            provenance['pinned_calculation_files']==PINS and provenance['boundary']==BOUNDARY and
            provenance['comparison_exactly_reproduced'] is True and
            provenance['per_request_accounting_exactly_reproduced'] is True,'Projection boundaries changed')
    require(provenance['projector_sha256']==sha((root/'verify_results_metadata.py').read_bytes()),'Projector/verifier identity differs')
    require((root/'README.md').read_bytes()==README.encode(),'Publication wrapper changed')
    return {'passed':True,'runs':20,'files':len(names)+1,'input_kind':manifest['input_kind'],
            'comparison_exactly_reproduced':True,'per_request_accounting_exactly_reproduced':True,
            'manifest_sha256':sha((root/'manifest.json').read_bytes()),
            'scope':'Public metadata equality and frozen-calculator replay; source audits/evaluations remain separately required.'}


README='''# Quantitative comparison metadata

The manifest labels this bundle as either a synthetic fixture or final assembled
results. A synthetic fixture is qualification data, not an experimental outcome.
Actual result production is authorized only after all twenty blinded protocol
audits and independent evaluations are complete. This builder neither performs
nor substitutes for those audits and physical tests.

Use **CPython 3.9.6**, the original controller/aggregation runtime. Run
`python3.9 -B verify_results_metadata.py --verify .` to verify membership and
recompute the exact comparison and per-request prices using the supplied,
unchanged, digest-pinned accounting.py and compare_results.py. Exact runtime is
enforced: Python 3.12 changed float summation, producing last-bit aggregate cost
differences in synthetic qualification. No tolerance or rounding substitutes for
exact reproduction. See https://docs.python.org/3.12/library/functions.html#sum .
This pin applies to quantitative postprocessing, not the simulation runtime. Python -O is
unsupported because the frozen calculators use assertions. The fixed historical
rates are API-equivalent reference prices, not invoices or current price advice.

rows.json retains all twenty attempts, the explicit eligibility/verdicts/claims,
public assigned case/arm/lane identifiers, input hashes and measured times. Private
model request/response/session/thread/turn identities are absent. Request ordinals
preserve original list order only; they do not identify an account or session.
Human interventions are count placeholders, without private IDs, text or dates.
Unmeasured human minutes remain null. No zero effort or human savings are inferred.

Token counts retain observed input/output totals and available cache/read/write
or reasoning subsets. Missing usage stays missing, unknown cache partitions stay
unknown, and missing terminal cost has an unknown upper bound. Reasoning tokens
are not added again to output totals. All-attempt and eligible-pair calculations,
false positives versus unresolved positive claims, and zero-acceptance undefined
unit costs are reproduced without dropping rows or requests.

per_request_accounting.json omits textual reason/scope fields while preserving
every quantitative accounting field. comparison.json contains only the exact
frozen comparator's output, including its fixed public scope caveats. No author
reason text, prompts, transcripts, adjudication prose, geometry or executable
authored code is copied. Provenance contains original input digests, not private
paths; hashes are attestations to retained originals, not a claim those private
inputs are included here. Projection equality does not establish the truth of
the underlying audit or physics result.
'''


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--verify',type=Path)
    p.add_argument('--root',type=Path);p.add_argument('--manifest',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args()
    if a.verify:result=verify(a.verify)
    elif a.root and a.manifest and a.output:result=build(a.root,a.manifest,a.output)
    else:p.error('Use --verify, or --root --manifest --output')
    print(json.dumps(result,indent=2))
