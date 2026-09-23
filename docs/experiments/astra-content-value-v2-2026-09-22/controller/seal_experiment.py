"""Bind qualified setup and analysis before any scored author is dispatched."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'harness'))
from common import read, require, sha256, object_hash, validate_protocol, write_new
from isolation import REQUIRED_WITNESSES


def seal(profile_path, templates, qualification_paths):
    require(not (ROOT/'protocol/freeze.json').exists(),'Experiment already frozen')
    require(not (ROOT/'protocol/frozen_protocol.json').exists(),'Protocol already exists')
    profile=read(profile_path)
    profile_directory=Path(profile_path).resolve().parent
    for name,digest in profile['files'].items():
        require(Path(name).name==name and sha256(profile_directory/name)==digest,'Shared profile member changed')
    for name,field in (('rootfs_manifest.json','common_rootfs_sha256'),('workflow_manifest.json','workflow_sha256'),('model_gateway_policy.json','model_gateway_policy_sha256')):
        require(object_hash(read(profile_directory/name))==profile['fields'][field],'Shared profile semantic digest differs')
    for name,digest in read(profile_directory/'model_gateway_policy.json')['code'].items():
        require(Path(name).name==name and sha256(ROOT/'harness'/name)==digest,'Qualified harness code changed: '+name)
    design=read(ROOT/'protocol/design_draft.json')
    all_evaluators=read(ROOT/'evidence/all_evaluators_qualification.json')
    require(all_evaluators['status']=='PASS' and len(all_evaluators['qualified_cases'])==10,'All evaluators required')
    require(read(ROOT/'protocol/public_inputs/inputs_manifest.json')['frozen'] is True,'Public inputs not frozen')
    require(read(ROOT/'protocol/analysis_plan.json')['frozen'] is True,'Analysis plan not frozen')
    mapping=read(ROOT/'protocol/astra_ultra_mapping.json')
    manifest=read(Path(templates)/'manifest.json')
    require(manifest.get('deployment_ready') is True,'Worker templates still pending qualification')
    for name in ('node0.json','node1.json'):
        require(sha256(Path(templates)/name)==manifest['files'][name],'Worker template changed')
    lanes=[]
    for row in manifest['lanes']:
        lane={k:row[k]for k in ('id','host','gpu_uuid','cpu_millicores','memory_bytes','pids_max','cpu_affinity')}
        path=ROOT/'harness/qualification/frozen'/(lane['id']+'.json')
        q=read(path)
        require(q.get('status')=='PASS' and q['host_id']==lane['host'] and q['lane_id']==lane['id'],'Unqualified lane')
        require(all(q.get('checks',{}).get(k) is True for k in REQUIRED_WITNESSES),'Missing lane witness')
        for key,value in profile['fields'].items():require(q.get(key)==value,'Lane profile mismatch: '+key)
        require(row.get('qualification_sha256')==sha256(path),'Template qualification digest differs')
        lane['qualification_sha256']=sha256(path);lanes.append(lane)
    protocol={'schema_version':'isolated-lanes.v2','frozen':True,
              'created_utc':datetime.now(timezone.utc).isoformat(),'model':'gpt-6-astra','reasoning_effort':'ultra',
              'wire_reasoning_effort':'xhigh','codex_version':'0.154.0',
              'reasoning_mapping_sha256':sha256(ROOT/'protocol/astra_ultra_mapping.json'),
              'model_catalog_sha256':mapping['astra_catalog_sha256'],
              'resource_rule':'four_exclusive_gpu_lanes','seeds':design['seeds'],
              'wall_seconds':design['per_arm_wall_time_limit_s'],'repair_limit':None,'repair_definition':'measured_not_capped',
              'network_policy':'model_gateway_only','fresh_context':True,'storage':design['storage_budget'],
              'lanes':lanes,'lane_assignment':design['lane_assignment'],
              'remote_experiment_root':'/opt/astra-content-value-20260922-rerun',
              'implementation_commit':design['source_commit'],'experiment_design':design,
              'analysis_plan_sha256':sha256(ROOT/'protocol/analysis_plan.json'),
              'source_closures_sha256':sha256(ROOT/'protocol/author_source_closures.json'),
              'public_inputs_manifest_sha256':sha256(ROOT/'protocol/public_inputs/inputs_manifest.json'),
              **profile['fields']}
    protocol['experiment_design']['state']='FROZEN_BEFORE_SCORED_AUTHORS'
    validate_protocol(protocol)
    # Resolve every gate and selected byte before writing either final file.
    selected=set(x['path'] for x in all_evaluators['files'])
    selected.update(str((Path(templates).resolve()/name).relative_to(ROOT))for name in ('manifest.json','node0.json','node1.json'))
    selected.add('evidence/all_evaluators_qualification.json')
    for group in ('protocol/public_inputs','provenance/source_manifests'):
        selected.update(str(p.relative_to(ROOT))for p in (ROOT/group).rglob('*.json') if p.is_file() and not p.is_symlink())
    for name in ('dataset.json','analysis_plan.json','treatment_adjudication.json','thread_capacity_amendment.json',
                 'history/analysis_plan_before_1024_thread_amendment.json','api_reference_rates.json','astra_ultra_mapping.json',
                 'author_source_closures.json','original_sources_manifest.json'):
        selected.add('protocol/'+name)
    for group in ('protocol/public_inputs','provenance/source_manifests','harness','controller','environment'):
        for path in (ROOT/group).rglob('*'):
            rel=path.relative_to(ROOT)
            # Bind source and selected fixed metadata, never changing caches,
            # auth, model catalogs, histories or running process logs.
            if any(x in {'private','__pycache__','history','controller_review_agent07','gpu_node1_retained'} for x in rel.parts):continue
            if path.is_file() and not path.is_symlink() and path.suffix in ('.py','.sh','.md','.toml','.txt'):
                selected.add(str(rel))
    for group in ('harness/qualification/frozen',str(Path(profile_path).resolve().parent.relative_to(ROOT))):
        selected.update(str(p.relative_to(ROOT))for p in (ROOT/group).glob('*.json'))
    selected.update(('environment/pins.json','environment/native_provider.json'))
    for relative in ('environment/namespace_parity_v8.json','environment/system_runtime_parity.json',
                     'environment/qualification_environment_cpu.json','environment/source_only_candidate_receipt.json',
                     'environment/geogram_build_identity.json','environment/codex_builtin_astra_metadata.json'):
        if (ROOT/relative).exists():selected.add(relative)
    evidence=[]
    for relative in qualification_paths:
        path=ROOT/relative;require(path.is_file() and not path.is_symlink(),'Missing qualification receipt')
        selected.add(relative);evidence.append({'path':relative,'sha256':sha256(path)})
    for row in all_evaluators['files']:require(sha256(ROOT/row['path'])==row['sha256'],'Evaluator freeze changed')
    records=[]
    for relative in sorted(selected):
        path=ROOT/relative
        require(path.is_file() and not path.is_symlink(),'Freeze member missing or symlink: '+relative)
        records.append({'path':relative,'sha256':sha256(path)})
    protocol_path=ROOT/'protocol/frozen_protocol.json';write_new(protocol_path,protocol)
    records.append({'path':'protocol/frozen_protocol.json','sha256':sha256(protocol_path)})
    freeze={'schema_version':'astra-content-value-freeze.v2','frozen':True,'status':'PASS',
            'created_utc':protocol['created_utc'],'qualified_cases':all_evaluators['qualified_cases'],
            'protocol_file':'protocol/frozen_protocol.json','public_inputs_directory':'protocol/public_inputs',
            'files':sorted(records,key=lambda x:x['path']),'qualification_evidence':evidence}
    write_new(ROOT/'protocol/freeze.json',freeze)
    return {'protocol_sha256':sha256(protocol_path),'freeze_sha256':sha256(ROOT/'protocol/freeze.json'),
            'files':len(records),'scored_authors_launched':0}


if __name__=='__main__':
    import json
    p=argparse.ArgumentParser();p.add_argument('--profile',type=Path,required=True);p.add_argument('--templates',type=Path,required=True)
    p.add_argument('--qualification-list',type=Path,required=True);a=p.parse_args()
    print(json.dumps(seal(a.profile,a.templates,read(a.qualification_list)['paths'])))
