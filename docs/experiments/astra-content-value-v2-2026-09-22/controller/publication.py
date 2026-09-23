"""Create-only, allowlisted public preregistration; never publish or run jobs.

A freeze manifest attests originals, not permission to publish their bytes.
Projection records explicitly retain the original and published SHA256 digests.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

SCHEMA = 'astra-content-value-public-preregistration.v2'
CASE = re.compile(r'(?:0[1-9]|10)_[a-z0-9_]+\Z')
SECRET = re.compile(r'(?:-----BEGIN [A-Z ]*PRIVATE KEY|\b(?:sk|ghp|github_pat)-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._~-]{12,}|https?://[^\s/]+:[^\s/@]+@)', re.I)
PRIVATE = re.compile(r'(?:/Users/[A-Za-z0-9_.-]+(?:/[^\s"<>]*)?|/home/(?!agent(?:/|\b))[A-Za-z0-9_.-]+(?:/[^\s"<>]*)?|horde@[\w.-]+|\b[\w.-]+\.teleport\.sh\b|\b[\w.-]+\.(?:horde|nvidia)\.com\b|GPU-[0-9a-f-]{20,}|\blingq-(?:astra|codex)[\w-]*)', re.I)
DENIED_COMPONENTS = {'private', 'session_captures', 'rollouts', 'raw', 'auth', '.codex', '.ssh', 'node_modules', '__pycache__'}
DENIED_NAMES = re.compile(r'(?:\.private\.|model_catalog|base_instructions|worker_config|gateway_config|tool_calls|session_meta|events\.jsonl|\.tgz$|\.tar$|\.gz$|\.key$)', re.I)
PRIVATE_KEYS = re.compile(r'(?:^|_)(?:token|password|secret|credential|authorization|auth|prompts?|messages?|response_ids?|request_ids?|session_ids?|child_session_id|thread_ids?|turn_ids?|uncovered_threads|own_turn_anchors|rollout|base_instructions|api_key|base_url)(?:$|_)|^(?:content|arguments|stdout|stderr|transcript|reasoning|tool_output|raw_output|raw_response)$', re.I)
METADATA_KEYS = set('schema_version status passed accepted ready frozen created_utc captured_at_utc started_utc finished_utc completed_utc elapsed_seconds model reasoning_effort wire_reasoning_effort codex_version source_identity_equal different_file_paths compared_file_count excluded end_to_end_workflow_qualified a_semantic_sha256 b_semantic_sha256 generated_cache_mount_targets_equal bundled_shader_cache_equal directory_boundary environment_and_native_cpu_ready exact_tools_parity system_runtime_parity_sha256 original_implementation_commit author_snapshot_commit no_scored_author_runs all_stage_chain_qualified pending limitations checks tests local_tests results summary findings open_blockers model_calls scored cgroup_populated source_unchanged cuda_only_assigned_accessible vulkan_only_assigned_accessible cost_accounting_complete complete totals currencies dollars reference_rates inputs outputs artifacts artifact_bindings qualification_sha256 source_sha256 receipt_sha256 code_sha256 manifest_sha256 tool_versions versions required observed count files_count images_count reason scope note notes'.split())
METADATA_KEYS.update('tool_parity_sha256 tool_semantic_sha256 lanes four_full_gpus_verified four_gpu_verified_utc initial_probe_sha256 ui_effort resolved_wire_effort provider_kind qualification_threshold_tokens scored_threshold_override requests requests_with_terminal_usage ui_gateway_response_ids_reconciled evaluator_layout_sha256 author_layout_sha256 author_skeleton_unchanged added_targets builder_sha256 handlers_executed stages provider_sha256 base_instructions_included model_count astra'.split())
METADATA_KEYS.update('hosts reviewed_entrypoints resolved_findings source_files capture_qualification_status prior_document_exact_copy supersedes_reason earlier_inference_corrected live_qualification_required limits automatic_capture_qualified whole_q13_ui_complete association downstream_disconnected'.split())
METADATA_KEYS.update('canonical_repository rejected_alias matched_lane_uid_gid bytes_modes_paths_unchanged_both_hosts changed_metadata_paths repository_entries_verified_per_host cross_host_repository_equal_except_git_admin repository_semantic_sha256 required_subtree_identity verifier_sha256 actual_child_execution_proven provider_file_sha256 provider_install_path'.split())

SOURCES = {
 'controller': ('accounting.py','compare_results.py','build_public_inputs.py','restore_sources.py','retain_run.py','worker_retention.py','evaluate_run.py','evaluation_driver.py','test_evaluate_run.py','build_run_plan.py','publication.py','test_publication.py','test_accounting.py','test_comparison.py','test_retention.py'),
 'harness': ('common.py','isolation.py','run_arm.py','run_queue.py','ledger.py','gateway.py','ui_audit.py','finalize.py','gpu_probe.py','run_gpu_qualification.py','kernel_smoke.py','prepare_affinity.py','network_health.py','run_namespace_qualification.py'),
 'environment': ('README.md','NATIVE_CHILD_ROUTING.md','native_provider.json','check_native_provider_parser.py','check_native_child_route.mjs','prepare_evaluator_rootfs.py','pins.json','thirdparty.constraints.txt','bootstrap.py','launch_bootstrap.sh','build_geogram.sh','verify_geogram_build.py','provision_native.py','record_runtime.py','compare_environments.py','emit_environment.py','smoke_backends.py','make_author_repo.py','export_namespace_tools.py','supplement_namespace_tools.py','repair_namespace_staging_locators.py','normalize_export_metadata.py','compare_namespace_exports.py','namespace_smoke.py','run_namespace_smoke.py','namespace_native_step.py','run_namespace_native_step.py','prepare_ovrtx_cache_targets.py','finalize_namespace_tools.sh','test_environment.py'),
}
SOURCES['controller'] += ('seal_experiment.py','run_evaluations.py','audit_authors.py','EVALUATIONS.md','test_run_evaluations.py')
SOURCES['harness'] += ('request_identity.py',)
METADATA_KEYS.update('ui_request_count request_count terminal_usage_count own_turn_count unmatched_request_count unmatched_response_count admission_complete ui_binding_complete request_ui_complete accounting_complete request_identity_complete'.split())
METADATA_KEYS.update('native_workflow native_task_verdict task_outcome automatic_sessions temporary_staging_within_captured_home unsafe_host_child native_child_sandbox exact_current_harness_matches_launch cgroup_reaped all_observed_ui_matches'.split())
SOURCES['environment'] += ('prepare_native_git_ownership.py','probe_native_source_verifier.py','probe_ovphysx_cli_guard.py','probe_render_then_physics.py','probe_render_then_physics_with_parent.py','retain_native_sequence.py','history/NATIVE_CHILD_ROUTING_before_private_staging_discovery.md')
PROTOCOL_METADATA = {'protocol/dataset.json','protocol/analysis_plan.json','protocol/api_reference_rates.json','protocol/astra_ultra_mapping.json','protocol/author_source_closures.json','protocol/original_sources_manifest.json','protocol/treatment_adjudication.json'}
PROTOCOL_METADATA |= {'protocol/thread_capacity_amendment.json','protocol/history/analysis_plan_before_1024_thread_amendment.json'}
ENV_METADATA = {'environment/namespace_parity_v8.json','environment/system_runtime_parity.json','environment/qualification_environment_cpu.json','environment/qualification_gpu_node1.json','environment/source_only_candidate_receipt.json','environment/geogram_build_identity.json','environment/native_provider_parser_exact.json','environment/native_child_route_static.json','environment/codex_builtin_astra_metadata.json','environment/qualification_native_provider_static.json'}
ENV_METADATA -= {'environment/qualification_native_provider_static.json'}
ENV_METADATA |= {'environment/qualification_native_provider_static_v3.json','environment/qualification_native_source_route.json','environment/native_child_capture_analysis.json'}
SAFE_SOURCE_LITERALS = {
 'harness/tests/test_harness.py': {'Bearer '+'synthetic-token','Bearer '+'foreign-token'},
 'environment/export_namespace_tools.py': {'/home/'+'version'},
 'controller/evaluate_run.py': {'/home/'+'evaluator'},
 'harness/network_health.py': {'Bearer '+'synthetic-health-invalid-token'},
}
SAFE_JSON_DIGESTS = {
 # Complete, manually reviewed no-secret loopback provider config. No broad URL/auth exception.
 'environment/native_provider.json': '658b1f75584a9e256b21d03f8051ebe5e47ddd6c03a362337761439b9b1cf84f',
}

def digest(data): return hashlib.sha256(data).hexdigest()
def encode(value): return (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
def safe_relative(value):
    p=Path(value)
    if not isinstance(value,str) or p.is_absolute() or '..' in p.parts or not p.parts:raise ValueError('Unsafe relative path')
    if any(x in DENIED_COMPONENTS for x in p.parts) or DENIED_NAMES.search(value):raise ValueError('Forbidden publication path')
    return p

def source(root, rel):
    p=root/safe_relative(rel)
    if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root.resolve()):raise ValueError('Source is not a contained regular file: '+rel)
    return p.read_bytes()

def scan(data, structured=True, relative_path=None):
    text=data.decode('utf-8')
    allowed=SAFE_SOURCE_LITERALS.get(relative_path,set()) if not structured else set()
    if any(m.group() not in allowed for regex in (SECRET,PRIVATE) for m in regex.finditer(text)):
        raise ValueError('Private locator or secret-like value in selected output')
    if structured and SAFE_JSON_DIGESTS.get(relative_path)!=digest(data):
        def inspect(v):
            if isinstance(v,dict):
                for k,x in v.items():
                    if PRIVATE_KEYS.search(k) and not safe_association_field(k,x) and x not in (None,False,0,'',[],{}):raise ValueError('Private field in exact JSON')
                    inspect(x)
            elif isinstance(v,list):
                for x in v:inspect(x)
        inspect(json.loads(text))

SAFE_ASSOCIATION_COUNTS = {'terminal_response_id_associations','admission_identity_associations','terminal_response_id_and_admission_associations'}
SAFE_ASSOCIATION_BOOLEANS = {'ui_gateway_response_ids_reconciled'}
SAFE_METADATA_COUNTS = {'sessions'}
METADATA_KEYS.update(SAFE_ASSOCIATION_COUNTS | SAFE_ASSOCIATION_BOOLEANS | {'terminal_billing_complete','timing_inference_used','audit_sha256','usage_sha256'})

def safe_association_field(key,value):
    return (key in SAFE_ASSOCIATION_COUNTS and type(value) is int and value >= 0) or (key in SAFE_ASSOCIATION_BOOLEANS and type(value) is bool)

def project(value, metadata_only=False):
    def clean(v):
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if (not PRIVATE_KEYS.search(k) or safe_association_field(k,x)) and (k not in SAFE_METADATA_COUNTS or type(x) is int and x >= 0)}
        if isinstance(v,list):return [clean(x) for x in v]
        if isinstance(v,str):
            if SECRET.search(v):raise ValueError('Secret-like content in selected projection')
            return PRIVATE.sub('<PRIVATE_RUNTIME_LOCATOR>',v)
        return v
    if metadata_only:
        value={k:v for k,v in value.items() if k in METADATA_KEYS or (k in SAFE_METADATA_COUNTS and type(v) is int and v >= 0)}
    return clean(value)

def build(root, freeze_path, destination):
    root=Path(root).resolve();freeze_path=Path(freeze_path).resolve();out=Path(destination)
    if out.exists():raise ValueError('Destination exists; this builder never overwrites')
    if not freeze_path.is_relative_to(root):raise ValueError('Freeze must be inside experiment root')
    freeze_bytes=freeze_path.read_bytes();freeze=json.loads(freeze_bytes)
    if freeze.get('schema_version')!='astra-content-value-freeze.v2' or freeze.get('frozen') is not True or freeze.get('status')!='PASS':raise ValueError('Final successful freeze required')
    cases=freeze['qualified_cases']
    if len(cases)!=10 or len(set(cases))!=10 or any(not CASE.fullmatch(x)for x in cases) or {int(x[:2])for x in cases}!=set(range(1,11)):raise ValueError('All ten unique qualified cases required')
    expected={}
    for row in freeze['files']:
        # Private files may be hash-bound without ever becoming publication candidates.
        rel=row['path'];p=Path(rel)
        if p.is_absolute() or '..' in p.parts or rel in expected:raise ValueError('Invalid or duplicate freeze path')
        expected[rel]=row['sha256']
    protocol=freeze['protocol_file'];inputs=freeze['public_inputs_directory']
    if protocol!='protocol/frozen_protocol.json' or inputs!='protocol/public_inputs':raise ValueError('Unexpected canonical final paths')
    if not {'controller/publication.py','environment/README.md','environment/pins.json'} <= expected.keys():
        raise ValueError('Freeze must bind the public verifier and environment instructions/pins')
    candidates={protocol:'project_json',str(freeze_path.relative_to(root)):'project_freeze'}
    for case in cases:
        for name in ('task.json',case+'_source_inventory.json'):
            candidates[f'{inputs}/{case}/{name}']='exact_json'
    for group,names in SOURCES.items():
        for name in names:
            rel=group+'/'+name
            if rel in expected:candidates[rel]='exact_json' if rel.endswith('.json') else 'exact_text'
    for rel in PROTOCOL_METADATA:
        if rel in expected:candidates[rel]='project_json'
    for case in cases:
        rel=f'provenance/source_manifests/{case}.json'
        if rel in expected:candidates[rel]='project_json'
    for rel in ENV_METADATA:
        if rel in expected:candidates[rel]='metadata'
    for row in freeze.get('qualification_evidence',[]):
        rel=row['path']
        if expected.get(rel)!=row['sha256']:raise ValueError('Qualification is not freeze-bound')
        try:safe_relative(rel)
        except ValueError:continue  # Binding private evidence never authorizes publication.
        if rel.endswith('.json') and rel.split('/')[0] in {'qualification','evidence','harness','environment'}:
            candidates[rel]='metadata'
    # Compute and scan everything first. Failure does not leave a partial public tree.
    staged={};records=[]
    for rel,kind in sorted(candidates.items()):
        original=source(root,rel)
        if kind!='project_freeze' and expected.get(rel)!=digest(original):raise ValueError('Selected frozen bytes changed: '+rel)
        if kind.startswith('exact'):
            published=original;scan(published,kind=='exact_json',rel);projection=False
        else:
            published=encode(project(json.loads(original),kind=='metadata'));scan(published);projection=published!=original
        staged[rel]=published
        records.append({'path':rel,'original_retained_sha256':digest(original),'published_sha256':digest(published),'bytes':len(published),'projection':projection,'projection_rule':kind if projection else None})
    note='''# Controlled experiment v2: preregistration

Results are pending. This is the frozen setup and analysis plan, not a comparison outcome.
Both arms receive the same original inputs, installed implementation and tool runtimes;
workflow activation is the treatment. Actual acceptance comes from separate frozen
physical task evaluations. Runtime qualification does not establish task success.

Run `python controller/publication.py --verify PUBLIC_DIRECTORY` to verify published
bytes. The publication manifest distinguishes exact files from privacy projections.
Original digests in projected records are retained attestations; the public projection
is not the original file and cannot satisfy its original digest. Replace runtime host,
GPU and gateway configuration with independently qualified local values to reproduce.

Environment preparation is described in [environment/README.md](environment/README.md).
Source acquisition manifests retain upstream identity and file digests. This small
preregistration bundle excludes original CAD payloads, binary geometry references,
private evaluator implementation, credentials, worker/gateway configuration, model
catalog contents, base instructions, raw account activity and model/session histories.
The private tunnel supervisor and its host-specific harness test module are retained
outside this preregistration; their qualification summary does not provide those bytes.
No scored results or proof of comparative benefit are included.
'''
    staged['README.md']=note.encode();staged['.gitattributes']=b'* -text\n*.png binary\n*.jpg binary\n*.usdc binary\n*.usdz binary\n*.gz binary\n'
    for rel in ('README.md','.gitattributes'):
        records.append({'path':rel,'original_retained_sha256':None,'published_sha256':digest(staged[rel]),'bytes':len(staged[rel]),'projection':False,'generated':True})
    manifest={'schema_version':SCHEMA,'results_status':'PENDING','freeze_original_sha256':digest(freeze_bytes),'files':sorted(records,key=lambda x:x['path']),'projection_boundary':'Verify published_sha256 against provided bytes. Original retained digests attest freeze-bound private originals, not byte availability.','builder_sha256':digest(Path(__file__).read_bytes())}
    staged['publication_manifest.json']=encode(manifest)
    out.mkdir(parents=True,exist_ok=False)
    for rel,data in sorted(staged.items()):
        p=out/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    return verify(out)

def verify(root):
    root=Path(root);manifest_bytes=(root/'publication_manifest.json').read_bytes();scan(manifest_bytes)
    manifest=json.loads(manifest_bytes)
    if set(manifest)!={'schema_version','results_status','freeze_original_sha256','files','projection_boundary','builder_sha256'}:
        raise ValueError('Unexpected publication manifest fields')
    if manifest['schema_version']!=SCHEMA or manifest['results_status']!='PENDING':raise ValueError('Unexpected publication state')
    expected=set()
    for row in manifest['files']:
        rel=row['path'];safe_relative(rel)
        if rel in expected:raise ValueError('Duplicate publication member')
        expected.add(rel);data=source(root,rel)
        if digest(data)!=row['published_sha256'] or len(data)!=row['bytes']:raise ValueError('Published bytes changed: '+rel)
        if not row['projection'] and row.get('original_retained_sha256') not in (None,row['published_sha256']):raise ValueError('Broken exact-byte claim')
        scan(data,rel.endswith('.json'),rel)
    actual={str(p.relative_to(root))for p in root.rglob('*') if p.is_file() or p.is_symlink()}
    if actual!=expected|{'publication_manifest.json'}:raise ValueError('Unlisted or missing publication member')
    return {'passed':True,'files':len(expected),'manifest_sha256':digest((root/'publication_manifest.json').read_bytes()),'results_status':'PENDING'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path);p.add_argument('--freeze',type=Path);p.add_argument('--output',type=Path);p.add_argument('--verify',type=Path);a=p.parse_args()
    if a.verify:r=verify(a.verify)
    elif a.root and a.freeze and a.output:r=build(a.root,a.freeze,a.output)
    else:p.error('Use --verify, or --root --freeze --output')
    print(json.dumps(r,sort_keys=True))
