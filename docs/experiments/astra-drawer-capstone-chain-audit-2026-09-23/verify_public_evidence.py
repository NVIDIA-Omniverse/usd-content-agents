"""Verify supplied public bytes; explicitly leave unavailable original closure unverified."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

MANIFEST_SHA256 = 'f42d029b483b07aa15743a69cadd431388a325651d2529f6f3ef388c2d41a348'
ASSET_SHA256 = '0ef7038845d569a2f502f38af9fb5b1e4e99cf06c5f4e04035e4c259b1b07156'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(bundle):
    bundle = Path(bundle).resolve()
    raw = (bundle / 'publication_manifest.json').read_bytes()
    if digest(raw) != MANIFEST_SHA256:
        raise ValueError('Unexpected publication manifest')
    manifest = json.loads(raw)
    rows = {row['path']: row for row in manifest['files']}
    for rel, row in rows.items():
        path = bundle / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or path.is_symlink() or not path.resolve().is_relative_to(bundle):
            raise ValueError('Unsafe publication path')
        data = path.read_bytes()
        if digest(data) != row['published_sha256'] or len(data) != row['published_bytes']:
            raise ValueError('Published byte mismatch: ' + rel)

    def load(rel):
        if rel not in rows:
            raise ValueError('Evidence not covered by publication manifest')
        return json.loads((bundle / rel).read_bytes())

    terminal = load('runs/drawer_validation_01_prepared/native_run/validation_terminal_receipt.json')
    closure = []
    for name, value in terminal.items():
        if not isinstance(value, dict) or not {'path', 'sha256'} <= value.keys():
            continue
        matches = [row for row in rows.values()
                   if row.get('original_retained_sha256') == value['sha256']
                   and value['path'].endswith('/' + row['path'])]
        if len(matches) != 1:
            closure.append({'binding': name, 'status': 'ORIGINAL_UNAVAILABLE'})
            continue
        row = matches[0]
        exact = row['mode'] == 'exact' and row['published_sha256'] == value['sha256']
        closure.append({'binding': name, 'public_path': row['path'],
                        'status': 'EXACT_ORIGINAL_BYTES_VERIFIED' if exact else
                                  'ORIGINAL_UNAVAILABLE_PUBLISHED_PROJECTION_VERIFIED',
                        'original_sha256_attestation': value['sha256'],
                        'published_sha256': row['published_sha256']})

    geometry = load('runs/drawer_geometry_04/geometry_validation_evidence.json')
    geometry_result = load('runs/drawer_geometry_04/geometry_workflow_result.json')
    joint = load('runs/drawer_joint_03/standalone_articulation_terminal_receipt.json')
    physics = load('runs/drawer_physics_10/physics_behavior_assessment.json')
    task = load('evaluations/native10_source_clear_v1/report.json')
    assessment = load('runs/drawer_validation_01_prepared/native_run/canonical_validation_assessment.json')
    final_asset = (bundle / 'runs/drawer_physics_10/physics.usda').read_bytes()
    gates = {'static_validation', 'runtime_validation', 'visual_quality', 'package_integrity'}
    checks = {
        'same_final_asset_and_task_digest': digest(final_asset) == ASSET_SHA256 == task['inputs']['usd_sha256'],
        'geometry_still_conditional': geometry['sim_ready_status'] == 'not_evaluated' and
                                     geometry_result['handoff_ready'] == 'conditional',
        'joint_completed_accepted': joint['status'] == 'completed' and joint['terminal_disposition'] == 'accept',
        'native_physics_assessment_pass': physics['status'] == 'pass',
        'native_validation_completed_accepted_pass': terminal['receipt_status'] == 'completed' and
            terminal['terminal_disposition'] == 'pass' and terminal['review_disposition'] == 'accept',
        'four_required_validation_gates_pass': {x['gate'] for x in assessment['gates']} == gates and
            all(x['required'] is True and x['disposition'] == 'pass' for x in assessment['gates']),
        'cross_stage_still_not_evaluated': terminal['gate_dispositions']['cross_stage_integrity'] == 'not_evaluated',
        'five_existing_task_verdicts_pass': task['pass'] is True and task['status'] == 'PASS' and
            [x['seed'] for x in task['trials']] == [11, 23, 47, 83, 131] and
            all(x['pass'] is True and all(x['checks'].values()) for x in task['trials']),
    }
    traces = []
    export = load('task_plots/native10_source_clear_v1/export_verification.json')
    for item in export['trials']:
        seed = item['seed']
        data = gzip.decompress((bundle / f'task_plots/native10_source_clear_v1/seed_{seed}_trace.jsonl.gz').read_bytes())
        okay = digest(data) == item['trace_sha256'] and len(data) == item['uncompressed_bytes'] and len(data.splitlines()) == 2460
        traces.append({'seed': seed, 'complete_published_trace_verified': okay,
                       'samples': len(data.splitlines()), 'uncompressed_sha256': digest(data)})
    if not all(checks.values()) or not all(x['complete_published_trace_verified'] for x in traces):
        raise ValueError('Published evidence disagrees with the bounded summary')
    return {'schema_version': 'capstone-public-byte-audit.v1',
            'status': 'PUBLIC_BYTES_AND_DECLARED_OUTCOMES_VERIFIED',
            'publication_manifest_sha256': MANIFEST_SHA256,
            'published_files_verified': len(rows), 'checks': checks,
            'terminal_bindings': closure, 'traces': traces,
            'full_original_terminal_closure_verified': all(x['status'] == 'EXACT_ORIGINAL_BYTES_VERIFIED' for x in closure),
            'unavailable_original_checks': [x['binding'] for x in closure if x['status'] != 'EXACT_ORIGINAL_BYTES_VERIFIED'],
            'new_task_or_native_acceptance': False,
            'limits': ['No original private receipt is reconstructed from a privacy projection.',
                       'No simulation, model, renderer, metric recomputation or USD decoding is performed.',
                       'Recorded outcomes and byte continuity do not establish universal SimReady, automatic cross-stage validation or hardware fidelity.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle), indent=2, sort_keys=True))
