"""Freeze only independently qualified cases02–10; never freezes root's drawer.

Run after every required receipt is local. No native/model invocation, source
conversion or author artifact is performed here. Refuse an existing freeze.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load(relative):
    return json.loads((ROOT / relative).read_text())


def dump(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def entries(paths):
    return [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
             'sha256': sha(p)} for p in sorted(set(paths))]


def retained(path):
    return path.is_file() and not path.is_symlink() and not any(
        part in ('__pycache__', '.pytest_cache') for part in path.parts)


def preflight_new_outputs(root, cases):
    targets = [root / 'freeze_manifest.json', root / 'qualification_manifest.json',
               root / 'qualification/final_receipt.json']
    targets += [case / name for case in cases for name in
                ('freeze.json', 'frozen_manifest.json', 'v2_freeze.json')]
    existing = [str(p) for p in targets if p.exists() or p.is_symlink()]
    assert not existing, 'Existing/partial freeze is immutable; no files written: ' + repr(existing)


def main():
    cases = sorted(p for p in (ROOT / 'cases').iterdir() if p.is_dir())
    assert len(cases) == 9
    preflight_new_outputs(ROOT, cases)
    q = 'qualification/'
    controls = {
        q + 'native_task_qualification.json': 'qualified',
        q + 'native_cam_extra_v2/summary.json': 'qualified',
        q + 'native_calibration_v2/summary.json': 'qualified',
        q + 'final_structural_controls/summary.json': 'qualified',
        q + 'branch_controls_v2/summary.json': 'qualified',
        q + 'source_hook_controls/summary.json': 'passed',
        q + 'reference_disposition_v2/result.json': 'qualified',
        q + 'paired_unloaded_calibration_review.json': 'passed',
        q + 'source09_evaluation_v2/summary.json': 'qualified',
        q + 'final_component_parity.json': 'runtime_components_match_all_five_seed_matrix',
    }
    for path, field in controls.items():
        assert load(path)[field] is True, path
    native = load(q + 'native_task_qualification.json')
    assert native['positive_count'] == 45 and native['negative_count'] == 13
    assert native['production_cam_seeds'] == [11, 23, 47, 83, 131]
    assert native['production_cam_broken_linkage_negative_passed'] is True
    performance = load(q + 'source09_evaluation_v2/summary.json')
    assert performance['complete'] is True and len(performance['rows']) == 3
    source = load(q + 'source09_evaluation_v2/retessellated_all_parts/result.json')
    assert source['part_count'] == 3092 and source['full_inventory'] is True
    assert source['comparison_code_sha256'] == sha(ROOT / 'common/v2_geometry.py')
    assert '182 passed' in (ROOT / q / 'final_freeze_tests.log').read_text()
    invocation = load('invocations.json')
    assert sorted(x['case_id'] for x in invocation['cases']) == [c.name for c in cases]
    for case in cases:
        task = load('tasks/' + case.name + '.json')
        assert task['frozen'] is False
        assert task['source_inventory_sha256'] == sha(ROOT / 'tasks' / (case.name + '_source_inventory.json'))
        assert (case / 'evaluate.py').is_file() and (case / 'bindings.schema.json').is_file()
    # Resolve every required input before the first exclusive-create write.
    for path in [ROOT / q / 'final_test_environment.json', ROOT / q / 'source09_evaluation_v2.limits.json',
                 ROOT / q / 'source09_remote_v2/summary.json', ROOT / q / 'source09_remote_v2.limits.json']:
        assert path.is_file()
    # Verify every original reference, including every instance, before freezing.
    part_count = 0
    for case in cases:
        for part in json.loads((case / 'reference/source_inventory.json').read_text())['parts']:
            assert sha(case / 'reference' / part['geometry_file']) == part['geometry_sha256']
            part_count += 1
    created = datetime.now(timezone.utc).isoformat()
    final_receipt = {
        'schema_version': 'evaluator-qualification.v2', 'qualified': True,
        'created_at': created, 'scope': 'Cases02–10 synthetic/native/source-only controls; no benchmark authored output or model call.',
        'positive_native_trials': 45, 'negative_native_controls': 13,
        'additional_production_gripper_cam_positive_seeds': [11, 23, 47, 83, 131],
        'additional_production_gripper_broken_linkage_negative': True,
        'local_tests_passed': 182, 'source_reference_instances_rehashed': part_count,
        'control_receipts': entries([ROOT / p for p in controls] + [
            ROOT / q / 'final_freeze_tests.log', ROOT / q / 'final_test_environment.json',
            ROOT / q / 'source09_evaluation_v2.limits.json',
            ROOT / q / 'source09_remote_v2/summary.json',
            ROOT / q / 'source09_remote_v2.limits.json']),
        'post_native_changes': 'Input/reference/calibration-dependency failure classification and bounded inspection helpers only; native controllers/solvers, measured predicates, source comparator and all thresholds retain qualified bytes. Failed physics predicates remain raw observations but are inconclusive when their seed witness failed; the paired engine response requires both witnesses. Complete structural inspection1800s, separate input load900s; native02 trial300s, others900s.',
        'prior_failures_preserved': 'Original loose-point false rejection; initial Trimesh stall;900s2CPU/8GiB full-source timeout; RLIMIT_AS native refusal; initial08/10 zero-span fixture refusals; initial small-inertia cam oscillation. No unsuccessful run overwritten.',
        'limits': 'Synthetic native controls prove measured operating regimes, not arbitrary submitted inertia/collider cookability. Source metric preserves all original instances and separately checks loose points; retopology uses deterministic symmetric samples and vertices. Full source visuals are not a solved printer. Global scoring isolation and drawer qualification remain root-owned.',
        'model_calls': 0,
    }
    receipt_path = ROOT / q / 'final_receipt.json'
    dump(receipt_path, final_receipt)
    evidence_paths = [p for p in (ROOT / 'qualification').rglob('*') if retained(p)]
    evidence_manifest = ROOT / 'qualification_manifest.json'
    dump(evidence_manifest, {'schema_version': 1, 'files': entries(evidence_paths),
                            'exclusions': ['__pycache__', '.pytest_cache'],
                            'scope': 'Retained preparation controls, including failed attempts and unused experiments; not every file is positive evidence.'})
    case_records = []
    for case in cases:
        code = {p.name: sha(p) for p in sorted(case.iterdir()) if p.is_file()}
        inventory = case / 'reference/source_inventory.json'
        runtime_name = 'freeze.json' if case.name == '02_conveyor' else 'frozen_manifest.json'
        runtime = ({'files': code, 'source_inventory_sha256': sha(inventory)}
                   if case.name == '02_conveyor' else
                   {'code_sha256': code, 'reference_inventory_sha256': sha(inventory)})
        dump(case / runtime_name, runtime)
        snapshot = {'case_id': case.name, 'frozen_at': created,
                    'runtime_manifest': runtime_name, 'runtime_manifest_sha256': sha(case / runtime_name),
                    'code_sha256': code, 'reference_files': entries(p for p in (case / 'reference').rglob('*') if retained(p)),
                    'qualification_receipt_sha256': sha(receipt_path),
                    'qualification_manifest_sha256': sha(evidence_manifest)}
        dump(case / 'v2_freeze.json', snapshot)
        task_path = ROOT / 'tasks' / (case.name + '.json')
        task = json.loads(task_path.read_text())
        task.update(frozen=True, private_evaluator_snapshot_sha256=sha(case / 'v2_freeze.json'),
                    evaluator_qualification={'frozen': True, 'native_requalification': 'passed',
                                             'qualification_receipt_sha256': sha(receipt_path)},
                    freeze_basis='Fresh v2 local, source-only and native synthetic qualification completed before scored authors; source/physics scope remains task-specific.')
        task_path.write_text(json.dumps(task, indent=2, allow_nan=False) + '\n')
        case_records.append({'case_id': case.name, 'snapshot_sha256': sha(case / 'v2_freeze.json'),
                             'task_sha256': sha(task_path), 'part_count': len(snapshot['reference_files']) - 1})
    invocation_path = ROOT / 'invocations.json'
    invocation = json.loads(invocation_path.read_text())
    invocation['status'] = 'qualified_frozen'
    invocation['qualification_receipt_sha256'] = sha(receipt_path)
    invocation_path.write_text(json.dumps(invocation, indent=2) + '\n')
    files = [p for folder in ('cases', 'tasks', 'common', 'tests') for p in (ROOT / folder).rglob('*') if retained(p)]
    files += [p for p in ROOT.iterdir() if p.is_file() and p.name != 'freeze_manifest.json']
    dump(ROOT / 'freeze_manifest.json', {'schema_version': 'evaluator-freeze.v2', 'frozen_at': created,
         'scope': 'Cases02–10 only; root combines separate drawer and global protocol freeze.',
         'cases': case_records, 'files': entries(files),
         'qualification_manifest_sha256': sha(evidence_manifest),
         'qualification_receipt_sha256': sha(receipt_path),
         'invocations_sha256': sha(invocation_path)})
    print(json.dumps({'freeze_manifest_sha256': sha(ROOT / 'freeze_manifest.json'),
                      'qualification_receipt_sha256': sha(receipt_path), 'cases': len(cases),
                      'source_instances': part_count}))


if __name__ == '__main__':
    main()
