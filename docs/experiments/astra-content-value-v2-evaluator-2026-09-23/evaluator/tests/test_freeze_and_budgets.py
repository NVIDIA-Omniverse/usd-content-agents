"""Frozen reference failures are infrastructure; no native process is run."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('case', sorted(p.name for p in (ROOT / 'cases').iterdir() if p.is_dir()))
def test_public_budgets_match_process_defaults(case):
    task = json.loads((ROOT / 'tasks' / (case + '.json')).read_text())
    assert task['public_acceptance']['runtime_budgets_s'] == {
        'input_stage_load': 900, 'complete_source_stage_inspection': 1800,
        'each_native_trial': 300 if case == '02_conveyor' else 900,
        'engine_paired_unloaded_trial': 900,
    }
    import inspect
    sys.path.insert(0, str(ROOT / 'common'))
    from v2_process import inspect_with_deadline, run
    assert inspect.signature(inspect_with_deadline).parameters['timeout_s'].default == 1800
    assert inspect.signature(run).parameters['timeout'].default == 900
    assert 'timeout_s=900' in (ROOT / 'cases' / case / 'v2_inputs.py').read_text()


def test_changed_trusted_inventory_hash_is_inconclusive(tmp_path):
    implementation = tmp_path / 'implementation'
    shutil.copytree(ROOT / 'cases/03_hinge', implementation, ignore=shutil.ignore_patterns('reference', '__pycache__'))
    (implementation / 'frozen_manifest.json').write_text(json.dumps({
        'reference_inventory_sha256': '0' * 64, 'code_sha256': {},
    }))
    generated = ROOT / 'qualification/local_preparation/03_hinge/generated'
    output = tmp_path / 'result'
    command = [sys.executable, str(implementation / 'evaluate.py'),
               '--usd', str(generated / 'positive.usda'), '--bindings', str(generated / 'bindings.json'),
               '--inventory', str(generated / 'reference/source_inventory.json'),
               '--source-root', str(generated / 'source'), '--output', str(output),
               '--case', '03_hinge', '--structural-only']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    report = json.loads((output / 'acceptance.json').read_text())
    assert report['status'] == 'inconclusive'
    assert 'reference_inventory_frozen' in report['inconclusive_checks']
    assert 'reference_inventory_frozen' not in report['concrete_failures']


@pytest.mark.parametrize('stale', ['qualification/final_receipt.json', 'cases/03_hinge/frozen_manifest.json'])
def test_freeze_preflight_rejects_partial_state_without_writing(tmp_path, stale):
    sys.path.insert(0, str(ROOT))
    from freeze_package import preflight_new_outputs
    case = tmp_path / 'cases/03_hinge'; case.mkdir(parents=True)
    stale_path = tmp_path / stale; stale_path.parent.mkdir(parents=True, exist_ok=True)
    stale_path.write_text('{"retained":"prior partial attempt"}')
    before = {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    with pytest.raises(AssertionError, match='no files written'):
        preflight_new_outputs(tmp_path, [case])
    assert before == {str(p.relative_to(tmp_path)): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
