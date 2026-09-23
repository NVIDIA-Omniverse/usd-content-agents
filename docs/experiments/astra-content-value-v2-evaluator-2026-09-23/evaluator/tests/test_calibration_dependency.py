"""Reclassify retained synthetic observations, without rerunning any simulator."""
import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from v2_verdict import WITNESSES, classify_uncalibrated_runtime, witness_check
sys.path.insert(0, str(ROOT / 'cases/03_hinge'))
from common import verdict

CASES = sorted(p.name for p in (ROOT / 'cases').iterdir() if p.is_dir())


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('mode', ['calibrated', 'witness_failed', 'independent_source_failed'])
def test_retained_blocked_controls_calibration_dependency(case, mode):
    folder = 'branch_native_corrections_v2' if case in ('08_excavator', '10_complex') else 'native_matrix_v2'
    record = json.loads((ROOT / 'qualification' / folder / (case + '_blocked_motion_11') / 'observed.json').read_text())
    checks = record['checks']
    if isinstance(checks, dict):
        checks = [{'name': 'seed11:' + name, 'passed': passed,
                   'failure_class': None if passed else ('evaluator_insufficient' if name == 'independent_gravity' else 'rejected')}
                  for name, passed in checks.items()]
    assert verdict(checks)['status'] == 'not_accepted'
    witness = [x for x in checks if x['name'].partition(':')[2] in WITNESSES]
    assert len(witness) == 1 and witness[0]['passed']
    if mode != 'calibrated':
        witness[0].update(passed=False, failure_class='evaluator_insufficient')
    if mode == 'independent_source_failed':
        checks.append({'name': 'source_surface:independent_negative', 'passed': False, 'failure_class': 'rejected'})
    original = copy.deepcopy(checks)
    classified = classify_uncalibrated_runtime(checks)
    assert checks == original
    assert [x['passed'] for x in classified] == [x['passed'] for x in checks]
    assert [x.get('evidence') for x in classified] == [x.get('evidence') for x in checks]
    result = verdict(classified)
    assert result['status'] == ('inconclusive' if mode == 'witness_failed' else 'not_accepted')
    if mode == 'calibrated':
        assert classified == checks
    elif mode == 'independent_source_failed':
        assert result['concrete_failures'] == ['source_surface:independent_negative']
    else:
        assert not result['concrete_failures']
        assert any(x.get('raw_failure_class') == 'rejected' for x in classified)


@pytest.mark.parametrize('bad_run', ['neither', 'loaded', 'unloaded', 'missing_unloaded'])
def test_engine_paired_response_requires_both_witnesses(bad_run):
    native = json.loads((ROOT / 'qualification/native_matrix_v2/06_engine_positive_11/result.json').read_text())
    good = witness_check(native, 'seed11')
    assert good['passed']
    other = witness_check(native, 'paired_seed11')
    if bad_run == 'loaded': good.update(passed=False, failure_class='evaluator_insufficient')
    if bad_run == 'unloaded': other.update(passed=False, failure_class='evaluator_insufficient')
    if bad_run == 'missing_unloaded': other = witness_check({}, 'paired_seed11')
    checks = [good, other, {'name': 'engine_paired_load_response', 'passed': False, 'failure_class': 'rejected'}]
    result = verdict(classify_uncalibrated_runtime(checks))
    assert result['status'] == ('not_accepted' if bad_run == 'neither' else 'inconclusive')
