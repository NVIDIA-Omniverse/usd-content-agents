"""Keep raw failed observations while respecting their calibration dependency."""
import math

WITNESSES = frozenset(('independent_freefall_witness',
                       'gripper_independent_gravity_witness', 'independent_gravity'))


def witness_check(result, prefix):
    """Same native witness tolerances, also required for the paired engine run."""
    witness = result.get('gravity_witness') or {}
    sample = witness.get('at_point_one_s')
    good = False
    if sample is not None:
        t = sample['t']; vz = sample['velocity'][2]
        dz = sample['pose'][2] - witness['initial_pose'][2]
        good = all(math.isfinite(x) for x in (t, vz, dz)) and abs(vz + 9.81*t) < .03 and abs(dz + .5*9.81*t*t) < .004
    return {'name': prefix + ':independent_freefall_witness', 'passed': good,
            'evidence': witness, 'failure_class': None if good else 'evaluator_insufficient'}


def classify_uncalibrated_runtime(checks):
    failed = set()
    for row in checks:
        prefix, _, criterion = row['name'].partition(':')
        if criterion in WITNESSES and not row['passed']:
            failed.add(prefix)
    output = []
    for original in checks:
        row = dict(original)
        prefix = row['name'].partition(':')[0]
        dependent = prefix in failed
        if row['name'] == 'engine_paired_load_response':
            dependent = bool(failed & {'seed11', 'paired_seed11'})
        if dependent and not row['passed'] and row['failure_class'] != 'evaluator_insufficient':
            row['raw_failure_class'] = row['failure_class']
            row['failure_class'] = 'evaluator_insufficient'
            row['calibration_dependency'] = 'Failed independent gravity witness; raw measurement and failed predicate retained, but no concrete physics rejection is inferred from this unqualified run.'
        output.append(row)
    return output
