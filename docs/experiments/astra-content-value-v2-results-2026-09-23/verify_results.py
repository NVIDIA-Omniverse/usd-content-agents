"""Verify published bytes and agreement across results, audits and evidence.

This performs no model, USD, renderer, author-program or physics execution.
It verifies the supplied record; it cannot independently establish its truth.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

CASES = ('01_drawer', '02_conveyor', '03_hinge', '04_gripper', '05_vise',
         '06_engine', '07_robot_arm', '08_excavator', '09_printer', '10_complex')
ARMS = ('plain_astra', 'content_agents')
RUNS = {f'v2_{case}_{arm}' for case in CASES for arm in ARMS}
PRIORITY_RUNS = {f'v2_{case}_{arm}' for case in CASES[:2] for arm in ARMS}
STATES = {'PASS', 'FAIL', 'INCONCLUSIVE'}
COMPLIANCE = {'observed_compliance', 'observed_noncompliance', 'insufficient_evidence'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=pairs)


def safe_name(name):
    require(isinstance(name, str), 'Non-string member path')
    p = PurePosixPath(name)
    require(bool(name) and not p.is_absolute() and '..' not in p.parts and
            p.as_posix() == name and '\\' not in name and
            not any(ord(char) < 32 or ord(char) == 127 for char in name),
            'Unsafe or noncanonical membership')
    return name


def verify(root, deep=False):
    root = Path(root).resolve()
    require((root / 'publication_manifest.json').is_file() and
            not (root / 'publication_manifest.json').is_symlink(), 'Missing regular publication manifest')
    manifest = read(root / 'publication_manifest.json')
    listed = {}
    for row in manifest['files']:
        name = safe_name(row['path'])
        require(name not in listed and name != 'publication_manifest.json',
                'Duplicate or self-referential membership')
        require(type(row['bytes']) is int and row['bytes'] >= 0 and
                isinstance(row['sha256'], str) and re.fullmatch('[0-9a-f]{64}', row['sha256']),
                'Invalid member size or digest')
        path = root / name
        cursor = root
        for component in PurePosixPath(name).parts:
            cursor = cursor / component
            require(not cursor.is_symlink(), 'Symlink in member path')
        require(path.resolve().is_relative_to(root), 'Member escapes publication')
        require(path.is_file() and not path.is_symlink(), 'Missing regular file: ' + name)
        data = path.read_bytes()
        require(len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256'],
                'Published bytes differ: ' + name)
        listed[name] = row
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'Symlink in publication')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(listed) | {'publication_manifest.json'}, 'Membership differs')
    def listed_json(name):
        require(safe_name(name) in listed, 'Referenced JSON is not listed: ' + name)
        return read(root / name)
    rows = listed_json('metrics/rows.json')['runs']
    physical = listed_json('physical_evidence/evidence_index.json')
    require(len(rows) == 20 and {r['run_id'] for r in rows} == RUNS,
            'Twenty canonical unique attempts required')
    require(physical['complete_twenty_attempts'] is True and physical['pending_run_ids'] == [],
            'Incomplete physical evidence')
    require(len(physical['runs']) == 20 and
            len({r['author_run_id'] for r in physical['runs']}) == 20 and
            physical['provided_runs'] == 20, 'Duplicate or incomplete physical attempts')
    native = {r['author_run_id']: r for r in physical['runs']}
    require(set(native) == {r['run_id'] for r in rows}, 'Physical membership differs')
    for row in rows:
        rid = row['run_id']
        require(row['case_id'] in CASES and row['arm'] in ARMS and
                rid == f"v2_{row['case_id']}_{row['arm']}", 'Row case/arm identity differs')
        require(type(row['protocol_eligible']) is bool and row['independent_status'] in STATES,
                'Invalid eligibility or physical status')
        audit = listed_json('protocol_audits/audits/' + rid + '.json')
        require(audit['run_id'] == rid and audit['outcome_blinded'] is True,
                'Audit identity or blinding differs')
        require(type(audit['protocol_eligible']) is bool and
                audit['compliance_status'] in COMPLIANCE and
                row['protocol_compliance_status'] == audit['compliance_status'],
                'Audit compliance status differs: ' + rid)
        for field in ('case_id', 'arm', 'lane_id', 'task_sha256', 'input_sha256',
                      'source_sha256', 'protocol_sha256', 'protocol_eligible'):
            require(audit[field] == row[field], 'Audit differs: ' + rid + ' ' + field)
        require(native[rid]['summary_status'] == row['independent_status'],
                'Independent verdict differs: ' + rid)
        for field in ('case_id', 'arm'):
            require(native[rid][field] == row[field], 'Physical identity differs')
        expected_summary = 'runs/' + rid + '/evaluation_summary.json'
        require(native[rid]['summary_path'] == expected_summary,
                'Physical summary path differs')
        summary = listed_json('physical_evidence/' + expected_summary)
        require(summary['run_id'] == 'eval_measured-v2_' + rid and
                summary['status'] == row['independent_status'],
                'Actual physical summary identity or verdict differs')
        require('case_id' not in summary or summary['case_id'] == row['case_id'],
                'Actual physical summary case differs')
    submitted = listed_json('authored_assets/manifest.json')
    require(len(submitted['runs']) == 4 and
            {r['run_id'] for r in submitted['runs']} == PRIORITY_RUNS,
            'Four canonical drawer/conveyor submissions required')
    for run in submitted['runs']:
        rid = run['run_id']
        require(rid in native, 'Published submission has no evaluation')
        require(run['case_id'] == native[rid]['case_id'] and
                run['arm'] == native[rid]['arm'] and
                run['directory'] == 'outputs/' + run['case_id'] + '/' + run['arm'],
                'Published submission identity or directory differs')
        for name, field in (('asset', 'submitted_scene_sha256'),
                            ('bindings', 'submitted_bindings_sha256')):
            relative = safe_name('authored_assets/' + run['directory'] + '/' + safe_name(run[name]))
            require(relative in listed, 'Missing published submission member')
            require(listed[relative]['sha256'] == native[rid][field],
                    'Published submission differs from evaluated bytes: ' + rid)
        require(run['archive_sha256'] == native[rid]['original_author_archive_sha256'],
                'Published submission archive differs')
    if deep:
        require(sys.flags.optimize == 0, 'Deep child verifiers require assertions enabled')
        for command in (
            [sys.executable, '-B', str(root / 'metrics/verify_results_metadata.py'),
             '--verify', str(root / 'metrics')],
            [sys.executable, '-B', str(root / 'physical_evidence/verify_physical_evidence.py'),
             '--verify', str(root / 'physical_evidence')],
            [sys.executable, '-B', str(root / 'authored_assets/verify.py'),
             str(root / 'authored_assets')],
            [sys.executable, '-B', str(root / 'protocol_audits/tools/verify_public.py'),
             '--bundle', str(root / 'protocol_audits')],
        ):
            subprocess.run(command, check=True)
    return {'status': 'PASS', 'files': len(listed), 'attempts': len(rows),
            'deep_verification': deep,
            'scope': 'Published-byte integrity and cross-record agreement; not a simulation replay.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--deep', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify(args.root.resolve(), args.deep), sort_keys=True))
