"""Render three measured task poses through native usd-cli/OVRTX.

This is presentation of an existing trace, never a new acceptance run.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from pxr import Usd, UsdGeom
from content_agent_workflows.simready.asset_identity import build_asset_dependency_manifest


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-root', type=Path, required=True)
    parser.add_argument('--trial', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root, trial, out = [p.resolve() for p in (args.experiment_root, args.trial, args.output)]
    assert trial.is_relative_to(root / 'capstone')
    assert out.is_relative_to(root / 'capstone') and not out.exists()
    paths = [trial / name for name in ('request.json', 'trial_report.json', 'trace.jsonl', 'scene.usda', 'replay.usda')]
    before = {str(p.relative_to(root)): sha(p) for p in paths}
    dependency_before = build_asset_dependency_manifest(paths[4])
    request = json.loads(paths[0].read_text())
    report = json.loads(paths[1].read_text())
    rows = []
    with paths[2].open() as f:
        for line in f:
            row = json.loads(line)
            rows.append({k: row[k] for k in ('step', 'time_s', 'phase', 'drawer_pose', 'payload_pose', 'q_m')})
    assert rows and request['seed'] == report['seed']
    groups = [[r for r in rows if r['phase'] == p] for p in ('settle', 'hold_open', 'hold_closed')]
    assert all(groups)
    chosen = [groups[0][-1], groups[1][len(groups[1]) // 2], groups[2][-1]]
    stage = Usd.Stage.Open(str(paths[4]))
    assert stage
    checks = []
    for row in chosen:
        cache = UsdGeom.XformCache(Usd.TimeCode(row['step']))
        for path, key in ((request['drawer_body'], 'drawer_pose'), (request['payload_body'], 'payload_pose')):
            matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
            position = list(matrix.ExtractTranslation())
            quat = matrix.RemoveScaleShear().ExtractRotationQuat()
            actual = [*quat.GetImaginary(), quat.GetReal()]
            expected = row[key]
            error = math.sqrt(sum((a-b)**2 for a, b in zip(position, expected[:3])))
            dot = abs(sum(a*b for a, b in zip(actual, expected[3:])))
            dot /= math.sqrt(sum(a*a for a in actual) * sum(a*a for a in expected[3:]))
            assert error <= 1e-6 and abs(1-dot) <= 1e-6, (path, error, dot)
            checks.append({'step': row['step'], 'body': path, 'position_error_m': error, 'absolute_quaternion_dot': dot})
    out.mkdir(parents=True)
    receipt = {'schema_version': 2, 'script_sha256': sha(Path(__file__)), 'seed': request['seed'], 'task_status': report['status'],
               'scope': 'Presentation of recorded poses; no simulation, asset authoring, or acceptance decision.',
               'trace_time_note': 'Replay time code equals zero-based step; physical time_s equals (step+1)*dt.',
               'selected_measured_rows': chosen, 'pose_readback_checks': checks,
               'inputs_before': before, 'dependency_manifest_before': dependency_before,
               'render_requested': not args.verify_only, 'operations': []}
    env = dict(os.environ)
    repo = root / 'capstone/repo'
    env.update(PATH=str(repo / '.venv/bin') + ':' + env.get('PATH', ''),
               WU_OVRTX_VENV_DIR=str(root / 'ovrtx-venv'),
               USD_CLI_TEL_BACKENDS='file', USD_CLI_TEL_FILE=str(out / 'operations.jsonl'))
    cli = repo / '.venv/bin/usd-cli-tel'

    def command(name, argv):
        start = datetime.datetime.now(datetime.timezone.utc).isoformat()
        proc = subprocess.run([str(cli), '--json', *argv], cwd=out, env=env, text=True, capture_output=True)
        (out / (name + '.json')).write_text(proc.stdout)
        (out / (name + '.stderr')).write_text(proc.stderr)
        receipt['operations'].append({'name': name, 'arguments': argv, 'returncode': proc.returncode,
                                      'start_utc': start, 'end_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        if proc.returncode:
            raise RuntimeError(f'{name} failed; retained command output in {out}')

    try:
        if not args.verify_only:
            config = out / '.usd-cli/config.toml'
            config.parent.mkdir()
            config.write_text('[server]\nallowed_roots = ' + json.dumps([str(root / 'capstone'), str(root / 'assets/01_drawer')])
                              + '\nallowed_write_roots = ' + json.dumps([str(out)])
                              + '\n[render]\nrenderer = "ovrtx"\novrtx_auto_install = false\n')
            command('probe', ['render-probe'])
            command('render', ['render-frames', '--scene', str(paths[4]), '--frames', ','.join(str(r['step']) for r in chosen),
                               '--focus', '/Asset', '--res', '1024x1024', '--mode', 'fast', '--no-animate', '--output', str(out / 'frames')])
    finally:
        if not args.verify_only:
            try:
                command('server_stop', ['server', 'stop'])
            except RuntimeError:
                pass
        receipt['inputs_after'] = {str(p.relative_to(root)): sha(p) for p in paths}
        receipt['input_bytes_unchanged'] = receipt['inputs_after'] == before
        receipt['dependency_manifest_after'] = build_asset_dependency_manifest(paths[4])
        receipt['dependency_bytes_unchanged'] = receipt['dependency_manifest_after'] == dependency_before
        receipt['outputs'] = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file() and '.usd-cli' not in p.parts}
        (out / 'presentation_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
        assert receipt['input_bytes_unchanged'], 'A retained task input changed during presentation'
        assert receipt['dependency_bytes_unchanged'], 'A replay dependency changed during presentation'
    print(out / 'presentation_receipt.json')


if __name__ == '__main__':
    main()
