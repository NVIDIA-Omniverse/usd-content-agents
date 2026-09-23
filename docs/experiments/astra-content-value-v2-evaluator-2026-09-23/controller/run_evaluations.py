"""Post-author batch evaluation and verified retention; never execute author code.

The local controller first proves that all twenty authors are reaped and retained.
Each of four sequential lane queues then stages only workspace data, invokes the
frozen evaluator, streams complete results home, verifies/fsyncs them, and finally
cleans the exact evaluated run. A failure stops that lane without automatic reuse.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import sqlite3
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'harness'))
from common import identifier, object_hash, read, require, sha256, validate_protocol, write_new
from isolation import check_host_owned, check_limits, kill_and_reap, populated
from evaluate_run import execute, pack_workspace, safe_members
from retain_run import verify_archive

MAX_STAGE_BYTES = 9 * 2**30


def durable_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    sync_directory(path.parent)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def archive_json(archive, member_name):
    with tarfile.open(archive, 'r:gz') as tar:
        matches = [m for m in tar.getmembers() if m.name == member_name]
        require(len(matches) == 1 and matches[0].isfile(), 'Required retained JSON is not one regular file')
        require(matches[0].size <= 16 * 2**20, 'Oversized retained declaration/receipt')
        value = json.load(tar.extractfile(matches[0]))
        require(isinstance(value, dict), 'Retained JSON must be an object')
        return value


def verify_freeze(path, root=ROOT):
    freeze = read(path)
    require(freeze.get('frozen') is True and freeze.get('status') == 'PASS', 'Complete protocol/evaluator freeze required')
    require(any(x['path'].startswith('evaluator/') for x in freeze['files']), 'Freeze does not bind evaluator')
    for row in freeze['files']:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe frozen path')
        require(sha256(root / rel) == row['sha256'], 'Frozen file changed: ' + row['path'])
    return freeze


def author_gate(cfg, jobs, protocol, jobs_file):
    """No remote calls or writes: all original assignments must have terminal proof."""
    require(__debug__, 'Archive verification requires Python assertions enabled')
    from run_queue import validate_jobs
    validate_jobs(protocol, jobs)
    require(len(jobs) == 20 and len({j['case_id'] for j in jobs}) == 10, 'All twenty original author attempts required')
    require({j['case_id'] for j in jobs} == set(protocol['lane_assignment']), 'Assigned case set differs')
    require(all(j['lane_id'] == protocol['lane_assignment'][j['case_id']] for j in jobs), 'Assigned lane differs')
    dispatch = Path(cfg['output'])
    queue = read(dispatch / 'queue_result.json')
    require(queue['protocol_sha256'] == sha256(cfg['protocol_file']) and queue['jobs_sha256'] == sha256(jobs_file), 'Queue completion digest differs')
    completed = [x['run_id'] for lane in queue['lanes'] for x in lane]
    require(len(queue['lanes']) == 4 and len(completed) == len(set(completed)) == 20
            and set(completed) == {j['run_id'] for j in jobs}, 'Full trusted author queue is not complete')
    connection = sqlite3.connect(Path(cfg['database']).resolve().as_uri() + '?mode=ro', uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = {r['id']: dict(r) for r in connection.execute('SELECT * FROM runs')}
        stored_protocol = connection.execute("SELECT value FROM settings WHERE key='protocol'").fetchone()
        require(stored_protocol and json.loads(stored_protocol[0]) == protocol, 'Ledger protocol differs')
    finally: connection.close()
    require(set(rows) == {j['run_id'] for j in jobs}, 'Unexpected/missing author ledger runs')
    prepared = []
    for job in jobs:
        run_id = identifier(job['run_id']); record = rows[run_id]
        require(record['status'] == 'reaped' and record['lane'] == job['lane_id'], 'Author is not reaped on its assigned lane')
        require(record['spec_sha'] == object_hash(job), 'Author ledger job specification differs')
        receipt = read(dispatch / run_id / 'worker_reap.json')
        require(receipt.get('run_id') == run_id and receipt.get('lease') == record['lease']
                and receipt.get('cgroup_populated') == 0 and receipt.get('namespace_init_exited') is True,
                'Missing trusted author reap proof')
        require(record['reap_sha'] == object_hash(receipt), 'Ledger/reap receipt differs')
        directory = Path(cfg['retained_archives']) / run_id
        archive = directory / 'complete.tgz'
        proof = read(directory / 'verification.json')
        require(proof.get('run_id') == run_id and proof.get('local_member_verification') is True, 'Author retention proof missing')
        verified = verify_archive(archive, run_id, proof)
        retained = read(dispatch / run_id / 'retention.json')
        require(retained.get('verified') is True and retained.get('run_id') == run_id
                and retained['archive_sha256'] == verified['archive_sha256'], 'Dispatch retention differs')
        cleanup = read(directory / 'worker_cleanup.json')
        require(cleanup.get('run_id') == run_id and cleanup.get('removed_only') == ['workspace', 'home']
                and cleanup.get('verified_archive_sha256') == verified['archive_sha256'], 'Author scratch cleanup not verified')
        declaration = archive_json(archive, run_id + '/private/author_delivery.json')
        prepared.append(dict(job, archive=str(archive.resolve()), proof=verified, declaration=declaration))
    return prepared


def evaluation_request(job, batch, lane, remote, invocations, archive_sha):
    run_id = identifier('eval_' + batch + '_' + job['run_id'])
    case = identifier(job['case_id'])
    source = Path(job['source'])
    ancestor = Path(remote) / 'assets' / case
    require(source.is_relative_to(ancestor), 'Author source outside original case ancestor')
    original = source if case == '01_drawer' else Path(next(x['source_root'] for x in invocations['cases'] if x['case_id'] == case))
    require(original.is_relative_to(ancestor), 'Evaluator original source outside case ancestor')
    return {'run_id': run_id, 'author_run_id': job['run_id'], 'case_id': case, 'arm': job['arm'],
            'lane': lane, 'source': str(source), 'original': str(original), 'declaration': job['declaration'],
            'author_archive_sha256': job['proof']['archive_sha256'],
            'workspace_archive': str(Path(remote) / 'evaluation_staging' / run_id / 'payload/workspace.tgz'),
            'workspace_archive_sha256': archive_sha, 'freeze_file': remote + '/protocol/freeze.json',
            'invocations_file': remote + '/evaluator/invocations.json'}


def evaluator_config(template, remote, batch):
    return {'rootfs': remote + '/evaluator_runtime/rootfs', 'tools': template['tools'],
            'system_mounts': template['system_mounts'], 'lanes': template['lanes'],
            'runs_root': remote + '/evaluation_runs/' + batch}


def verify_original_inputs(request):
    """Bind source aliases to frozen originals, never an author-provided path."""
    rows = read(ROOT / 'protocol/author_source_closures.json')['cases']
    source = next(x for x in rows if x['case_id'] == request['case_id'])
    directory = ROOT / source['source_root']
    require(str(directory) == request['source'], 'Source mount differs from frozen original closure')
    for row in source['files']:
        relative = Path(row['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Unsafe original source path')
        path = directory / relative
        require(not path.is_symlink() and path.stat().st_size == row['size']
                and sha256(path) == row['sha256'], 'Original source bytes changed')
    expected = str(directory) if request['case_id'] == '01_drawer' else next(
        x['source_root'] for x in read(request['invocations_file'])['cases'] if x['case_id'] == request['case_id'])
    require(request['original'] == expected, 'Original source ancestor differs from frozen invocation')


def worker_load(stage):
    require(os.geteuid() == 0, 'Trusted root worker required')
    stage = Path(stage)
    require(stage.parent == ROOT / 'evaluation_staging' and not stage.is_symlink(), 'Unexpected staging root')
    identifier(stage.name)
    for name in ('config.json', 'request.json', 'stage.json'): check_host_owned(stage / name)
    cfg, request, info = (read(stage / name) for name in ('config.json', 'request.json', 'stage.json'))
    require(request['run_id'] == stage.name and cfg['rootfs'] == str(ROOT / 'evaluator_runtime/rootfs'), 'Evaluation identity/rootfs differs')
    require(Path(request['workspace_archive']) == stage / 'payload/workspace.tgz', 'Unexpected archive path')
    require(Path(cfg['runs_root']).parent == ROOT / 'evaluation_runs', 'Unexpected evaluation output root')
    group = Path(cfg['lanes'][request['lane']['id']]['cgroup'])
    check_limits(group, request['lane'])
    lease = ROOT / 'evaluation_lane_leases' / identifier(request['lane']['id']) / 'lease.json'
    check_host_owned(lease)
    require(read(lease)['run_id'] == request['run_id'], 'Evaluation lane reservation differs')
    return stage, cfg, request, info, group


def enter_lane(group, lane):
    (group / 'cgroup.procs').write_text(str(os.getpid()))
    os.sched_setaffinity(0, lane['cpu_affinity'])
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def require_root_directory(path):
    require(path.is_dir() and not path.is_symlink() and path.stat().st_uid == 0, 'Unexpected evaluated run root')


def worker_prepare(payload):
    require(os.geteuid() == 0, 'Trusted root worker required')
    cfg, request = payload['config'], payload['request']
    require(cfg['rootfs'] == str(ROOT / 'evaluator_runtime/rootfs'), 'Wrong evaluator rootfs')
    require(Path(cfg['runs_root']).parent == ROOT / 'evaluation_runs', 'Unexpected output ancestor')
    run_id = identifier(request['run_id']); group = Path(cfg['lanes'][request['lane']['id']]['cgroup'])
    require(populated(group) == 0, 'Evaluation lane occupied')
    limits = check_limits(group, request['lane'])
    require(request['lane']['memory_bytes'] <= 32 * 2**30, 'Evaluation RAM cap exceeds budget')
    size = payload['archive_bytes']
    require(type(size) is int and 0 < size <= MAX_STAGE_BYTES, 'Workspace compressed archive exceeds staging cap')
    enter_lane(group, request['lane'])
    require(sha256(request['freeze_file']) == request['freeze_sha256'], 'Controller/worker evaluator freeze differs')
    verify_freeze(request['freeze_file'])
    verify_original_inputs(request)
    stage = ROOT / 'evaluation_staging' / run_id
    require(Path(request['workspace_archive']) == stage / 'payload/workspace.tgz', 'Unexpected staging archive')
    # Atomic root-owned reservation persists across upload/execute/export SSH
    # processes. A failed or disconnected operation never releases it by time.
    lease = ROOT / 'evaluation_lane_leases' / identifier(request['lane']['id'])
    lease.mkdir(parents=True, mode=0o700, exist_ok=False)
    durable_json(lease / 'lease.json', {'run_id': run_id, 'lane_id': request['lane']['id']})
    stage.mkdir(parents=True, mode=0o700, exist_ok=False)
    durable_json(stage / 'config.json', cfg); durable_json(stage / 'request.json', request)
    payload_root = stage / 'payload'; payload_root.mkdir(mode=0o700)
    capacity = ((size + 65535) // 65536 + 1) * 65536
    subprocess.run(['/usr/bin/mount', '-t', 'tmpfs', '-o', f'size={capacity},mode=0700,nosuid,nodev,noexec',
                    'astra-v2-evaluation-input', str(payload_root)], check=True)
    mounts = [line for line in Path('/proc/self/mountinfo').read_text().splitlines() if line.split()[4] == str(payload_root)]
    require(len(mounts) == 1 and ' - tmpfs ' in mounts[0], 'Staging tmpfs not established')
    durable_json(stage / 'stage.json', {'run_id': run_id, 'archive_bytes': size, 'capacity_bytes': capacity,
                 'mountinfo': mounts[0], 'limits': limits, 'pages_charged_to_evaluator_lane': True})
    return {'run_id': run_id, 'prepared': True, 'staging_kind': 'bounded_tmpfs'}


def worker_terminal(stage):
    stage, cfg, request, info, group = worker_load(stage)
    require(populated(group) == 0, 'Evaluation lane is still populated')
    out = Path(cfg['runs_root']) / request['run_id']
    require_root_directory(out)
    receipt = read(out / 'private/batch_evaluation.json')
    require(receipt['run_id'] == request['run_id'] and receipt['lane_id'] == request['lane']['id']
            and receipt['cgroup_populated'] == 0 and receipt['terminal'] is True,
            'Trusted evaluator terminal receipt missing')
    require(receipt['evaluation_sha256'] == sha256(out / 'private/evaluation.json'), 'Evaluation receipt changed')
    require(receipt['request_sha256'] == sha256(stage / 'request.json'), 'Evaluation request changed')
    return stage, cfg, request, info, group, out, receipt


def worker(mode, stage=None):
    if mode == 'prepare': return worker_prepare(json.load(sys.stdin))
    stage, cfg, request, info, group = worker_load(stage)
    if mode == 'upload':
        require(populated(group) == 0, 'Evaluation lane occupied before upload')
        enter_lane(group, request['lane'])
        archive = Path(request['workspace_archive']); digest = hashlib.sha256(); count = 0
        with archive.open('xb') as target:
            while chunk := sys.stdin.buffer.read(1024 * 1024):
                count += len(chunk); require(count <= info['archive_bytes'], 'Upload exceeds declared length')
                digest.update(chunk); target.write(chunk)
            target.flush(); os.fsync(target.fileno())
        require(count == info['archive_bytes'] and digest.hexdigest() == request['workspace_archive_sha256'], 'Workspace transfer mismatch')
        archive.chmod(0o400)
        return {'uploaded': True, 'bytes': count, 'sha256': digest.hexdigest()}
    if mode == 'execute':
        require(populated(group) == 0, 'Evaluation lane occupied before execution')
        out = Path(cfg['runs_root']) / request['run_id']
        require(not out.exists(), 'Fresh evaluation ID required')
        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        started = time.time()
        try:
            result = execute(cfg, request)
        except Exception as exc:
            # Preserve setup/restore/runtime errors as infrastructure evidence;
            # never convert a failed supervisor into an authored-artifact FAIL.
            kill_and_reap(group)
            out.mkdir(exist_ok=True, mode=0o755)
            for name in ('workspace', 'home', 'private'): (out / name).mkdir(exist_ok=True, mode=0o700)
            result = {'run_id': request['run_id'], 'case_id': request['case_id'], 'status': 'INCONCLUSIVE',
                      'exception_type': type(exc).__name__, 'infrastructure_error': True, 'cgroup_populated': populated(group)}
            if not (out / 'private/evaluation.json').exists(): durable_json(out / 'private/evaluation.json', result)
        require(populated(group) == 0, 'Evaluator not reaped; preserve lane')
        durable_json(out / 'private/batch_request.json', request)
        durable_json(out / 'private/batch_config.json', cfg)
        receipt = {'run_id': request['run_id'], 'author_run_id': request['author_run_id'], 'case_id': request['case_id'],
                   'lane_id': request['lane']['id'], 'terminal': True, 'cgroup_populated': 0, 'status': result['status'],
                   'started_unix': started, 'completed_unix': time.time(), 'evaluation_sha256': sha256(out / 'private/evaluation.json'),
                   'request_sha256': sha256(stage / 'request.json'), 'supervisor_sha256': sha256(__file__)}
        durable_json(out / 'private/batch_evaluation.json', receipt)
        return receipt
    stage, cfg, request, info, group, out, receipt = worker_terminal(stage)
    from worker_retention import export, cleanup
    if mode == 'export':
        enter_lane(group, request['lane'])
        export(out, request['run_id'])
        return None  # stdout is the complete binary archive.
    if mode == 'receipt': return read(out / 'private/archive_receipt.json')
    if mode == 'cleanup':
        proof = json.load(sys.stdin)
        # Validate exact immutable input mount before any destructive action.
        payload_root = stage / 'payload'
        require(info['mountinfo'] in Path('/proc/self/mountinfo').read_text().splitlines()
                and os.path.ismount(payload_root), 'Staging mount changed; preserve all outputs')
        require(sha256(request['workspace_archive']) == request['workspace_archive_sha256'], 'Staged input changed')
        cleanup(out, request['run_id'], proof)  # Emits the existing exact cleanup receipt.
        subprocess.run(['/usr/bin/umount', '--', str(payload_root)], check=True)
        payload_root.rmdir()
        durable_json(stage / 'cleanup.json', {'run_id': request['run_id'], 'unmounted_exact_path': str(payload_root),
                     'verified_archive_sha256': proof['archive_sha256'], 'configs_and_receipts_retained': True})
        require(populated(group) == 0, 'Evaluation lane repopulated during cleanup')
        lease = ROOT / 'evaluation_lane_leases' / request['lane']['id']
        require(read(lease / 'lease.json')['run_id'] == request['run_id'], 'Lane reservation changed during cleanup')
        (lease / 'lease.json').unlink(); lease.rmdir(); sync_directory(lease.parent)
        return None
    raise ValueError('Unknown trusted worker operation')


def retain_evaluation(remote_call, directory, run_id, terminal):
    require(__debug__, 'Archive verification requires Python assertions enabled')
    directory.mkdir(mode=0o700)
    partial = directory / 'complete.partial.tgz'
    with partial.open('xb') as stream, (directory / 'export.stderr.log').open('x') as errors:
        remote_call('export', stdout=stream, stderr=errors, timeout=1800, check=True)
        stream.flush(); os.fsync(stream.fileno())
    receipt = json.loads(remote_call('receipt', capture_output=True, text=True, timeout=60, check=True).stdout)
    proof = verify_archive(partial, run_id, receipt)
    require(archive_json(partial, run_id + '/private/batch_evaluation.json') == terminal, 'Retained evaluation differs from trusted terminal receipt')
    final = directory / 'complete.tgz'; partial.rename(final)
    proof['local_archive'] = str(final.resolve())
    durable_json(directory / 'verification.json', proof)
    sync_directory(directory); sync_directory(directory.parent)
    cleaned = remote_call('cleanup', input=json.dumps(proof), capture_output=True, text=True, timeout=180, check=True)
    cleanup_receipt = json.loads(cleaned.stdout)
    require(cleanup_receipt['run_id'] == run_id and cleanup_receipt['verified_archive_sha256'] == proof['archive_sha256'], 'Evaluation cleanup identity differs')
    durable_json(directory / 'worker_cleanup.json', cleanup_receipt)
    return {'verified': True, 'run_id': run_id, 'archive_sha256': proof['archive_sha256'],
            'archive_bytes': proof['archive_bytes'], 'local_archive': str(final.resolve()), 'status': terminal['status']}


def launch(config_file, jobs_file, output, batch, validate_only=False):
    identifier(batch); require(len(batch) <= 16, 'Keep batch identifier at most16 characters')
    cfg = read(config_file); protocol = validate_protocol(read(cfg['protocol_file']))
    jobs = read(jobs_file)['jobs']; verify_freeze(ROOT / 'protocol/freeze.json')
    prepared = author_gate(cfg, jobs, protocol, jobs_file)
    if validate_only: return {'author_gate_passed': True, 'author_runs': len(prepared), 'launched': False}
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False, mode=0o700)
    remote = protocol['remote_experiment_root']; invocations = read(ROOT / 'evaluator/invocations.json')
    durable_json(output / 'batch_gate.json', {'all_twenty_authors_reaped_and_retained': True,
                 'jobs_sha256': sha256(jobs_file), 'protocol_sha256': sha256(cfg['protocol_file']),
                 'freeze_sha256': sha256(ROOT / 'protocol/freeze.json'),
                 'archives': {j['run_id']: j['proof']['archive_sha256'] for j in prepared}})

    def lane_queue(lane):
        host = cfg['hosts'][lane['host']]; reports = []
        base = ['tsh', 'ssh', '--proxy=' + cfg['teleport_proxy'], 'horde@' + host['hostname']]
        script = remote + '/controller/run_evaluations.py'
        # This trusted config contains mount/resource paths only, never credentials.
        try:
            fetched = subprocess.run([*base, 'sudo -n ' + shlex.join(['/bin/cat', host['worker_config']])],
                                     capture_output=True, text=True, check=True, timeout=60)
            worker_cfg = evaluator_config(json.loads(fetched.stdout), remote, batch)
        except Exception as exc:
            failure = {'lane_id': lane['id'], 'evaluations': [], 'exception_type': type(exc).__name__, 'lane_reusable': False}
            durable_json(output / ('lane_failure_' + lane['id'] + '.json'), failure)
            return failure
        for job in (j for j in prepared if j['lane_id'] == lane['id']):
            directory = output / job['run_id']; directory.mkdir(mode=0o700)
            try:
                workspace = directory / 'workspace.tgz'
                packed = pack_workspace(job['archive'], job['proof'], job['run_id'], workspace)
                with tarfile.open(workspace, 'r:gz') as tar: safe_members(tar)
                require(workspace.stat().st_size <= MAX_STAGE_BYTES, 'Workspace archive exceeds bounded staging')
                durable_json(directory / 'workspace_proof.json', packed)
                request = evaluation_request(job, batch, lane, remote, invocations, packed['workspace_archive_sha256'])
                request['freeze_sha256'] = sha256(ROOT / 'protocol/freeze.json')
                durable_json(directory / 'request.json', request); durable_json(directory / 'config.json', worker_cfg)
                stage = remote + '/evaluation_staging/' + request['run_id']
                def call(mode, **kwargs):
                    command = ['sudo', '-n', host['python'], '-B', script, '--worker-mode', mode]
                    if mode != 'prepare': command += ['--stage', stage]
                    return subprocess.run([*base, shlex.join(command)], **kwargs)
                payload = {'config': worker_cfg, 'request': request, 'archive_bytes': workspace.stat().st_size}
                call('prepare', input=json.dumps(payload), capture_output=True, text=True, check=True, timeout=600)
                with workspace.open('rb') as stream:
                    call('upload', stdin=stream, capture_output=True, check=True, timeout=1800)
                response = call('execute', capture_output=True, text=True, check=True, timeout=10800)
                terminal = json.loads(response.stdout)
                require(terminal['run_id'] == request['run_id'] and terminal['lane_id'] == lane['id']
                        and terminal['terminal'] is True and terminal['cgroup_populated'] == 0, 'Trusted evaluation reap missing')
                durable_json(directory / 'terminal.json', terminal)
                retained = retain_evaluation(call, directory / 'retained', request['run_id'], terminal)
                durable_json(directory / 'retention.json', retained)
                reports.append(dict(retained, author_run_id=job['run_id'], case_id=job['case_id'], arm=job['arm']))
            except Exception as exc:
                durable_json(directory / 'controller_failure.json', {'author_run_id': job['run_id'],
                             'exception_type': type(exc).__name__, 'lane_reusable': False,
                             'instruction': 'Preserve exact staging and evaluation outputs. Inspect/reap/retain before any explicit recovery.'})
                break
        return {'lane_id': lane['id'], 'evaluations': reports}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        lanes = list(pool.map(lane_queue, protocol['lanes']))
    result = {'batch': batch, 'lanes': lanes, 'complete': sum(len(x['evaluations']) for x in lanes) == 20,
              'scope': 'Independent frozen evaluator only; no author scripts executed.'}
    durable_json(output / 'evaluation_queue_result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config'); parser.add_argument('--jobs'); parser.add_argument('--output'); parser.add_argument('--batch')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--worker-mode', choices=['prepare', 'upload', 'execute', 'export', 'receipt', 'cleanup'])
    parser.add_argument('--stage')
    args = parser.parse_args()
    value = worker(args.worker_mode, args.stage) if args.worker_mode else launch(args.config, args.jobs, args.output, args.batch, args.validate_only)
    if value is not None: print(json.dumps(value, sort_keys=True))
