"""Controller: download, verify every archive member, then drain new scratch."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_archive(path, run_id, receipt):
    assert path.stat().st_size == receipt['archive_bytes']
    assert digest(path) == receipt['archive_sha256']
    with tarfile.open(path, 'r:gz') as tar:
        members = tar.getmembers()
        assert len({x.name for x in members}) == len(members)
        for member in members:
            p = PurePosixPath(member.name)
            assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == run_id
            assert member.isfile() or member.isdir() or member.issym()
        manifest_name = run_id+'/private/retention_manifest.json'
        manifest_bytes = tar.extractfile(manifest_name).read()
        assert hashlib.sha256(manifest_bytes).hexdigest() == receipt['manifest_sha256']
        manifest = json.loads(manifest_bytes)
        assert manifest['run_id'] == run_id
        expected = {run_id+'/'+x['path']: x for x in manifest['files'] if x['kind'] != 'special_excluded'}
        assert set(expected) | {manifest_name} == {x.name for x in members}
        for member in members:
            if member.name == manifest_name:
                continue
            row = expected[member.name]
            if row['kind'] == 'directory':
                assert member.isdir()
            elif row['kind'] == 'symlink':
                assert member.issym() and member.linkname == row['target']
            else:
                assert member.isfile() and member.size == row['bytes']
                h = hashlib.sha256()
                f = tar.extractfile(member)
                for data in iter(lambda: f.read(1024*1024), b''):
                    h.update(data)
                assert h.hexdigest() == row['sha256']
    return dict(receipt, local_member_verification=True,
                verified_member_count=len(members), local_archive=str(path.resolve()))


def retain(cfg, job):
    host = cfg['hosts'][next(x['host'] for x in cfg['protocol']['lanes'] if x['id'] == job['lane_id'])]
    run_id = job['run_id']
    destination = Path(cfg['retained_archives'])/run_id
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    base = ['tsh', 'ssh', '--proxy='+cfg['teleport_proxy'], 'horde@'+host['hostname']]
    def command(mode):
        return [*base, 'sudo -n '+shlex.join([host['python'], host['retention_supervisor'],
                '--config', host['worker_config'], '--run-id', run_id, '--mode', mode])]
    partial = destination/'complete.partial.tgz'
    with partial.open('xb') as out, (destination/'export.stderr.log').open('x') as err:
        p = subprocess.run(command('export'), stdout=out, stderr=err, timeout=1800)
        out.flush()
        os.fsync(out.fileno())
    if p.returncode:
        raise RuntimeError('Retention export failed; preserve remote run and stop lane')
    r = subprocess.run(command('receipt'), text=True, capture_output=True, timeout=60, check=True)
    receipt = json.loads(r.stdout)
    proof = verify_archive(partial, run_id, receipt)
    final = destination/'complete.tgz'
    partial.rename(final)
    proof['local_archive'] = str(final.resolve())
    with (destination/'verification.json').open('x') as f:
        f.write(json.dumps(proof, indent=2)+'\n')
        f.flush()
        os.fsync(f.fileno())
    for path in (destination, destination.parent):
        directory = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    r = subprocess.run(command('cleanup'), input=json.dumps(proof), text=True,
                       capture_output=True, timeout=120, check=True)
    cleanup = json.loads(r.stdout)
    (destination/'worker_cleanup.json').write_text(json.dumps(cleanup, indent=2)+'\n')
    return {'verified': True, 'run_id': run_id, 'archive_sha256': proof['archive_sha256'],
            'archive_bytes': proof['archive_bytes'], 'local_archive': str(final.resolve())}
