"""Trusted streaming retention after author reap; no evaluator or model calls."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'harness'))
from common import identifier, sha256, read, write_new, require
from isolation import check_host_owned, populated


def inventory(root):
    rows = []
    skipped = {'private/retention_manifest.json', 'private/archive_receipt.json',
               'private/retention_cleanup.json'}
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            p = Path(parent)/name
            rel = p.relative_to(root).as_posix()
            if rel in skipped:
                continue
            s = p.lstat()
            row = {'path': rel, 'mode': stat.S_IMODE(s.st_mode)}
            if stat.S_ISLNK(s.st_mode):
                row.update(kind='symlink', target=os.readlink(p))
            elif stat.S_ISREG(s.st_mode):
                row.update(kind='file', bytes=s.st_size, sha256=sha256(p))
            elif stat.S_ISDIR(s.st_mode):
                row.update(kind='directory')
            else:
                row.update(kind='special_excluded', special_mode=stat.S_IFMT(s.st_mode))
            rows.append(row)
    return sorted(rows, key=lambda x: x['path'])


class HashedWriter:
    def __init__(self, out):
        self.out, self.hash, self.bytes = out, hashlib.sha256(), 0

    def write(self, data):
        self.hash.update(data)
        self.bytes += len(data)
        return self.out.write(data)

    def flush(self):
        return self.out.flush()


def export(root, run_id):
    rows = inventory(root)
    manifest = {'schema_version': 'retention.v2', 'run_id': run_id, 'files': rows,
                'scope': 'Complete regular-file bytes and link metadata; sockets/devices excluded without following them.'}
    manifest_path = root/'private/retention_manifest.json'
    if manifest_path.exists():
        require(read(manifest_path) == manifest, 'Run changed after previous export')
    else:
        write_new(manifest_path, manifest)
    rows.append({'path': 'private/retention_manifest.json', 'kind': 'file', 'mode': 0o600,
                 'bytes': manifest_path.stat().st_size, 'sha256': sha256(manifest_path)})
    writer = HashedWriter(sys.stdout.buffer)
    with gzip.GzipFile(fileobj=writer, mode='wb', mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode='w|') as tar:
            for row in rows:
                if row['kind'] == 'special_excluded':
                    continue
                p = root/row['path']
                info = tarfile.TarInfo(run_id+'/'+row['path'])
                info.mode = row['mode']
                info.mtime = 0
                if row['kind'] == 'directory':
                    info.type = tarfile.DIRTYPE
                    tar.addfile(info)
                elif row['kind'] == 'symlink':
                    info.type = tarfile.SYMTYPE
                    info.linkname = row['target']
                    tar.addfile(info)
                else:
                    require(not p.is_symlink() and sha256(p) == row['sha256'], 'File changed during retention')
                    info.size = row['bytes']
                    with p.open('rb') as f:
                        tar.addfile(info, f)
    writer.flush()
    receipt = {'run_id': run_id, 'archive_sha256': writer.hash.hexdigest(), 'archive_bytes': writer.bytes,
               'manifest_sha256': sha256(manifest_path), 'export_complete': True}
    receipt_path = root/'private/archive_receipt.json'
    if receipt_path.exists():
        require(read(receipt_path) == receipt, 'Re-export changed retained bytes')
    else:
        write_new(receipt_path, receipt)


def cleanup(root, run_id, proof):
    receipt = read(root/'private/archive_receipt.json')
    require(proof.get('local_member_verification') is True and proof.get('run_id') == run_id,
            'Controller verification required')
    for key in ('archive_sha256', 'manifest_sha256', 'archive_bytes'):
        require(proof.get(key) == receipt[key], 'Controller/archive receipt mismatch')
    if (root/'private/retention_cleanup.json').exists():
        print(json.dumps(read(root/'private/retention_cleanup.json')))
        return
    manifest = read(root/'private/retention_manifest.json')
    require(inventory(root) == manifest['files'], 'Run changed after retention; preserve it')
    mounts = {}
    if Path('/proc/self/mountinfo').exists():
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            before, after = line.split(' - ', 1)
            target = before.split()[4]
            mounts.setdefault(target, []).append(after.split()[0])
    temporary_mounts = []
    storage_path = root/'private/storage.json'
    recorded = {}
    if storage_path.exists():
        volumes = read(storage_path)['volumes']
        require(len(volumes) == 2, 'Unexpected recorded storage volume count')
        recorded = {x['path']: x for x in volumes}
        require(set(recorded) == {str(root/'workspace'), str(root/'home')},
                'Recorded storage paths differ; preserve outputs')
    for name in ('workspace', 'home'):
        p = root/name
        require(p.is_dir() and not p.is_symlink(), 'Unexpected author directory')
        if os.path.ismount(p):
            require(mounts.get(str(p)) == ['tmpfs'], 'Unexpected mount; preserve outputs')
            if recorded:
                volume = recorded[str(p)]
                require(volume['kind'] == 'tmpfs' and volume['mountinfo'] in
                        Path('/proc/self/mountinfo').read_text().splitlines(),
                        'Recorded storage mount changed; preserve outputs')
            temporary_mounts.append(p)
        else:
            require(not recorded, 'Recorded tmpfs is no longer mounted; preserve outputs')
    # No source, evaluator, environment, other run or private receipt is removed.
    for name in ('workspace', 'home'):
        p = root/name
        if p in temporary_mounts:
            # Only the exact two verified per-run tmpfs mounts may be unmounted.
            # No recursive/parent unmount, external target, or source deletion.
            subprocess.run(['umount', '--', str(p)], check=True)
            p.rmdir()
        else:
            shutil.rmtree(p)
    result = {'run_id': run_id, 'removed_only': ['workspace', 'home'],
              'verified_archive_sha256': receipt['archive_sha256'], 'private_receipts_retained': True,
              'unmounted_exact_paths': [str(x) for x in temporary_mounts]}
    write_new(root/'private/retention_cleanup.json', result)
    print(json.dumps(result))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-id', required=True)
    p.add_argument('--mode', choices=['export', 'receipt', 'cleanup'], required=True)
    a = p.parse_args()
    require(os.geteuid() == 0, 'Trusted root supervisor required')
    check_host_owned(a.config)
    cfg = read(a.config)
    run_id = identifier(a.run_id)
    root = Path(cfg['runs_root'])/run_id
    require(root.is_dir() and not root.is_symlink() and root.stat().st_uid == 0, 'Unexpected run root')
    launch = read(root/'private/launch.json')
    cgroup = cfg['lanes'][launch['lane_id']]['cgroup']
    require(populated(cgroup) == 0, 'Author lane must be empty before retention')
    if a.mode == 'receipt':
        print(json.dumps(read(root/'private/archive_receipt.json')))
    elif a.mode == 'export':
        # This process exits after streaming; CPU/RAM charged to its freed lane.
        (Path(cgroup)/'cgroup.procs').write_text(str(os.getpid()))
        export(root, run_id)
    else:
        cleanup(root, run_id, json.load(sys.stdin))


if __name__ == '__main__':
    main()
