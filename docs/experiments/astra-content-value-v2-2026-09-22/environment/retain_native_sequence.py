"""Hash and verify completed synthetic-runtime evidence; never remove inputs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def inventory(root, names):
    root = Path(root)
    rows = []
    for name in names:
        assert '/' not in name and name not in {'.', '..'}
        directory = root / name
        assert (directory / 'supervisor.json').is_file()
        assert json.loads((directory / 'supervisor.json').read_text())['cgroup_populated'] == 0
        for path in sorted(directory.rglob('*')):
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows.append({'path': rel, 'type': 'symlink', 'target': os.readlink(path)})
            elif path.is_file():
                with path.open('rb') as stream:
                    rows.append({'path': rel, 'type': 'file', 'size': path.stat().st_size, 'sha256': digest(stream)})
    return {'schema_version': 'synthetic-native-sequence-retention.v1', 'files': rows}


def verify(archive, manifest, extraction):
    expected = {x['path']: x for x in json.loads(Path(manifest).read_text())['files']}
    seen = set()
    destination = Path(extraction)
    destination.mkdir(parents=True, exist_ok=False)
    extracted = []
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            path = Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            if member.isdir():
                continue
            assert member.name not in seen
            seen.add(member.name)
            row = expected[member.name]
            if member.issym():
                assert row['type'] == 'symlink' and row['target'] == member.linkname
                continue
            assert member.isfile() and row['type'] == 'file' and row['size'] == member.size
            with tf.extractfile(member) as stream:
                assert digest(stream) == row['sha256']
            # Keep every original byte in the archive. Unpack only inspectable evidence,
            # not generated caches or private daemon session state.
            if '.runtime-caches' not in path.parts and '.usd-cli' not in path.parts and '.cache' not in path.parts:
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                with tf.extractfile(member) as stream, target.open('xb') as output:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        output.write(block)
                extracted.append(member.name)
    assert seen == set(expected)
    with Path(archive).open('rb') as stream:
        archive_sha = digest(stream)
        os.fsync(stream.fileno())
    return {'schema_version': 'synthetic-native-sequence-retention-verification.v1', 'passed': True,
            'archive_sha256': archive_sha, 'archive_bytes': Path(archive).stat().st_size,
            'original_manifest_sha256': hashlib.sha256(Path(manifest).read_bytes()).hexdigest(),
            'members_verified': len(seen), 'extracted_evidence_paths': extracted,
            'all_original_bytes_retained': True, 'remote_inputs_deleted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='mode', required=True)
    item = commands.add_parser('inventory')
    item.add_argument('root')
    item.add_argument('names', nargs='+')
    item = commands.add_parser('verify')
    item.add_argument('archive')
    item.add_argument('manifest')
    item.add_argument('extraction')
    args = parser.parse_args()
    result = inventory(args.root, args.names) if args.mode == 'inventory' else verify(args.archive, args.manifest, args.extraction)
    print(json.dumps(result, indent=2, sort_keys=True))
