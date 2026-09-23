"""Restore only the hash-verified original source closure into a fresh worker."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path('/opt/astra-content-value-20260922-rerun')
manifest = json.loads((ROOT / 'original_sources_manifest.json').read_text())
archive = ROOT / 'original_sources.tgz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest['archive_sha256']
expected = {x['path']: x for x in manifest['files']}
with tarfile.open(archive) as tar:
    members = tar.getmembers()
    assert len(members) == len(expected) and {m.name for m in members} == set(expected)
    for member in members:
        path = Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
        content = tar.extractfile(member).read()
        record = expected[member.name]
        assert len(content) == record['bytes']
        assert hashlib.sha256(content).hexdigest() == record['sha256']
        target = ROOT / path
        target.parent.mkdir(parents=True, exist_ok=True)
        assert not target.is_symlink()
        if target.exists():
            assert target.read_bytes() == content
        else:
            with target.open('xb') as stream:
                stream.write(content)
for rel, record in expected.items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == record['sha256']
receipt = dict(status='PASS', files_verified=len(expected),
               manifest_sha256=hashlib.sha256((ROOT / 'original_sources_manifest.json').read_bytes()).hexdigest(),
               archive_sha256=manifest['archive_sha256'])
out = ROOT / 'evidence' / 'original_sources_restored.json'
out.parent.mkdir(parents=True, exist_ok=True)
with out.open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt))
# Only this new transport copy is expendable after verified restoration. The
# original source files and the local retained archive are not deleted.
archive.unlink()
