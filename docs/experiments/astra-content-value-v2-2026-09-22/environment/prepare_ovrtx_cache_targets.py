"""Add only an empty MDL generated-cache mount target; hash installed bytes.

The native library writes cache siblings beside a bundled shader cache. No
writable overlay is installed here; the namespace harness owns private mounts.
"""
import argparse
import hashlib
import json
from pathlib import Path
from export_namespace_tools import manifest, sha

TARGET='ovrtx-venv/lib/python3.12/site-packages/ovrtx/bin/mdl/omniverse_exts'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tools',type=Path,required=True)
    p.add_argument('--parent-receipt',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    a=p.parse_args()
    if a.receipt.exists():raise ValueError('Receipt is create-only')
    root=a.tools.resolve();parent=json.loads(a.parent_receipt.read_text())
    if not parent['passed']:raise ValueError('Unqualified parent')
    before=manifest(root)
    stable=lambda rows:{r['path']:r for r in rows if not r['path'].startswith('repo/.git/')}
    if stable(before)!=stable(parent['files']):raise ValueError('Installed files changed since parent')
    target=root/TARGET
    if not target.parent.is_dir() or target.is_symlink():raise ValueError('Invalid installed MDL target')
    existed=target.exists()
    if existed and (not target.is_dir() or any(target.iterdir())):raise ValueError('Expected absent or empty target')
    target.mkdir(mode=0o755,exist_ok=True)
    target.chmod(0o755)
    cache=root/'ovrtx-venv/lib/python3.12/site-packages/ovrtx/bin/cache'
    cache_files=manifest(cache)
    result=dict(parent)
    result.update(schema_version='namespace-tools-export.v3',parent_receipt_sha256=sha(a.parent_receipt),
      cache_target_preparation_sha256=sha(Path(__file__)),files=before,
      manifest_sha256=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
      generated_cache_mount_targets=[{'path':TARGET,'kind':'empty_directory','mode':'0755','created':not existed}],
      bundled_shader_cache={'path':str(cache.relative_to(root)),'files':len(cache_files),'bytes':sum(x.get('bytes',0) for x in cache_files),
        'inventory_sha256':hashlib.sha256(json.dumps(cache_files,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
        'top_level_entries':sorted(x.name for x in cache.iterdir())},
      installed_file_bytes_unchanged=True,
      cache_scope='Only an empty generated MDL directory was added. Runtime code and bundled shaders remain unchanged; qualified private namespace cache mounts are still required.')
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    print(json.dumps({'passed':True,'receipt_sha256':sha(a.receipt),'file_bytes_unchanged':True,'target':TARGET,'bundled_shader_cache':result['bundled_shader_cache']}))
if __name__=='__main__':main()
